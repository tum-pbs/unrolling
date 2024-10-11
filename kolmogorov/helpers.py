import tensorflow as tf
from diffpiso import * 

@tf.custom_gradient
def circular_padding_differentiable(tensor):
    result = math.pad(tensor, ((0, 0), (0, 1), (0, 1), (0, 0)), mode='circular')
    def grad(dtensor):
        return dtensor[:,:-1,:-1,:]
    return result, grad


@tf.custom_gradient
def circular_slicing_differentiable(tensor):
    result = tensor[:,:-1,:-1]
    def grad(dtensor):
        components = []
        for i in range(math.spatial_rank(dtensor)):
            slicing = [slice(None), ] + [slice(0,1) if d == i else slice(None) for d in range(math.spatial_rank(dtensor))] + [slice(i, i + 1), ]
            components.append(math.concat([dtensor[..., slice(i, i+1)], dtensor[slicing]], axis=i+1))
        return stack_staggered_components(components)
    return result, grad


def supervised_loss_centered(prediction_data, label_data):
    return tf.nn.l2_loss(prediction_data - label_data)


def resnet_variable_count(resnet_features):
    # VARIABLE COUNT
    count = 0
    count += (3*3+1)*3*resnet_features[0] + 3* resnet_features[0] +(3*3)*resnet_features[0]**2
    for i in range(len(resnet_features)-1):
        count += (3*3+1)*resnet_features[i]*resnet_features[i+1] + 3* resnet_features[i+1] +(3*3)*resnet_features[i+1]**2
    count += (3*3)*3*resnet_features[-1]+3
    print("VARIABLE COUNT" , count)
    return


def periodic_phiflow_domain(resolution, batch_size, set_viscosity=False, Re=1000):
    domain = Domain(resolution, box=box[0:2 * np.pi, 0:2 * np.pi], boundaries=((PERIODIC, PERIODIC), (PERIODIC, PERIODIC)))
    staggered_shape = calculate_staggered_shape(1, domain.resolution)
    centered_shape = calculate_centered_shape(1, domain.resolution)
    bcx = []
    bcy = []
    boundary_array = ((bcy, bcy), (bcx, bcx))
    boundary_bool = ((False, False), (False, False))
    dirichlet_mask, dirichlet_values, boundary_bool_mask, active_mask, accessible_mask = compute_preiodic_masks(staggered_shape, boundary_bool, boundary_array)
    no_slip = math.zeros(staggered_shape + np.array([0, 1, 1, -1])).astype(np.bool)
    grid_spacing = [np.ones(domain.resolution[0] + 2) * domain.dx[0], np.ones(domain.resolution[1] + 2) * domain.dx[1]]
    viscosity_placeholder = tf.placeholder(tf.float32, shape=(batch_size, np.sum([np.prod(d.data.shape) for d in lr_grids[0].data]), ))
    sim_physics = SimulationParameters(dirichlet_mask=dirichlet_mask.astype(bool), dirichlet_values=dirichlet_values, active_mask=active_mask,
                                        accessible_mask=accessible_mask, bool_periodic=(True, True), no_slip_mask=no_slip, 
                                        viscosity=1/Re if set_viscosity else None,
                                        linear_solve_class=LinearSolverCudaMultiBicgstabILU,
                                        linear_solve_kwargs={'cast_to_double': True, 'max_iterations': 1000, 'accuracy': 1e-8},
                                        pressure_solve_class=PisoPressureSolverCudaCsrOldCg,
                                        pressure_solve_kwargs={'cast_to_double': True, 'max_iterations': 2000, 'accuracy': 1e-8,
                                                                'laplace_rank_deficient': True, 'residual_reset_steps':200},
                                        grid_spacing=grid_spacing)
    return domain, sim_physics, viscosity_placeholder, centered_shape, staggered_shape


def kolmogorov_forcing(domain, dataset_wavenumber):
    kolmogorov_force_sim = [np.zeros((1, domain.resolution[0] + 1, domain.resolution[1], 1)),
                        tf.reshape(tf.sin(tf.linspace(0., 2 * np.pi * dataset_wavenumber, domain.resolution[1])), (1, domain.resolution[0], 1, 1)) *
                        np.ones((1, domain.resolution[0], domain.resolution[1] + 1, 1))]
    kolmogorov_force_sim = stack_staggered_components(kolmogorov_force_sim)
    return kolmogorov_force_sim



def run_testsim(model_path, model_initialiser, domain, dataset_wavenumber, stepcount, initial_condition, sim_physics, dt, storing_frames, momForce=False):
    tf.reset_default_graph()
    sim_physics.initialize_linear_solver(linear_solve_class = LinearSolverCudaMultiBicgstabILU,
                                         linear_solve_kwargs = {'cast_to_double': True, 'max_iterations': 1000, 'accuracy': 1e-6})
    sim_physics.initialize_pressure_solver(pressure_solve_class = PisoPressureSolverCudaCsrOldCg,
                                           pressure_solve_kwargs = {'cast_to_double': True, 'max_iterations': 1000, 'accuracy': 1e-6, 'residual_reset_steps': 1000})

    net, layers = model_initialiser()
    sample_preiodic = tf.placeholder(shape=initial_condition.shape, dtype=tf.float32, name='supervised_sequence')
    velocity = StaggeredGrid.sample(math.pad(sample_preiodic[..., :2], ((0, 0), (0, 1), (0, 1), (0, 0)), mode='circular'), domain)
    pressure = CenteredGrid(sample_preiodic[..., 2:], box=domain.box, extrapolation=pressure_extrapolation(domain.boundaries))

    kolmogorov_force_sim = kolmogorov_forcing(domain, dataset_wavenumber)
    viscosity_placeholder = tf.placeholder(tf.float32, shape=(1, np.sum([np.prod(d.data.shape) for d in velocity.data]),))
    viscosity_sliced = stagger_flattened_data(math.flatten(viscosity_placeholder), velocity.staggered_tensor().shape.as_list(), True, True)[:, :-1, :-1, :]

    velocity, pressure = piso_step(velocity, pressure, dt, sim_physics, forcing_term=kolmogorov_force_sim, viscosity_field=viscosity_placeholder)
    network_input = math.concat([velocity.staggered_tensor()[:, :-1, :-1, :], pressure.data, 0.001 / viscosity_sliced], axis=-1)
    network_output = net(network_input)
    corrected_state = network_input[..., :3] + math.concat([network_output, tf.zeros_like(pressure.data)], axis=-1)

    frame_init = initial_condition
    init_op = tf.global_variables_initializer()
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    trajectory = np.zeros([len(storing_frames),] + sample_preiodic.shape.as_list(), dtype=np.float64)
    with tf.Session(config=session_config) as sess:
        sess.run(init_op)
        weights = sum([l.weights for l in layers], [])
        saver = tf.train.Saver(weights)
        saver.restore(sess, model_path)

        k = 0
        frame = frame_init
        for j in range(stepcount + 1):
            if j in storing_frames:
                trajectory[k] = frame
                k += 1
            feed_dict = {sample_preiodic: frame, viscosity_placeholder: np.ones(viscosity_placeholder.shape.as_list())*sim_physics.viscosity}
            frame = sess.run(corrected_state, feed_dict)
            if np.isnan(frame).any():
                print('NaN encountered in frame ' + str(j))
                trajectory[k:] = np.ones(trajectory[k:].shape)*np.inf
                break
    return trajectory


def calculate_mse_stats(mean_squared_error, model_groups):
    grouped_mse = [mean_squared_error[model_groups[i]:model_groups[i + 1]] for i in range(len(model_groups) - 1)]
    nan_idx = []
    # Filter extreme diverged values
    trshld_nan = 1e2
    for idx , g in enumerate(grouped_mse):
        if np.any(np.isnan(g)) or np.any(np.array(g) > trshld_nan):
            nan_idx.append(idx)
            keep_idx = np.logical_not(np.logical_or(np.isnan(g), np.array(g) > trshld_nan))
            g = np.array(g)[keep_idx].tolist()
            grouped_mse[idx] = g
    average_mse = [np.average(m) for m in grouped_mse]
    std_mse = [np.std(m) for m in grouped_mse]
    return average_mse, std_mse, nan_idx, grouped_mse


def get_legend_selector(models, model_groups):
    legend_selector = []
    for mg in model_groups[:-1]:
        splitstring = "_corr_"
        
        print(models[mg])
        step_count = int(models[mg].split('step')[0].split('_')[-1])
        if step_count ==1:
            legend_selector.append(0)
        elif models[mg].split(splitstring)[1].split('_')[0] == 'sup':
            legend_selector.append(1)
        else:
            legend_selector.append(2)
    legends = ['ONE', 'NOG', 'WGR']
    return legends, legend_selector


def get_color_selector(models, model_groups):
    color_selector = []
    for mg in model_groups[:-1]:
        splitstring = "_corr_"
        
        step_count = int(models[mg].split('step')[0].split('_')[-1])
        if step_count ==1:
            color_selector.append("rosybrown")
        elif models[mg].split(splitstring)[1].split('_')[0] == 'sup':
            color_selector.append("tab:blue")
        else:
            color_selector.append("tab:orange")
    return color_selector


def load_trajectories(models, frame_count, evaluation_interval, Res):
    trajectories = []
    reference_frames = []
    for Re in Res:
        one_re_traj=[]
        for model in models:
            loaded = np.load(model.split('.ckpt')[0] + "_inference_traj_Re" + str(Re)+'.npz')['arr_0']
            assert np.all([l.shape[0]>= frame_count for l in loaded])
            one_re_traj.append(np.stack(loaded,1)[:frame_count:evaluation_interval].astype(np.float32))
        trajectories.append(np.stack(one_re_traj,0))
        
    reference_frames = [np.load('reference_data_1000.npy'), np.load('reference_data_600.npy')]
    reference_frames = np.stack(reference_frames,0)
    trajectories = np.stack(trajectories,1)
    return trajectories, reference_frames


def get_features(models):
    resnet_features = [[8, 20, 30, 20, 8],
                    [8, 16, 32, 64, 32, 16, 8],
                    [16, 32, 64, 128, 64, 32, 16],
                    [16, 32, 64, 128, 128, 128, 64, 32, 16]]

    feature_selector = []
    for m in models:
        feature_count = int(m.split('nFeat')[1].split('_')[0])
        feature_selector.append({31529: 0, 114963: 1, 458403: 2, 1081763: 3, 
                                32369: 0, 115803: 1, 461235: 2, 1084595: 3}[feature_count])
    return resnet_features, feature_selector