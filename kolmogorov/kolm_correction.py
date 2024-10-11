import sys
import socket

base_path = '../kolm_tests'

from diffpiso import *
from kolmogorov_flow.networks import *
import socket
from diffpiso.datamanagement import write_config_file
from helpers import *

def correct(params):
    gpu = params['gpu']
    batch_size = params['batch_size']
    feat_selector = params['feat_selector']
    gradient_stop = params['nogradient']
    random_seeds = params['seeds']
    step_counts = params['step_counts']
    starting_learning_rates = params['start_learning_rate']
    epochs = params['epochs']
    directory_identifier = params['directory_name']
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        
    Re = [300, 400, 500, 700, 800, 900]

    hr_resolution = [128,128]
    resolution_factor = 4
    resolution = [r//resolution_factor for r in hr_resolution]
    frame_count = 1500*4
    relative_data_size = frame_count/1500
    dataset_wavenumber = 6
    dataset_timestep = 0.005
    frame_increment = 4

    resnet_features = [[8, 20, 30, 20, 8], 
                    [8, 16, 32, 64, 32, 16, 8], 
                    [16, 32, 64, 128, 64, 32, 16], 
                    [16, 32, 64, 128, 128, 128, 64, 32, 16]][feat_selector]

    dataset_periodic = np.load(base_path+'/DS6000.npz')['arr_0']
    
    count = resnet_variable_count(resnet_features)
    identifier_path = create_base_dir(base_path,'/'+directory_identifier)
    step_paths = []

    for sei, seed in enumerate(random_seeds):
        for si, step_count in enumerate(step_counts):
            trainable_dataset_indices = [np.arange(frame_count - step_count - 1) + frame_count * d for d in range(len(Re))]
            trainable_dataset_indices = np.concatenate(trainable_dataset_indices, 0)
            if sei ==0:
                step_path = create_base_dir(identifier_path, ('/supervised' if gradient_stop else '/differentiable')
                                            +'_nFeat'+str(count)+'_'+str(step_count)+'step_DS'+str(frame_count)+'_')
                step_paths.append(step_path)
            else:
                step_path = step_paths[si]
                
            learning_rate = starting_learning_rates[si]
            learning_rate_decay = 0.9**relative_data_size
            tf.reset_default_graph()
            tf.random.set_random_seed(seed)
            net, layers = convResNet_2D_centered(resnet_features, 2, conv_skips=True)
            domain, sim_physics, viscosity_placeholder, centered_shape, staggered_shape = periodic_phiflow_domain(resolution, batch_size)
            kolmogorov_force_sim = kolmogorov_forcing(domain, dataset_wavenumber)

            dt = tf.placeholder(shape=(), dtype=tf.float32)
            sequence_periodic = tf.placeholder(shape=(batch_size, step_count+1, resolution[0], resolution[1],3), dtype=tf.float32, name='supervised_sequence')

            velocity = StaggeredGrid.sample(math.pad(sequence_periodic[:,0,...,:2], ((0,0), (0,1), (0,1), (0,0)), mode='circular'), domain)
            pressure = CenteredGrid(sequence_periodic[:,0,...,2:], box=domain.box, extrapolation=pressure_extrapolation(domain.boundaries))

            viscosity_sliced = stagger_flattened_data(math.flatten(viscosity_placeholder), velocity.staggered_tensor().shape.as_list(), True, True)[:,:-1,:-1,:]
            network_outputs = []
            corrected_tensors = []
            velocity_tensors = [velocity.staggered_tensor()]
            warning_pl = [tf.placeholder(shape=(batch_size,), dtype=tf.int32) for s in range(step_count)]
            solve_warning = []
            for i in range(step_count):

                velocity, pressure, p1, p2, mv, ci, rp, v1, v2, A, Af, irhs, isol, v3, v1div, H, Hdiv, L1, L2, warn, beta, stagVol, p1g, p2g, vdiff, irhscond =\
                    piso_step(velocity, pressure, dt, sim_physics, forcing_term=kolmogorov_force_sim, full_output=True, viscosity_field=viscosity_placeholder,
                                warning=warning_pl[i], unrolling_step=i)
                    
                network_input = math.concat([circular_slicing_differentiable(velocity.staggered_tensor()), pressure.data, 0.001/viscosity_sliced], axis=-1)
                network_output = net(network_input)
                network_outputs.append(network_output)
                corrected_state = network_input[...,:3]+math.concat([network_outputs[-1], tf.zeros_like(pressure.data)], axis=-1)
                corrected_tensors.append(corrected_state)

                if gradient_stop:
                    solve_warning.append(tf.stop_gradient(warn))
                    velocity = StaggeredGrid(tf.stop_gradient(circular_padding_differentiable(corrected_state[...,:2])),velocity.box, extrapolation=velocity.extrapolation)
                    pressure = CenteredGrid(tf.stop_gradient(corrected_state[...,2:]), box=pressure.box, extrapolation=pressure.extrapolation)
                else:
                    solve_warning.append(warn)
                    velocity = StaggeredGrid(circular_padding_differentiable(corrected_state[..., :2]), velocity.box, extrapolation=velocity.extrapolation)
                    pressure = CenteredGrid(corrected_state[..., 2:], box=pressure.box, extrapolation=pressure.extrapolation)

                velocity_tensors.append(velocity.staggered_tensor())

            loss = supervised_loss_centered(tf.stack(corrected_tensors, axis=1)[...,:2], sequence_periodic[:,1:,...,:2])

            learning_rate_placeholder = tf.placeholder(shape=(), dtype=tf.float32)
            trainable_vars = tf.trainable_variables()
            all_gradients = tf.gradients([loss, ] + solve_warning[1:], trainable_vars+ warning_pl[1:],
                                            grad_ys=[None,]+[np.zeros((batch_size), dtype=np.int32) for s in range(step_count-1)])
            variable_grads, warn_grads = all_gradients[:len(trainable_vars)], all_gradients[len(trainable_vars):]
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_placeholder)
            grads_pl = [tf.placeholder(dtype=tf.float32, shape=v.shape) for v in tf.trainable_variables()]
            train_op = optimizer.apply_gradients(zip(grads_pl, tf.trainable_variables()))
            cutoff_norms = np.ones(len(grads_pl))*1e6

            init_op = tf.global_variables_initializer()
            session_config = tf.ConfigProto()
            session_config.gpu_options.allow_growth=True
            save_path = create_base_dir(step_path, '/kolm_corr_'+('sup_' if gradient_stop else 'diff_')
                                        + str(resolution[0]) + '-' + str(resolution[1]) + '_wave' + str(dataset_wavenumber) + '_step'+str(step_count)+'_inc'+str(frame_increment)+'_rand'
                                        + str(seed) + '_lr'+str(learning_rate)+'_b'+str(batch_size)+'_' +socket.gethostname()+'_')
            save_source(__file__, save_path, '/src.py')
            config_dict = params
            config_dict.update({'current_step_count': step_count, 'current_step_index': si, 'previous_runs': step_paths})
            write_config_file(config_dict, save_path)

            loss_progression = []
            consecutive_skip_count = 0

            with tf.Session(config=session_config) as sess:
                sess.run(init_op)
                weights = sum([l.weights for l in layers], [])
                saver = tf.train.Saver(weights, max_to_keep=epochs+1)
                if si>0:
                    saver.restore(sess,  step_paths[si-1]+'/starting_model_'+str(seed)+"_step"+str(step_counts[si-1])+'.ckpt'.replace('//','/'))

                for epoch in range(epochs):
                    grad_skip_count = 0
                    running_loss = 0.0
                    perm = np.random.permutation(trainable_dataset_indices.shape[0])
                    for i in range(trainable_dataset_indices.shape[0] // batch_size):
                        batch_perm = perm[batch_size * i:batch_size * (i + 1)]
                        indices = trainable_dataset_indices[batch_perm]
                        batch = [dataset_periodic[idx:idx + step_count + 1] for idx in indices]
                        batch = np.stack(batch, axis=0)
                        visc = np.ones(viscosity_placeholder.shape.as_list()).reshape((batch_size,-1))
                        visc /= np.array(Re)[indices//frame_count, np.newaxis]
                        feed_dict = {sequence_periodic: batch[...,:3], viscosity_placeholder: visc, dt: dataset_timestep * frame_increment,}
                        for s in range(step_count):
                            feed_dict[warning_pl[s]] = np.zeros((batch_size))

                        if gradient_stop: 
                            var_grads_out, loss_out, warn_out = sess.run([variable_grads, loss, solve_warning], feed_dict)
                            warn_back = [np.zeros((batch_size,), np.int32),]
                        else:
                            var_grads_out, loss_out, warn_out, warn_back = sess.run([variable_grads, loss, solve_warning, warn_grads], feed_dict)

                        all_warnings = np.array(warn_out+warn_back)
                        grad_norms = np.array([np.linalg.norm(v) for v in var_grads_out])
                        if (not np.any([np.any(np.isnan(v)) for v in var_grads_out])) and np.all(grad_norms<cutoff_norms) and np.sum(all_warnings)==0:
                            grad_dict = {learning_rate_placeholder: learning_rate * (min(i / (step_count * 64), 1) if (epoch < 1 and step_count > 1) else 1)}
                            grad_dict.update(dict(zip(grads_pl, var_grads_out)))
                            _=sess.run(train_op, feed_dict=grad_dict)
                            cutoff_norms = (9*cutoff_norms+(100*grad_norms))/10
                            loss_progression.append(loss_out)
                            running_loss += loss_out
                            consecutive_skip_count = 0
                        else:
                            print('gradient skip',not np.any([np.any(np.isnan(v)) for v in var_grads_out]),np.all(grad_norms<cutoff_norms), np.sum(all_warnings)==0)
                            print(warn_out, consecutive_skip_count, batch_perm, grad_norms)
                            consecutive_skip_count += 1
                            grad_skip_count += 1

                    print(f'[{epoch + 1}] loss: {running_loss / frame_count :.3f}  grad skips: {grad_skip_count}')

                    learning_rate *= learning_rate_decay
                    saver.save(sess, save_path + '/model_epoch_' + str(epoch).zfill(3)+ '.ckpt')
                    np.save(save_path+'/loss_progression.npy', loss_progression)
                    if (epoch == epochs-1):
                        saver.save(sess, step_path+'/starting_model_'+str(seed)+"_step"+str(step_count)+'.ckpt')
    return