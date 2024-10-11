# -*- coding: utf-8 -*-
"""

KS solver and unsupervised losses

"""

import scipy as scp
from phi.torch.flow import * 





# KS solver class
# note: domain sizes are pre-multiplied by 1/(2 PI) , i.e. a good range for chaotic behavior is around 10
class DifferentiableKS():
    def __init__(self, resolution, dt, doublestep=1):
        self.resolution = resolution
        self.dt = dt
        # enabled by default, increased difficulty of correction tasks , note - needs to be the same / enabled for data generation
        self.doublestep = doublestep 
        
    def calc_nonlinear(self,u):
        return -0.5*self.wavenumbers*math.fft(u**2)
    
    def etd1_step(self, u, domain_size):
        wavenumbers = math.fftfreq(math.spatial(x=self.resolution), domain_size/self.resolution).vector[0] * 1j
        L_mat = -wavenumbers**2-wavenumbers**4
        exp_lin = math.exp(L_mat * self.dt)

        nonlinear_coef_1 = math.divide_no_nan((exp_lin - 1) , L_mat)
        nonlinear_coef_1 = math.where(nonlinear_coef_1==0, self.dt, nonlinear_coef_1)
        nonlin_current = -0.5*wavenumbers*math.fft(u**2)

        u_new = exp_lin * math.fft(u) + nonlin_current*nonlinear_coef_1
        return math.real(math.ifft(tensor(u_new, u.shape)))
    
    def etrk2_step(self, u, domain_size):
        wavenumbers = math.fftfreq(math.spatial(x=self.resolution), domain_size/self.resolution).vector[0] * 1j
        L_mat = -wavenumbers**2-wavenumbers**4
        exp_lin = math.exp(L_mat * self.dt)

        nonlinear_coef_1 = math.divide_no_nan((exp_lin - 1) , L_mat)
        nonlinear_coef_1 = math.where(nonlinear_coef_1==0, self.dt, nonlinear_coef_1)
        nonlinear_coef_2 = math.divide_no_nan((exp_lin - 1 - L_mat*self.dt), (self.dt * L_mat**2))
        nonlinear_coef_2 = math.where(nonlinear_coef_2==0, self.dt/2, nonlinear_coef_2)
        nonlin_current = -0.5*wavenumbers*math.fft(u**2)
                
        u_interm = exp_lin * math.fft(u) + nonlin_current*nonlinear_coef_1
        u_new  = u_interm + ( -0.5*wavenumbers*math.fft(math.real(math.ifft(tensor(u_interm,u.shape)))**2) -
                 nonlin_current) * nonlinear_coef_2
        return math.real(math.ifft(tensor(u_new,u.shape)))

    def etd1(self, u, domain_size):
        if not self.doublestep:
            return self.etd1_step(u, domain_size)
        return self.etd1_step(self.etd1_step(u, domain_size), domain_size)

    def etrk2(self, u, domain_size):
        if not self.doublestep:
            return self.etrk2_step(u, domain_size)
        return self.etrk2_step(self.etrk2_step(u, domain_size), domain_size)




loss = torch.nn.MSELoss() # == supervised_loss()

# multi step
def unsupervised_loss(prediction, diff_ks, domain_size, steps=2): 
    prediction_input = prediction[:,:-steps,:].reshape((-1,prediction.shape[-1])) # merge time steps and batch size into one dimension, remove #steps entries
    prediction_input = tensor(prediction_input, instance('i'), spatial(x=prediction_input.shape[-1])) #  time*BS are instances, keep spatial x
    stepped_prediction = prediction_input
    for j in range(steps):
        stepped_prediction = diff_ks.etrk2( stepped_prediction, domain_size=domain_size ) # integrate all time*BS states forward in time 
    L = loss(prediction[:,steps:,:].reshape((-1,prediction.shape[-1])) , stepped_prediction.native(['i','x']) ) # compute loss towards #steps entries that were removed above
    return L


# run reference predictions for test cases, no network involved
def test_predict(diff_ks, pred_in, domain_size):
    pred_in = pred_in[:-1,:]
    pred_in = tensor(pred_in[:,0,:], instance('i'), spatial(x=pred_in.shape[-1]))
    stepped_prediction = diff_ks.etrk2(pred_in, domain_size).native(['i','x']).detach().cpu().numpy()

    # fix up content for eval to prevent overflows
    stepped_prediction2 = np.nan_to_num( np.clip( stepped_prediction , -1e20,1e20) , nan=1e21)
    return stepped_prediction2


# unroll simulation at training time, keep in synw w run_sim; runs a full batch of inputs
# do_detach > 0 indicates number of initial steps for which to cut the gradient flow (for values >= prediction_horizon all steps are cut)
# detach_interval: special mode for gradient cutting in intervals (a la Bjoern) , combine with do_detach=0
def train_sim(diff_ks, network_type, learning_task, input, net, prediction_horizon, edges_torch, domain_size, do_detach=0, detach_interval=0):
    inputs = [input[:,0:1,:]] # strip DS
    ds = input[:,1:2,:] * (1/8) # concat for net , normalize
    outputs = []

    if detach_interval>0 and (learning_task != 'correct' or network_type != 'CNN'): # sanity check
        print("error: detach_interval currently only partially supported!"); exit(1)

    if learning_task == 'predict':
        # prediction task
        if network_type == 'GCN':
            for m in range(prediction_horizon):
                net_in = torch.concat( [inputs[-1],ds], axis=1 )
                outputs.append( net(net_in.transpose(1,2), edges_torch).transpose(1,2) )
                if m >= do_detach: # regular, differentiable recurrent backprop training (no solver in the loop for predictions)
                    inputs.append(outputs[-1])  
                else: # detach for supervised data-matching 
                    inputs.append( outputs[-1].detach() )
        elif network_type == 'CNN':
            for m in range(prediction_horizon):
                outputs.append( net( torch.concat( [inputs[-1],ds],axis=1) ))
                if m >= do_detach: # regular
                    inputs.append(outputs[-1])  
                else: # detach
                    inputs.append( outputs[-1].detach() )

    elif learning_task == 'correct':
        # correction, parallel evaluation of NN and P solver (sequential version below)
        if network_type == 'GCN':
            for m in range(prediction_horizon):
                net_in = torch.concat( [inputs[-1],ds], axis=1 )
                net_out = net(net_in.transpose(1,2), edges_torch).transpose(1,2)
                outputs.append( diff_ks.etd1(tensor(inputs[-1], batch('b'), instance('time'), spatial('x')),domain_size).native(['b','time','x'])
                                + diff_ks.dt * net_out )
                if m >= do_detach: # regular, differentiable physics training
                    inputs.append( outputs[-1] )  
                else: # detach for supervised data-matching -> no differentiation through time
                    inputs.append( outputs[-1].detach() )
        elif network_type == 'CNN':
            for m in range(prediction_horizon):   
                outputs.append( diff_ks.etd1(tensor(inputs[-1][:,0:1,:], batch('b'), instance('time'), spatial('x')),domain_size).native(['b','time','x'])
                                + diff_ks.dt * net( torch.concat( [inputs[-1],ds],axis=1)) )
                if m >= do_detach: # regular, differentiable physics training
                    inputs.append( outputs[-1] ) 
                else: # detach for supervised data-matching -> no differentiation through time
                    inputs.append( outputs[-1].detach() )

                # interval cutting, assumes do_detach=0
                if detach_interval>0 and (m % detach_interval)==(detach_interval-1):
                    inputs[-1] = outputs[-1].detach() # overwrite last entry

    elif learning_task == 'corr-pred-transition':
        # sequential version with 2nd order , for varying corr/pred task
        if network_type == 'GCN':
            for m in range(prediction_horizon):
                st1 = diff_ks.etrk2(tensor(inputs[-1], batch('b'), instance('time'), spatial('x')),domain_size).native(['b','time','x'])
                net_in = torch.concat( [st1,ds],axis=1 )
                st2 = (1-2*diff_ks.dt) * net(net_in.transpose(1,2), edges_torch).transpose(1,2)
                outputs.append( st1 + st2 )
                if m >= do_detach: # regular differentiable physics
                    inputs.append(outputs[-1])  
                else: # detach
                    inputs.append( outputs[-1].detach() )
        elif network_type == 'CNN':
            for m in range(prediction_horizon):   
                st1 = diff_ks.etrk2(tensor( inputs[-1], batch('b'), instance('time'), spatial('x')),domain_size).native(['b','time','x'])
                st2 = (1-2*diff_ks.dt) * net( torch.concat( [st1,ds],axis=1) )
                outputs.append( st1 + st2 )
                if m >= do_detach: # regular
                    inputs.append(outputs[-1]) 
                else: # detach
                    inputs.append( outputs[-1].detach() )

    # alternative versions of prediction / correction

    elif learning_task == 'correct-alt': 
        # correction sequential P(NN()) alternative, regular "correct" has parallel NN() & P()
        if network_type == 'GCN':
            print("error NYI"); exit(1)
        elif network_type == 'CNN':
            for m in range(prediction_horizon):   
                cx1 = diff_ks.etd1(tensor(inputs[-1][:,0:1,:], batch('b'), instance('time'), spatial('x')),domain_size).native(['b','time','x'])  
                cx2 = diff_ks.dt * net( torch.concat( [cx1,ds],axis=1))
                outputs.append(cx1 + cx2)
                if m >= do_detach: # regular, differentiable physics training
                    inputs.append( outputs[-1] ) 
                else: # detach for supervised data-matching -> no differentiation through time
                    inputs.append( outputs[-1].detach() )

                # interval cutting, assumes do_detach=0
                if detach_interval>0 and (m % detach_interval)==(detach_interval-1):
                    inputs[-1] = outputs[-1].detach() # overwrite last entry

    elif learning_task == 'predict-alt':
        # prediction task
        if network_type == 'GCN':
            print("error NYI"); exit(1)
        elif network_type == 'CNN':
            for m in range(prediction_horizon):
                outputs.append( inputs[-1] + net( torch.concat( [inputs[-1],ds],axis=1) ))
                if m >= do_detach: # regular
                    inputs.append(outputs[-1])  
                else: # detach
                    inputs.append( outputs[-1].detach() )

    elif learning_task == 'correct-alt2': 
        # correction sequential P(NN()) alternative without residual addition
        if network_type == 'GCN':
            print("error NYI"); exit(1)
        elif network_type == 'CNN':
            for m in range(prediction_horizon):   
                cx1 = diff_ks.etd1(tensor(inputs[-1][:,0:1,:], batch('b'), instance('time'), spatial('x')),domain_size).native(['b','time','x'])  
                cx2 = net( torch.concat( [cx1,ds],axis=1))
                outputs.append(cx2)
                if m >= do_detach: # regular, differentiable physics training
                    inputs.append( outputs[-1] ) 
                else: # detach for supervised data-matching -> no differentiation through time
                    inputs.append( outputs[-1].detach() )

                # interval cutting, assumes do_detach=0
                if detach_interval>0 and (m % detach_interval)==(detach_interval-1):
                    inputs[-1] = outputs[-1].detach() # overwrite last entry

    else:
        print("Error, unknown learning task "+learning_task); exit(1)

    outputs = torch.concat(outputs ,axis=1) # concat along time steps
    return inputs, outputs



# append to output list, small helper to reduce copy paste code below
def app_out(out,device_iter,i,horizon_append_start):
    if i>=(horizon_append_start-1):
        out.append( device_iter.detach().cpu() )
    return

# run simulation w trained network in batches, similar to train_sim, but a bit more efficient (primarily for ks_eval script)
# runs a full batch, but retains old ("wrong") return type that accumulates time steps in batch dimesions, and stores versions in a list (for compatiblity with legacy code in ks-evalgrid1.py)
def run_sim_eval(device, diff_ks, network_type, learning_task, initial_full, net, horizon, edges_torch, domain_size, out_ret = [], split = 1, horizon_append_start=-1):
    if not net:
        return []
    # out_ret: turn into list again at the end, for legacy code in evalgrid , is an input param now

    # batches can be too large to fit into memory (esp for large models) -> split
    # needs to be computed outside for split...
    for s in range(split):
        irange = int( initial_full.shape[0] / split )
        irs = irange*s 
        ire = min( initial_full.shape[0], irange*(s+1) )
        if s==split-1:
            ire = initial_full.shape[0] # extend last batch to end
        initial = initial_full[irs:ire,...]
        if horizon_append_start<=0:
            out = [ initial.detach().clone().cpu() ]
        else:
            out = [] 
        device_iter = initial.clone().to(device)
        ds = torch.ones(device_iter.shape) * domain_size * (1/8) # concat for net , normalize
        ds = ds.to(device)

        if learning_task == 'predict':
            if network_type == 'GCN':
                for j in range(horizon):
                    net_in = torch.concat( [device_iter,ds],axis=1 )
                    device_iter = net(net_in.transpose(1,2), edges_torch).transpose(1,2)
                    app_out(out, device_iter, j, horizon_append_start)
            elif network_type == 'CNN':
                for j in range(horizon):
                    device_iter = net(torch.concat( [device_iter,ds],axis=1))
                    app_out(out, device_iter, j, horizon_append_start)

        elif learning_task == 'correct':
            if network_type == 'GCN':
                for j in range(horizon):
                    net_in = torch.concat( [device_iter,ds],axis=1 )
                    net_out = net(net_in.transpose(1,2), edges_torch).transpose(1,2)
                    device_iter = ( diff_ks.etd1(tensor(device_iter, batch('b'), instance('time'), spatial('x')),domain_size).native(['b','time','x'])
                                    + diff_ks.dt * net_out )
                    app_out(out, device_iter, j, horizon_append_start)

            elif network_type == 'CNN':
                for j in range(horizon):
                    device_iter = ( diff_ks.etd1(tensor(device_iter, batch('b'), instance('time'), spatial('x')),domain_size).native(['b','time','x'])
                                + diff_ks.dt * net(torch.concat( [device_iter,ds],axis=1) ) )
                    app_out(out, device_iter, j, horizon_append_start)

        elif learning_task == 'corr-pred-transition':
            if network_type == 'GCN':
                for j in range(horizon):
                    sr1 = diff_ks.etrk2(tensor(device_iter, batch('b'), instance('time'), spatial('x')),domain_size).native(['b','time','x'])
                    net_in = torch.concat( [sr1,ds],axis=1 )
                    sr2 = (1-2*diff_ks.dt) * net(net_in.transpose(1,2), edges_torch).transpose(1,2)
                    device_iter = sr1 + sr2
                    app_out(out, device_iter, j, horizon_append_start)
            elif network_type == 'CNN':
                for j in range(horizon):
                    sr1 = diff_ks.etrk2(tensor(device_iter, batch('b'), instance('time'), spatial('x')),domain_size).native(['b','time','x'])
                    sr2 = (1-2*diff_ks.dt) * net( torch.concat( [sr1,ds],axis=1) ) 
                    device_iter = sr1 + sr2
                    app_out(out, device_iter, j, horizon_append_start)

        elif learning_task == 'correct-alt':
            if network_type == 'GCN':
                print("error NYI"); exit(1)
            elif network_type == 'CNN':
                for j in range(horizon):
                    cr1 = diff_ks.etd1(tensor(device_iter, batch('b'), instance('time'), spatial('x')),domain_size).native(['b','time','x'])
                    cr2 = diff_ks.dt * net( torch.concat( [cr1,ds],axis=1) ) 
                    device_iter = cr1 + cr2
                    app_out(out, device_iter, j, horizon_append_start)

        elif learning_task == 'predict-alt':
            if network_type == 'GCN':
                print("error NYI"); exit(1)
            elif network_type == 'CNN':
                for j in range(horizon):
                    device_iter = device_iter + net(torch.concat( [device_iter,ds],axis=1))
                    app_out(out, device_iter, j, horizon_append_start)

        elif learning_task == 'correct-alt2':
            if network_type == 'GCN':
                print("error NYI"); exit(1)
            elif network_type == 'CNN':
                for j in range(horizon):
                    cr1 = diff_ks.etd1(tensor(device_iter, batch('b'), instance('time'), spatial('x')),domain_size).native(['b','time','x'])
                    cr2 = net( torch.concat( [cr1,ds],axis=1) ) 
                    device_iter = cr2 # more like original predict, no residual!
                    app_out(out, device_iter, j, horizon_append_start)

        else:
            print("Error, unknown learning task "+learning_task); exit(1)

        # return numpy version , directly fix up large values and NANs
        # warning! stacks up results along batch dim instead of time... (this is what the eval code uses)
        #out_ret_clip = np.nan_to_num( np.clip( torch.stack(out).numpy() , -1e12,1e12) , nan=1e13, posinf=1e14, neginf=-1e14)  
        out_ret_clip = np.clip( torch.stack(out).numpy() , -1e12,1e12) 
        np.nan_to_num( out_ret_clip , copy=False, nan=1e13, posinf=1e14, neginf=-1e14)  
        for i in range(out_ret_clip.shape[1]):
            out_ret.append( out_ret_clip[:,i,:,:])
    return out_ret






# run simulation w trained network , similar to train_sim, but a bit more efficient 
# (this function is not too important, and does not support all learning tasks)
# runs via "device_iter" , unlike previous version, doesnt use batches... (slower) still used in train script
def run_sim(device, diff_ks, network_type, learning_task, initial, net, horizon, edges_torch, domain_size):
    out = [ initial.detach().cpu() ]
    device_iter = initial.to(device)
    ds = torch.ones(device_iter.shape) * domain_size * (1/8) # concat for net , normalize
    ds = ds.to(device)

    if learning_task == 'predict':
        if network_type == 'GCN':
            for j in range(horizon):
                net_in = torch.concat( [device_iter,ds],axis=1 )
                device_iter = net(net_in.transpose(1,2), edges_torch).transpose(1,2)
                out.append( device_iter.detach().cpu() )
        elif network_type == 'CNN':
            for j in range(horizon):
                device_iter = net( torch.concat( [device_iter,ds],axis=1) )
                out.append( device_iter.detach().cpu() )

    elif learning_task == 'correct':
        if network_type == 'GCN':
            for j in range(horizon):
                net_in = torch.concat( [device_iter,ds],axis=1 )
                netout = net(net_in.transpose(1,2), edges_torch).transpose(1,2)
                device_iter = ( diff_ks.etd1(tensor(device_iter, batch('b'), instance('time'), spatial('x')),domain_size).native(['b','time','x'])
                                + diff_ks.dt * netout )
                out.append( device_iter.detach().cpu() )
        elif network_type == 'CNN':
            for j in range(horizon):
                device_iter = ( diff_ks.etd1(tensor(device_iter, batch('b'), instance('time'), spatial('x')),domain_size).native(['b','time','x'])
                                + diff_ks.dt * net( torch.concat( [device_iter,ds],axis=1) )) 
                out.append( device_iter.detach().cpu() )
    elif learning_task == 'corr-pred-transition':
        if network_type == 'GCN':
            for j in range(horizon):
                sr1 = diff_ks.etrk2(tensor(device_iter, batch('b'), instance('time'), spatial('x')),domain_size).native(['b','time','x'])
                net_in = torch.concat( [sr1,ds],axis=1 )
                sr2 = (1-2*diff_ks.dt) * net(net_in.transpose(1,2), edges_torch).transpose(1,2)
                device_iter = sr1 + sr2
                out.append( device_iter.detach().cpu() )
        elif network_type == 'CNN':
            for j in range(horizon):
                sr1 = diff_ks.etrk2(tensor(device_iter, batch('b'), instance('time'), spatial('x')),domain_size).native(['b','time','x'])
                sr2 = (1-2*diff_ks.dt) * net( torch.concat( [sr1,ds],axis=1) ) 
                device_iter = sr1 + sr2
                out.append( device_iter.detach().cpu() )

    # return numpy version , directly fix up large values and NANs
    #out_ret_clip = np.nan_to_num( np.clip( torch.concat(out, axis=0).numpy() , -1e12,1e12) , nan=1e13, posinf=1e14, neginf=-1e14)
    out_ret_clip = np.clip( torch.concat(out, axis=0).numpy() , -1e12,1e12) 
    np.nan_to_num( out_ret_clip , copy=False, nan=1e13, posinf=1e14, neginf=-1e14)  
    return out_ret_clip

