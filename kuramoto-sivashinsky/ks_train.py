# -*- coding: utf-8 -*-
"""

KS test
warning: WIG version from paper is called WGR here

"""
import numpy as np
import os, argparse, subprocess, logging, inspect, shutil, pickle, time
import torch

DISABLE_GCN = int(os.environ.get("KS_DISABLE_GCN", "0"))>0
if not DISABLE_GCN:
    import torch_geometric

time_start = time.time()
parser = argparse.ArgumentParser(description='CmdLine Parser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# main training /simulation options
parser.add_argument(  '--learning_task', default='-', type=str, help='Inference task: predict|correct|corr-pred-transition , pure NN prediction or correction of low-order solver; for details see solver implementation')
parser.add_argument(  '--seed',       default=0,   type=int, help='Random seed, when 0 it is taken from output directory')
parser.add_argument(  '--epochs',  default=4,  type=int, help='training epochs, is increased when the dataset fraction is reduced (use --epochs_override to set number directly)')
parser.add_argument(  '--timestep', default=0.5,  type=float, help='KS solver time step, all old tests use 1/2')
parser.add_argument(  '--dataset_fraction', default=0.25,  type=float, help='reduce default size of 6*5k steps by this amount (automatically increases epochs!)')
parser.add_argument(  '--network_type',  default='-', type=str, help='Type of network architecture: GCN or CNN')
parser.add_argument(  '--channels', default=16, type=int, help='NN resnet conv channels')
parser.add_argument(  '--depth',    default=10, type=int, help='NN resnet depth (in blocks, ie 2 conv layers)')
parser.add_argument(  '--numds',    default=6,  type=int, help='number of domain sizes to train with, reduces to default size 1 for numds==1')
parser.add_argument(  '--dtpartial', default=-1.0,  type=float, help='Special mode for "corr-pred-transition" task, train with partial steps in correction, warning - relies on dt=0.5')
parser.add_argument(  '--lr_gamma',     default=0.9, type=float, help="Gamma factor for exponential learning rate decay scheduler (1 means off)")
parser.add_argument(  '--lr_factor',    default=1.0, type=float, help="learning rate factor")
parser.add_argument(  '--predhori',  default=3,  type=int, help='prediction horizon, how many steps to unroll (crucial parameter)')
parser.add_argument(  '--batch_size',  default=16, type=int, help='Batch size for training')
parser.add_argument(  '--save_inter',  default=-1,  type=int, help='Save interval for models, by default only save final one')
parser.add_argument(  '--tag',          default=0,  type=int, help='Integer tag to append to params pickel, to be used for eval and plotting (eg varying predhori)')
parser.add_argument(  '--tag_comment',  default='', type=str, help='Comment about tag, just for debugging for now')
parser.add_argument(  '--cpu',       default=0,  type=int, help='force CPU only')
parser.add_argument(  '--write_images',  default=1,  type=int, help='write out full simulation images, debugging')
# fine tuning options 
parser.add_argument(  '--outstart',   default=0,  type=int, help='Start nr of outNNN dirs, more for debugging...')
parser.add_argument(  '--do_train',   default=1, type=int, help='Run training?')
parser.add_argument(  '--do_eval',    default=1,  type=int, help='Run evaluations?')
parser.add_argument(  '--do_gendata', default=1,  type=int, help='Generate data?')
parser.add_argument(  '--train_nog',      default=1,  type=int, help='Train no grad version')
parser.add_argument(  '--train_wig',      default=1,  type=int, help='Train with grad version')
parser.add_argument(  '--train_physloss', default=0,  type=int, help='Train third version with full physics loss (previously called DP)')
parser.add_argument(  '--train_onestep',  default=1,  type=int, help='Train fourth version with naive one-step supervised loss')
parser.add_argument(  '--train_continue',  default=-1,  type=int, help='Continue training by loading a previously trained model? (on if >0) For now this int determines with outNNN directory to load the corresponding model from (admittedly a bit inflexible, outdirNum-params[outstart] is added)')
parser.add_argument(  '--train_continue_dir', default="",  type=str, help='For continue training options: load from (relative) directory path instead of current one, add "/" at end!')
parser.add_argument(  '--train_continue_loadone', default=1,  type=int, help='If 1, always load ONE model, not corresponding NOG/WGR model (default)')
parser.add_argument(  '--epochs_override', default=-1, type=int, help="Readjust epochs after dataset size adjustment, direct override for \"fine\" control")
parser.add_argument(  '--self_connect_nodes', default=1, type=int, help="Choose to have connecting edges that connect node to itself in the connectivity graph, default value is true(1)")
parser.add_argument(  '--compile',      default=1, type=int, help="Choose to have JIT compiled networks, slower start, but some performance improvements at runtime")
parser.add_argument(  '--warmup_steps', default=0, type=int, help="Reduce initial steps for which to compute loss")
parser.add_argument(  '--detach_interval', default=0, type=int, help="Interval based gradient cutting")
pargs = parser.parse_args()
params = {}
params.update(vars(pargs))

# ----
compile = params['compile'] > 0
DO_TRAIN = params['do_train'] > 0

WARMUP_STEPS = params['warmup_steps']

# write to outXXX
outdirNum = -1
#for i in range(1000):
for i in range(params['outstart'],1000):
    OUTDIR = "out{:03d}".format(i)
    if not os.path.exists(OUTDIR):
        if not DO_TRAIN:
            # continue last
            OUTDIR = "out{:03d}".format(i-1)
            outdirNum = i
            print("continuing in dir "+OUTDIR)
        else:
            # new run
            os.mkdir(OUTDIR)
            outdirNum = i
            print("new output dir "+OUTDIR)
        break

os.chdir(OUTDIR)

# check whether source dir to load models from exists, train_continue_dir is added as prefix relative to original CWD
LOADDIR = ""

if len(params['train_continue_dir'])>0 and params['train_continue_dir'][-1]!='/': # append / if necessary
    params['train_continue_dir'] = params['train_continue_dir'] + '/'

if params['train_continue']>=0:    
    print(os.path.abspath(os.getcwd()))
    if params['train_continue']==0:      # use same IDs
        LOADDIR = "../{}out{:03d}".format(params['train_continue_dir'], outdirNum )
    else:                                # otherwise add offset relative to current outstart 
        LOADDIR = "../{}out{:03d}".format(params['train_continue_dir'], params['train_continue'] + outdirNum-params['outstart'])
    if not os.path.exists(LOADDIR):
        print("Error, train_continue active but dir not found: "+LOADDIR); exit(1)


# start log
mylog = logging.getLogger()
mylog.addHandler(logging.StreamHandler())
mylog.setLevel(logging.INFO)
mylog.addHandler(logging.FileHandler('./run.log'))
mylog.info("Params "+str(params))

# print and log
def log(s):
    #print(s)
    mylog.info(s)

# dont use print() from now on! only log() ...


# small dataset class to mix multiple datasets
class DatasetKS():
    def __init__(self, inp,outp, batch_size):
        self.inp  = inp
        self.outp = outp
        self.batch_size = batch_size

    def reset(self): # init, always call before enumerate()
        self.dataset = []
        self.dataloader = []
        self.indices = []
        for i in range(len(self.outp)):
            self.dataset.append( torch.utils.data.TensorDataset(self.inp[i], self.outp[i]) )
            self.dataloader.append( torch.utils.data.DataLoader(self.dataset[i], shuffle=True, batch_size=self.batch_size) )

    def __iter__(self): # start enumeration
        self.indices = []
        self.dataenum = []
        for i in range(len(self.outp)):
            self.dataenum.append( enumerate(self.dataloader[i], 0) )
            self.indices.append( np.ones( len(self.outp[i]) , dtype=int) * i )

        self.indices = np.concatenate(self.indices, axis=0)
        t = 0 
        for i in range(len(self.outp)):
            t += len(self.dataloader[i])
        self.i = 0
        perm = np.random.permutation(len(self.indices))
        self.indices = self.indices[perm]
        return self

    def __len__(self): # precompute?
        l = 0
        for i in range(len(self.outp)):
            l += len(self.outp[i])
        return int(l/self.batch_size)

    def __next__(self): 
        if self.i<len(self.indices):
            idx = self.indices[self.i]
            self.i += 1
            dli, data = self.dataenum[idx].__next__()
            return data
        else:
            raise StopIteration



# ------------------------------------------------------------------------------------------------------------------------------

# PART 1 , generat data

import scipy as scp
from phi.torch.flow import * 
import matplotlib.pyplot as plt
TORCH.set_default_device("GPU") 
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if params['cpu'] > 0:
    device = 'cpu' ; TORCH.set_default_device("CPU") # DEBUG
device = torch.device(device)
#torch.set_float32_matmul_precision('high') # recommended

if params['dtpartial']!=-1. and params['learning_task']!='corr-pred-transition':
    print("Error, dtpartial settings require learning task 'corr-pred-transition' "); exit(1)

if params['seed']==0:
    print("Using seed from outdir! "+format(outdirNum+1000)) # for reproducible randomness
    params['seed'] = outdirNum+1000

torch.manual_seed(params['seed'])
np.random.seed(params['seed'])

log("pytorch device: "+format(device))

from ks_solver import DifferentiableKS, test_predict, train_sim, run_sim, unsupervised_loss




# solver instance
SN = 48 
domain_size = 8 # note, domain size is already pre-multiplied by 1/(2*PI)
diff_ks = DifferentiableKS(resolution=SN, dt=params['timestep'])


# generate data
NUM_DS  = params['numds']
dataset_dssizes = [0.9, 0.95, 1.0, 1.1, 1.2, 1.3]
NUM_DS_START = 0
if NUM_DS==1:
    NUM_DS_START = 3 # only use 1.0

if params['do_gendata'] > 0:
    WARMUP=200; LEN=5000 # warning , "huge" datasset to be on the safe side; could be reduced...
    domain_size_base = domain_size

    # generate all domain sizes
    for di in range(len(dataset_dssizes)):
        domain_size = domain_size_base * dataset_dssizes[di] 
        dsname = '../ks-dataset-'+str(di)  # a bit ugly, but we're already in directory outNNN/
        if os.path.exists(dsname+".npz"):
            continue
        log("Generating data "+str(di)+" for domain_size "+str(domain_size))
        with math.precision(64):
            u_dataset_init = [math.cos(2*math.PI*x) +0.1*math.cos(2*math.PI*x/domain_size)*(1+0.1*math.sin(2*math.PI*x/domain_size)),
                              math.cos(2*math.PI*x) +0.1*math.cos(2*math.PI*x/domain_size)*(1-0.1*math.sin(2*math.PI*x/domain_size)),
                              math.cos(2*math.PI*x) -0.2*math.cos(2*math.PI*x/domain_size)*(1+0.3*math.sin(2*math.PI*x/domain_size)),
                              math.cos(2*math.PI*x) -0.2*math.cos(2*math.PI*x/domain_size)*(1-0.3*math.sin(2*math.PI*x/domain_size)),
                              math.cos(2*math.PI*x) +0.3*math.cos(2*math.PI*x/domain_size)*(1+0.5*math.sin(2*math.PI*x/domain_size)),
                              math.cos(2*math.PI*x) +0.3*math.cos(2*math.PI*x/domain_size)*(1-0.5*math.sin(2*math.PI*x/domain_size)),]

            u_dataset_init = math.stack(u_dataset_init, batch('b'))
            u_traj = [u_dataset_init]
            u_iter = u_dataset_init

            for i in range(WARMUP+LEN): 
                u_iter = diff_ks.etrk2(u_iter,domain_size)
                u_traj.append(u_iter)

        u_traj = tensor(u_traj,instance('time') , u_iter.shape).numpy(['b','time', 'x']).astype(np.single)
        # remove warmup steps to make sure we have randomized conditions
        u_traj = u_traj[:,WARMUP+1:,:]
        np.savez(dsname,u_traj) 
        log("Data written to "+dsname+".npz")
    domain_size = domain_size_base

# if partial steps are active, reinit solver to only do a part
if params['dtpartial']!=-1. and params['dtpartial']<1.:
    # just include params['dtpartial'] factor in solver init:
    diff_ks = DifferentiableKS(resolution=SN, dt=params['dtpartial']*params['timestep'])
    log("Partial time step active, using dt="+str(params['dtpartial']*params['timestep']) )

# setup NNs
from ks_networks import ConvResNet1D, GCN, stack_prediction_inputs, stack_prediction_targets, direction_feature

# graph network init: setup GCN "graph" and globals
def I(i):
    return (i % SN)  # periodic

neighbour_range = 1
edge_index = []
for i in range(SN):
    if params['self_connect_nodes']:
        edge_index.append([I(i), I(i)])
    for j in range(1, neighbour_range + 1):
        edge_index.append([I(i), I(i - j)])
        edge_index.append([I(i), I(i + j)])

edge_index = np.array(edge_index)
edges_torch = torch.from_numpy(edge_index.transpose()).to(device)
edges_torch = edges_torch.long()



# load data

prediction_horizon = params['predhori']

# Computing it here so that networks can take it as an input arg and avoid graph breaks
if compile:
    edgeft_channels = 4 # only needed for pytorch JIT compilation bugs, remove at some point
else:
    edgeft_channels = 1
direction_ft = direction_feature(edges_torch, edgeft_channels, device=device)

torch_inputs = [] 
torch_outputs = []
dataset_org = []
#for di in range(2,5): # center 0.1

for di in range(NUM_DS_START,NUM_DS_START+NUM_DS):
    dsname = '../ks-dataset-'+str(di)+'.npz'
    dataset = np.load(dsname)['arr_0']
    log("Loaded file "+dsname+", shape "+str(dataset.shape) ) # -> (6, 8001, 48)

    if params['dataset_fraction']<1.0:
        dlen = int(dataset.shape[1]*params['dataset_fraction'])
        dataset = dataset[:,0:dlen,:]
    dataset_org.append(dataset) # keep around, eg for reinit of ONE models 

    torch_input = np.array([stack_prediction_inputs(d, prediction_horizon,1) for d in dataset])
    torch_input = torch.Tensor(torch_input.reshape(-1,1,torch_input.shape[-1]))

    # add/concat constant channel encoding domain size; warning only single size supported per batch at the moment
    torch_input_ds = torch.ones( torch_input.shape ) * domain_size * dataset_dssizes[di] 
    torch_input = torch.concat( [torch_input, torch_input_ds], axis=1)
    torch_inputs.append( torch_input )

    torch_output = np.array([stack_prediction_targets(d, prediction_horizon,1) for d in dataset])
    torch_outputs.append( torch.Tensor(torch_output.reshape(-1,prediction_horizon, torch_output.shape[-1])) )
    #log("single input output shapes:" + format([torch_inputs[-1].shape,torch_outputs[-1].shape])) 


datasetKS = DatasetKS(torch_inputs, torch_outputs, params['batch_size'])


if params['dataset_fraction']<1.0:
    params['epochs'] = int( params['epochs']/params['dataset_fraction'] + 0.5 )
    log("reduced dataset by {} to length {}, increased epochs to {}".format( params['dataset_fraction'], dlen ,params['epochs'] ) )

# readjust epochs after dataset size adjustment, direct override for "fine" control
if params['epochs_override']>0:
    params['epochs'] = params['epochs_override']
    log("Override epochs to {}".format( params['epochs'] ) )




"""---"""

# network settings
channels = params['channels']
depth    = params['depth']
network_type = params['network_type']
learning_task = params['learning_task']

# sanity checks
if network_type=='GCN' or network_type=='CNN':
	log("Network type "+network_type)
else:
	log("Unknown network type "+params['network_type']+"! Must be specificed via --network_type GCN|CCN ")
	exit(1)

log("Learning task "+learning_task)

# training settings
epochs = params['epochs']
lr_gamma = params['lr_gamma']



# write parameters
if params['train_onestep'] or params['train_nog'] or params['train_wig'] or params['train_physloss']:
    if DO_TRAIN:
        if os.path.exists(inspect.stack()[-1][1]):
            shutil.copy(inspect.stack()[-1][1], './')  # copy main script via absolute path, we're already in OUTDIR
        elif os.path.exists("../"+inspect.stack()[-1][1]):
            shutil.copy("../"+inspect.stack()[-1][1], './')  # macs dont have an absolute one somehow
        else:
            log("Warning, skipping script backup copy") # not found
        shutil.copy("../ks_networks.py", './') 
        shutil.copy("../ks_solver.py", './') 

        params['channels'] = channels
        params['depth'] = depth
        params['network_type'] = network_type
        params['SN'] = SN

        # a bit wasteful, allocate net just to count params
        if network_type=='GCN':
            netTEMP = GCN(hidden_channels=channels, depth=depth, device=device, edgeconv=1, direction_ft=direction_ft)
            if compile:
                netTEMP = torch_geometric.compile(netTEMP)
        elif network_type=='CNN':
            netTEMP = ConvResNet1D(channels, depth, device=device)
            if compile:
                netTEMP = torch.compile(netTEMP)
        num_params = sum(p.numel() for p in netTEMP.parameters())
        params['numparams'] = num_params; del netTEMP

        with open('params.pickle', 'wb') as f: pickle.dump(params, f)




loss = torch.nn.MSELoss() # == supervised_loss()

INTER_OUT = 200 
INTER_LRD = 2000 # LR decay

# ------------------------------------------------------------------------------------------------------------------------------
# PART 2: SUPERVISED TRAINING w/o gradients, always included


netNOG = None
if params['train_nog']:
    torch.manual_seed(params['seed']) # reset seed for network init & training
    np.random.seed(params['seed'])
    if network_type=='GCN':
        netNOG = GCN(hidden_channels=channels, depth=depth, device=device, edgeconv=1, direction_ft=direction_ft)
        if compile:
            netNOG = torch_geometric.compile(netNOG)
        optimizer = torch.optim.Adam(netNOG.parameters(), lr=1e-3*params['lr_factor'])
    elif network_type=='CNN':
        netNOG = ConvResNet1D(channels, depth, device=device)
        if compile:
            netNOG = torch.compile(netNOG)
        optimizer = torch.optim.Adam(netNOG.parameters(), lr=1e-4*params['lr_factor'])    
    else:
        log("Unknown network type "+params['network_type']+"! Must be specificed via --network_type GCN|CCN ")
        exit(1)
    num_params = sum(p.numel() for p in netNOG.parameters())
    log('Number of parameters for '+params['network_type']+' NOG: '+format( num_params ))

    if params["train_continue"]>=0: # load pre-trained model to continue training
        if params["train_continue_loadone"]==0: 
            modelname = "model-nog.pickle"
        else:
            modelname = "model-one.pickle" # forced use of ONE as pretraining version
        netNOG.load_state_dict(torch.load(LOADDIR+"/"+modelname, map_location=device ))
        log('Continuing training from '+LOADDIR+"/"+modelname)

    if DO_TRAIN:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = lr_gamma)
        datasetKS.reset()
        log("data set size "+str(len(datasetKS)) )
        
        for epoch in range(epochs):
            for i, data in enumerate(datasetKS, 0):
                inputs, labels = data
                ds_curr = inputs[0,1,0] # extract current domain_size from batch channel 1 , all constant take entry 0
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                # NOG: do_detach=prediction_horizon -> detach all
                inputs, outputs = train_sim(diff_ks, network_type, learning_task, inputs, netNOG, prediction_horizon, edges_torch, 
                                            domain_size=ds_curr, do_detach=prediction_horizon , detach_interval=params['detach_interval'] )

                if WARMUP_STEPS>0: # simply remove initial warmup steps for loss
                    outputs = outputs[:,WARMUP_STEPS:,:]
                    labels  = labels[ :,WARMUP_STEPS:,:]

                loss_value = loss(outputs, labels)
                loss_value.backward()
                optimizer.step()

                if ((epoch*len(datasetKS)+i)%INTER_OUT)==0: 
                    log(f's[{epoch},{i}] supervised no-grad NOG loss: {(loss_value.item()*10000.) :.3f}')
                if ((epoch*len(datasetKS)+i)%INTER_LRD)==0: 
                    scheduler.step()
            
            if params['save_inter']>0 and epoch%params['save_inter'] == (params['save_inter']-1):
                torch.save(netNOG.state_dict(), "model-nog-{:03d}.pickle".format( epoch ))

        # print final loss 
        log(f'e[{epoch + 1}] supervised no-grad NOG loss: {(loss_value.item()*10000.) :.3f}')
        torch.save(netNOG.state_dict(), "model-nog.pickle")
    else:
        netNOG.load_state_dict(torch.load("model-nog.pickle", map_location=device ))
        log("loaded model-nog.pickle")


# ------------------------------------------------------------------------------------------------------------------------------
# PART 3: differentiable physics training, restart all

netWGR = None
if params['train_wig']:
    torch.manual_seed(params['seed']) # reset seed for network init & training
    np.random.seed(params['seed'])
    if network_type == 'GCN':
        netWGR = GCN(hidden_channels=channels, depth=depth, device=device, edgeconv=1, direction_ft= direction_ft)
        if compile:
            netWGR = torch_geometric.compile(netWGR)
        optimizer = torch.optim.Adam(netWGR.parameters(), lr = 1e-3*params['lr_factor'])
    elif network_type == 'CNN':
        netWGR = ConvResNet1D(channels, depth, device=device)
        if compile:
            netWGR = torch.compile(netWGR)
        optimizer = torch.optim.Adam(netWGR.parameters(), lr = 1e-4*params['lr_factor'])

    if params["train_continue"]>=0: # load pre-trained model to continue training
        if params["train_continue_loadone"]==0: 
            modelname = "model-wgr.pickle"
        else:
            modelname = "model-one.pickle" # forced use of ONE as pretraining version
        netWGR.load_state_dict(torch.load(LOADDIR+"/"+modelname, map_location=device ))
        log('Continuing training from '+LOADDIR+"/"+modelname)

    if DO_TRAIN:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = lr_gamma)
        datasetKS.reset()
        for epoch in range(epochs):
            for i, data in enumerate(datasetKS, 0):
                inputs, labels = data
                ds_curr = inputs[0,1,0] # extract current domain_size from batch channel 1 , all constant take entry 0
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                
                inputs, outputs = train_sim(diff_ks, network_type, learning_task, inputs, netWGR, prediction_horizon, edges_torch, 
                                            domain_size=ds_curr, do_detach=WARMUP_STEPS , detach_interval=params['detach_interval'])

                if WARMUP_STEPS>0: # simply remove initial warmup steps for loss
                    outputs = outputs[:,WARMUP_STEPS:,:]
                    labels  = labels[ :,WARMUP_STEPS:,:]

                loss_value = loss(outputs, labels) # regular DP , no detach
                loss_value.backward()
                optimizer.step()

                if ((epoch*len(datasetKS)+i)%INTER_OUT)==0: 
                    log(f's[{epoch},{i}] with gradient WGR loss: {(loss_value.item()*10000.) :.3f}')
                if ((epoch*len(datasetKS)+i)%INTER_LRD)==0: 
                    scheduler.step()
                    
            if params['save_inter']>0 and epoch%params['save_inter'] == (params['save_inter']-1):
                torch.save(netWGR.state_dict(), "model-wgr-{:03d}.pickle".format( epoch ))
        # print loss 
        log(f'e[{epoch + 1}] with gradient WGR loss: {(loss_value.item()*10000.) :.3f}')
        torch.save(netWGR.state_dict(), "model-wgr.pickle") # old: model-dp.pickle
    else:
        netWGR.load_state_dict(torch.load("model-wgr.pickle", map_location=device ))
        log("loaded model-wgr.pickle")



# ------------------------------------------------------------------------------------------------------------------------------
# part X, run "reference physics loss" with gradients (formerly unsupervised DP)
# optional! not active for now...

netPHY = None
if params['train_physloss']:

    torch.manual_seed(params['seed']) # reset seed for network init & training
    np.random.seed(params['seed'])
    if network_type == 'GCN':
        netPHY = GCN(hidden_channels=channels, depth=depth, device=device, edgeconv=1, direction_ft= direction_ft)
        if compile:
            netPHY = torch_geometric.compile(netPHY)        
        optimizer = torch.optim.Adam(netPHY.parameters(), lr = 1e-3*params['lr_factor'])
    elif network_type == 'CNN':
        netPHY = ConvResNet1D(channels, depth, device=device)
        if compile:
            netPHY = torch_geometric.compile(netPHY)        
        optimizer = torch.optim.Adam(netPHY.parameters(), lr = 1e-4*params['lr_factor'])

    if params["train_continue"]>=0: # load pre-trained model to continue training
        if params["train_continue_loadone"]==0: 
            modelname = "model-phy.pickle"
        else:
            modelname = "model-one.pickle" # forced use of ONE as pretraining version
        netPHY.load_state_dict(torch.load(LOADDIR+"/"+modelname, map_location=device ))
        log('Continuing training from '+LOADDIR+"/"+modelname)

    if DO_TRAIN:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = lr_gamma)
        datasetKS.reset()        

        for epoch in range(epochs):
            for i, data in enumerate(datasetKS, 0):
                inputs, labels = data
                ds_curr = inputs[0,1,0] # extract current domain_size from batch channel 1
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                inputs, outputs = train_sim(diff_ks, network_type, learning_task, inputs, netPHY, prediction_horizon, edges_torch, 
                                            domain_size=ds_curr, do_detach=WARMUP_STEPS , detach_interval=params['detach_interval'])

                if WARMUP_STEPS>0: # simply remove initial warmup steps for loss
                    outputs = outputs[:,WARMUP_STEPS:,:]
                    labels  = labels[ :,WARMUP_STEPS:,:]

                # unsupervised for 1,2,3 steps (3 is max):
                # could simply be added, but needs to be limited by prediction horizon steps
                loss_value =      unsupervised_loss(torch.concat([inputs[0],outputs],axis=1), diff_ks, domain_size=ds_curr,steps=1) 
                if prediction_horizon>1:
                    loss_value += unsupervised_loss(torch.concat([inputs[0],outputs],axis=1), diff_ks, domain_size=ds_curr,steps=2) 
                if prediction_horizon>2:
                    loss_value += unsupervised_loss(torch.concat([inputs[0],outputs],axis=1), diff_ks, domain_size=ds_curr,steps=3) 

                loss_value.backward()
                optimizer.step()

                if ((epoch*len(datasetKS)+i)%INTER_OUT)==0: 
                    log(f's[{epoch},{i}] phys loss PHY: {(loss_value.item()*10000.) :.3f}')
                if ((epoch*len(datasetKS)+i)%INTER_LRD)==0: 
                    scheduler.step()
                    
            if params['save_inter']>0 and epoch%params['save_inter'] == (params['save_inter']-1):
                torch.save(netPHY.state_dict(), "model-phy-{:03d}.pickle".format( epoch ))
        # print loss 
        log(f'e[{epoch + 1}] phys loss PHY: {(loss_value.item()*10000.) :.3f}')
        torch.save(netPHY.state_dict(), "model-phy.pickle")
    else:
        netPHY.load_state_dict(torch.load("model-phy.pickle", map_location=device ))
        log("loaded model-phy.pickle")





# ------------------------------------------------------------------------------------------------------------------------------
# PART 4: 1-STEP SUPERVISED TRAINING , naive variant

netONE = None
if params['train_onestep']:
    torch.manual_seed(params['seed']) # reset seed for network init & training
    np.random.seed(params['seed'])
    if network_type=='GCN':
        netONE = GCN(hidden_channels=channels, depth=depth, device=device, edgeconv=1, direction_ft=direction_ft)
        if compile:
            netONE = torch_geometric.compile(netONE)
        optimizer = torch.optim.Adam(netONE.parameters(), lr=1e-3*params['lr_factor'])
    elif network_type=='CNN':
        netONE = ConvResNet1D(channels, depth, device=device)
        if compile:
            netONE = torch.compile(netONE)
        optimizer = torch.optim.Adam(netONE.parameters(), lr=1e-4*params['lr_factor'])    

    if params["train_continue"]>=0: # load pre-trained model to continue training
        netONE.load_state_dict(torch.load(LOADDIR+"/model-one.pickle", map_location=device ))
        log('Continuing training from '+LOADDIR+"/model-one.pickle")

    if DO_TRAIN:
        # reinit data for horizon=1
        prediction_horizon_one = 1 # reduced horizon
        torch_inputs_one = []
        torch_outputs_one = []
        for di in range(NUM_DS_START,NUM_DS_START+NUM_DS):
            dataset = dataset_org[di-NUM_DS_START] # starts at 0, not NUM_DS_START

            torch_input = np.array([stack_prediction_inputs(d, prediction_horizon_one,1) for d in dataset])
            torch_input = torch.Tensor(torch_input.reshape(-1,1,torch_input.shape[-1]))

            # add/concat constant channel encoding domain size; warning only single size supported per batch at the moment
            torch_input_ds = torch.ones( torch_input.shape ) * domain_size * dataset_dssizes[di] 
            torch_input = torch.concat( [torch_input, torch_input_ds], axis=1)
            torch_inputs_one.append( torch_input )

            torch_output = np.array([stack_prediction_targets(d, prediction_horizon_one,1) for d in dataset])
            torch_outputs_one.append( torch.Tensor(torch_output.reshape(-1,prediction_horizon_one, torch_output.shape[-1])) )

        # enlarge batch size by prediction horizon of previous runs 
        # to match other runs as closely as possible
        # the larger batch size reduces the number of updates, hence the epochs below 
        # are increased by the same factor
        datasetKS_one = DatasetKS(torch_inputs_one, torch_outputs_one, params['batch_size'] * prediction_horizon)
        datasetKS_one.reset()
        log("data set size for ONE "+str(len(datasetKS_one)) +", epochs increased by prediction_horizon to: "+str(epochs * prediction_horizon) )

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = lr_gamma)
        loss = torch.nn.MSELoss() # == supervised_loss()

        for epoch in range(epochs * prediction_horizon): # longer
            datasetKS_one.reset()
            for i, data in enumerate(datasetKS_one, 0):
                inputs, labels = data
                ds_curr = inputs[0,1,0] # extract current domain_size from batch channel 1 , all constant take entry 0
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                inputs, outputs = train_sim(diff_ks, network_type, learning_task, inputs, netONE, prediction_horizon_one, edges_torch, 
                                            domain_size=ds_curr ) # no detaching for ONE

                loss_value = loss(outputs, labels)
                loss_value.backward()
                optimizer.step()

                if ((epoch*len(datasetKS_one)+i)%INTER_OUT)==0: 
                    log(f's[{epoch},{i}] one-step loss ONE: {(loss_value.item()*10000.) :.3f}')
                if ((epoch*len(datasetKS_one)+i)%(INTER_LRD))==0: 
                    scheduler.step()
            
            #log( [epoch, (params['save_inter'] * prediction_horizon) , (params['save_inter'] * prediction_horizon -1)] )
            if params['save_inter']>0 and epoch%(params['save_inter'] * prediction_horizon) == (params['save_inter'] * prediction_horizon -1):
                torch.save(netONE.state_dict(), "model-one-{:03d}.pickle".format( int((epoch+1) / prediction_horizon -1) ))
                log("   saved model-one-{:03d}.pickle".format( int((epoch+1) / prediction_horizon -1) ) )
        # print loss
        log(f'e[{epoch + 1}] one-step loss ONE: {(loss_value.item()*10000.) :.3f}')
        torch.save(netONE.state_dict(), "model-one.pickle")
    else:
        netONE.load_state_dict(torch.load("model-one.pickle", map_location=device ))
        log("loaded model-one.pickle")






# ------------------------------------------------------------------------------------------------------------------------------

if not params['do_eval'] > 0:
    log("No eval, done")
    logging.shutdown()
    exit(0)

log("Running evaluation ...")

torch.manual_seed(params['seed']) # reset seed for eval as well...
np.random.seed(params['seed'])

# PART 5: evaluate versions

# plotting
import matplotlib
import matplotlib.gridspec as gridspec
matplotlib.__version__

# plot settings
horizon = 350
plot_increment = 1
showMin = -1. ; showMax = 1.

# evaluation with 5 tests
NUM_TESTS = 5
u_tests = [] 
for i in range(NUM_TESTS):
  with math.precision(64):
      x = domain_size*math.tensor(np.arange(0,diff_ks.resolution),spatial('x'))/diff_ks.resolution 
      # initialize 5 new variations , one (case 0) with only cos()
      u_test_n = [ math.expand( math.cos(2*x) + 0.12*i*math.cos(2*math.PI*x/domain_size) * (1-2.7*i*math.sin(2*math.PI*x/domain_size) ), batch(b=1)) ] 
      u_iter = u_test_n[0]
      for i in range(1000):
          u_iter = diff_ks.etrk2(u_iter,domain_size)
          u_test_n.append(u_iter)
  u_tests.append( tensor(u_test_n,instance('time') , u_iter.shape).numpy(['b','time', 'x']).astype(np.single) )


# test frames
tfs_nog = []; tfs_wgr  = []; tfs_phy = []; tfs_one = []
for i in range(NUM_TESTS):
    test_frame_in = torch.tensor(u_tests[i][0:1,100:101,:])

    if netNOG:
        tf_out_nog = run_sim(device, diff_ks, network_type, learning_task,  test_frame_in.to(device) , netNOG , horizon=horizon, domain_size=domain_size, edges_torch=edges_torch)
        tfs_nog.append( tf_out_nog )

    if netWGR:
        tf_out_wgr  = run_sim(device, diff_ks, network_type, learning_task,  test_frame_in.to(device) , netWGR  , horizon=horizon, domain_size=domain_size, edges_torch=edges_torch)
        tfs_wgr.append( tf_out_wgr ) 

    if netPHY:
        tf_out_phy = run_sim(device, diff_ks, network_type, learning_task,  test_frame_in.to(device) , netPHY , horizon=horizon, domain_size=domain_size, edges_torch=edges_torch)
        tfs_phy.append( tf_out_phy ) 

    if netONE:
        tf_out_one = run_sim(device, diff_ks, network_type, learning_task,  test_frame_in.to(device) , netONE , horizon=horizon, domain_size=domain_size, edges_torch=edges_torch)
        tfs_one.append( tf_out_one ) 

# print stats

if netNOG:
    for i in range(NUM_TESTS):
        test_frame_out = tfs_nog[i]
        stepped_prediction    = test_predict(diff_ks, test_frame_out, domain_size)
        log("Case "+format(i)+", NOG losses: "+format( [np.sum((test_frame_out[:,0,:]-u_tests[i][0,100:101+horizon,:])**2) ,
                np.sum((test_frame_out[1:,0,:]-stepped_prediction)**2)] ))

if netWGR:
    for i in range(NUM_TESTS):
        test_frame_out_wgr = tfs_wgr[i]
        stepped_prediction_wgr = test_predict(diff_ks, test_frame_out_wgr, domain_size)
        log("Case "+format(i)+", WGR losses: "+format( [np.sum((test_frame_out_wgr[:,0,:]-u_tests[i][0,100:101+horizon,:])**2) ,
                np.sum((test_frame_out_wgr[1:,0,:]-stepped_prediction_wgr)**2)] ))

if netPHY:
    for i in range(NUM_TESTS):
        stepped_prediction_wgr = test_predict(diff_ks, tfs_phy[i], domain_size)
        log("Case "+format(i)+", PHY losses: "+format( [np.sum((tfs_phy[i][:,0,:]-u_tests[i][0,100:101+horizon,:])**2) ,
                np.sum((tfs_phy[i][1:,0,:]-stepped_prediction_wgr)**2)] ))

if netONE:
    for i in range(NUM_TESTS):
        stepped_prediction_wgr = test_predict(diff_ks, tfs_one[i], domain_size)
        log("Case "+format(i)+", ONE losses: "+format( [np.sum((tfs_one[i][:,0,:]-u_tests[i][0,100:101+horizon,:])**2) ,
                np.sum((tfs_one[i][1:,0,:]-stepped_prediction_wgr)**2)] ))


# plot

# sanity check, plot all 5 as images
if params['write_images']:
    for i in range(NUM_TESTS):
        u_test = u_tests[i] # select case i

        if netNOG:
            test_frame_out_nog = tfs_nog[i]
            plt.figure(figsize=(50,10))
            plt.imshow( np.clip(np.transpose(test_frame_out_nog[::plot_increment,0,:]) , showMin,showMax))
            plt.title(r'No grad')
            plt.savefig("test"+str(i)+"_2nog.pdf", format='pdf')
            plt.close('all')

        if netWGR:
            test_frame_out_wgr = tfs_wgr[i]
            plt.figure(figsize=(50,10))
            plt.imshow( np.clip(np.transpose(test_frame_out_wgr[::plot_increment,0,:]) , showMin,showMax))
            plt.title(r'W grad')
            plt.savefig("test"+str(i)+"_3wgr.pdf", format='pdf')
            plt.close('all')

        plt.figure(figsize=(50,10))
        plt.imshow(np.clip(np.transpose(u_test[0,100:100+horizon:plot_increment,:]) , showMin,showMax))
        plt.title(r'-reference-')
        plt.savefig("test"+str(i)+"_1ref.pdf", format='pdf')
        plt.close('all')

        if netPHY:
            plt.figure(figsize=(50,10))
            plt.imshow( np.clip(np.transpose(tfs_phy[i][::plot_increment,0,:]) , showMin,showMax))
            plt.title(r'Ref phy')
            plt.savefig("test"+str(i)+"_4phy.pdf", format='pdf')
            plt.close('all')

        if netONE:
            plt.figure(figsize=(50,10))
            plt.imshow( np.clip(np.transpose(tfs_one[i][::plot_increment,0,:]) , showMin,showMax))
            plt.title(r'1 step')
            plt.savefig("test"+str(i)+"_5one.pdf", format='pdf')
            plt.close('all')

# L2
if 1:
    #PLOT_THRESH = 1e20 # too high?
    PLOT_THRESH = 1e06
    for i in range(NUM_TESTS):
        u_test = u_tests[i] # select case i

        if netNOG:
            plt.plot( np.clip( np.sum((tfs_nog[i][:,0,:]-u_test[0,100:101+horizon,:])**2,    axis=-1), -PLOT_THRESH,PLOT_THRESH), color='y', label="NoG")
        if netWGR:
            plt.plot( np.clip( np.sum((tfs_wgr[i][:,0,:]-u_test[0,100:101+horizon,:])**2, axis=-1), -PLOT_THRESH,PLOT_THRESH), color='g', label="wGr")
        if netPHY:
            plt.plot( np.clip( np.sum((tfs_phy[i][:,0,:]-u_test[0,100:101+horizon,:])**2, axis=-1), -PLOT_THRESH,PLOT_THRESH), color='thistle', label="Phy")
        if netONE:
            plt.plot( np.clip( np.sum((tfs_one[i][:,0,:]-u_test[0,100:101+horizon,:])**2, axis=-1), -PLOT_THRESH,PLOT_THRESH), color='lightsteelblue', label="1St")

    plt.yscale('log')
    plt.title('Comparison L2',fontsize=16)
    plt.legend()
    #plt.show()
    plt.savefig("comp1_l2.pdf", format='pdf')
    plt.close('all')

# LP
if 1:
    for i in range(NUM_TESTS):
        u_test = u_tests[i] # select case i

        if netNOG:
            # store predictions? this code resimulates everything here...
            test_frame_out = tfs_nog[i]
            stepped_prediction    = test_predict(diff_ks, test_frame_out, domain_size)
            plt.plot( np.clip( np.sum((test_frame_out[1:,0,:]-stepped_prediction)**2,       axis=-1), -PLOT_THRESH,PLOT_THRESH), color='y', label="NoG"+str(i))
        if netWGR:
            stepped_prediction_wgr = test_predict(diff_ks, tfs_wgr[i], domain_size)
            plt.plot( np.clip( np.sum((tfs_wgr[i][1:,0,:]-stepped_prediction_wgr)**2, axis=-1), -PLOT_THRESH,PLOT_THRESH), color='g', label="wGr"+str(i))
        if netPHY:
            stepped_prediction_phy = test_predict(diff_ks, tfs_phy[i], domain_size)
            plt.plot( np.clip( np.sum((tfs_phy[i][1:,0,:]-stepped_prediction_phy)**2, axis=-1), -PLOT_THRESH,PLOT_THRESH), color='thistle', label="Phy"+str(i))
        if netONE:
            stepped_prediction_one = test_predict(diff_ks, tfs_one[i], domain_size)
            plt.plot( np.clip( np.sum((tfs_one[i][1:,0,:]-stepped_prediction_one)**2, axis=-1), -PLOT_THRESH,PLOT_THRESH), color='lightsteelblue', label="1St"+str(i))

    plt.yscale('log')
    plt.title('Comparison LP',fontsize=16)
    plt.legend()
    #plt.show()
    plt.savefig("comp2_lp.pdf", format='pdf')
    plt.close('all')


# simple mean, SN steps -1 to 1 -> compute abs around 1, show deviation
def rolling_mean(a,w=10):
    l = a.shape[0]-w
    out = np.copy(a[0:l])
    for i in range(1,w):
        out += a[(0+i):(l+i)]
    return(out/w)

# print smoothed version of mean absolute content in -1 to 1 range
if 1:
    offset=0 # start later, note reference (u_test) has 100 more at beginning
    end=200+offset
    for i in range(NUM_TESTS):
        u_test = u_tests[i] # select case i
        plt.plot( rolling_mean( np.clip( np.sum( np.abs(u_test[0,100+offset:100+end,:]   ), axis=-1)/SN -1, -1,1), 50) , color='lightgray', label="gt"+str(i))
        if netNOG:
            plt.plot( rolling_mean( np.clip( np.sum( np.abs(tfs_nog[i][offset:end,0,:]   ), axis=-1)/SN -1, -1,1), 50) , color='y', label="NoG"+str(i))
        if netWGR:
            plt.plot( rolling_mean( np.clip( np.sum( np.abs(tfs_wgr[i][offset:end,0,:]), axis=-1)/SN -1, -1,1), 50) , color='g', label="wGr"+str(i))
        if netPHY:
            plt.plot( rolling_mean( np.clip( np.sum( np.abs(tfs_phy[i][offset:end,0,:]), axis=-1)/SN -1, -1,1), 50) , color='thistle', label="Phy"+str(i))
        if netONE:
            plt.plot( rolling_mean( np.clip( np.sum( np.abs(tfs_one[i][offset:end,0,:]), axis=-1)/SN -1, -1,1), 50) , color='lightsteelblue', label="1St"+str(i))
    plt.title('Comparison mean',fontsize=16)
    plt.legend()
    plt.savefig("comp3_mean.pdf", format='pdf')
    plt.close('all')


# all done
time_end = time.time()
log(f'Total Script Run Time : {time_end - time_start}')
logging.shutdown()

