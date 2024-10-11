# -*- coding: utf-8 -*-
"""
...  KS evaluation script ...
"""

import os, argparse, subprocess, pickle, logging, time
parser = argparse.ArgumentParser(description='CmdLine Parser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(  '--horizon', default=750,  type=int, help='evaluation horizon, number of time steps to simulate')
parser.add_argument(  '--numtests', default=5,  type=int, help='number of tests to run per model and domain size')
parser.add_argument(  '--metric', default="rellinf", type=str, help='Choose metric to evaluate with: "rellinf" computes a relative LInf metric, "l2" computes an L-2 loss wrt ground truth for full length; "correlation" computes time until cross correlation drops below decorr_thresh; "divtime" performs stability check, ie measures time until divergence (error above --thresh) and labels sim results as Static, Unstable, Recurring or Regular; "divergence-time-simple" is same as divtime without classification; "lphys" only computes physics based loss (no GT needed) ')
parser.add_argument(  '--thresh', default=500.0, type=float, help='LP threshold for divergence time')
parser.add_argument(  '--decorr_thresh', default=0.2, type=float, help='LP threshold for divergence time')
parser.add_argument(  '--timestep', default=0.5, type=float, help='KS solver time step, all tests typically use 1/2')
parser.add_argument(  '--numds',    default=4,  type=int, help='number of domain sizes to run for all tests (warning , switch to single-DS manually if necessary)')
parser.add_argument(  '--outstart', default=0,  type=int, help='start nr of outNNN dirs')
parser.add_argument(  '--outend', default=1000,  type=int, help='optional: end nr of outNNN dirs, default of -1 means outstart+100')
parser.add_argument(  '--run_eval',  default=1,  type=int, help='Actually run model eval, or just load existing data and re-plot in selected evaldir (use --evaldir N)?')
parser.add_argument(  '--agg_mode',  default=0,  type=int, help='Aggregation mode for loss values, default is 0 (mean), alternatives max (1) or min (2 or -1), or mean relative-to-ONE mode (3)')  # relone rel-one mode is agg_mode 3
parser.add_argument(  '--evaldir',  default=-1,  type=int, help='When run_eval 0/off, load from and write here')
parser.add_argument(  '--limitmodels', default=-1,  type=int, help='limit nr of models to evaluate per case (mostly for debugging)')
parser.add_argument(  '--limitstring', default="",  type=str, help='limit evaluation to very specific models whose strID matches the given string, eg use 16,10 to include all 16,10 models; separate multiple ones with - ')
parser.add_argument(  '--sortgraph', default=1,  type=int, help='sort graph entries along X by their label info (roughly small to large)')
parser.add_argument(  '--write_images', default=0,  type=int, help='write out full simulation images, debugging')
parser.add_argument(  '--cpu', default=0,  type=int, help='force CPU only')
parser.add_argument(  '--evaluate_multiple_configs', default=0, type=int, help='Goes through various model config directories')
parser.add_argument(  '--configs_search_dir', default='./', type=str, help='Directory location to search')
parser.add_argument(  '--decorr_N', default=5, type=int, help='For correlation metric, steps below thresh; threshold')
parser.add_argument(  '--yscale_log', default=0, type=int, help='Error plot L2: use log for y axis (>0), otherwise linear (0) , note: div-time always linear')
parser.add_argument(  '--print_corr_plots', default=0, type=int, help='Enables publishing of correlation plots to inspect where solution becomes static or recurring')
parser.add_argument(  '--compute_minmax',  default=1,  type=int, help='Recompute min/max values for graph')
parser.add_argument(  '--set_ylim_min',  default=-1,  type=float, help='Overrite compute-min for plotting (<= -1 is off)')
parser.add_argument(  '--set_ylim_max',  default=-1,  type=float, help='Overrite compute-min for plotting (<= -1 is off)')
parser.add_argument(  '--default_learning_task', default="<unspecified>", type=str, help='For older models, run_params may not specify the learning task. Then it needs to be set manually here, either correct or predict')
parser.add_argument(  '--tag_sort',  default=0,  type=int, help='Integer tag to append to params pickel, to be used for eval and plotting (eg varying predhori); in contrast to eg batchsize, dont auto detect whether to use it')
parser.add_argument(  '--tag_default',  default=0,  type=int, help='Value to use as default for models that dont yet have the tag (when evaluating a mix of old and new models)')
parser.add_argument(  '--tag_labels',  default='', type=str, help='NYI , When using tag for sorting, use these labels, separated by comma')
parser.add_argument(  '--tag_replacefix',  default=0,  type=int, help='Manual fix up for certain "wrong" tags')
parser.add_argument(  '--load_phy_models',  default=0,  type=int, help='New: skip PHY models by default... set to 1 to load them again')
parser.add_argument(  '--rerun_eval_prefix', default="",  type=str, help='Additional prefix of run_eval is 0')
parser.add_argument(  '--self_connect_nodes', default=1, type=int, help="Choose to have connecting edges that connect node to itself in the connectivity graph, default value is true(1)")
parser.add_argument(  '--print_lpd_plot', default=0, type=int, help="Enables publishing of lossp ** 2  plots to inspect where solution becomes unstable")
parser.add_argument(  '--per_model_summary', default=0, type=int, help="Print per NN model mean and std-dev of losses (main output aggregates all), mostly for debugging and inspecting single models")
parser.add_argument(  '--timeerr_filter_name', default="", type=str, help="For agg mode 4, error over time plots, filter by model name (include those that match)")
parser.add_argument(  '--write_graph_data',  default=0,  type=int, help='Write data for graph generation')
parser.add_argument(  '--horizon_append_start', default=0,  type=int, help='Only append results for later section of run? useful for very long horizons, if only a section at the end should be evaluated')

pargs = parser.parse_args()
eval_params = {}
eval_params.update(vars(pargs))
print("Eval params: "+format(eval_params))


HORIZON = eval_params['horizon'] 
APPEND_START = eval_params['horizon_append_start'] # only start "recording" results after this many steps 
THRESH = eval_params['thresh'] 
if eval_params['outend'] == -1:
    eval_params['outend'] = eval_params['outstart'] + 100

# filename prefix for output files: first add custom prefix ("" be default) , then append
prefix = eval_params['rerun_eval_prefix']
if not eval_params['run_eval']:
    prefix = prefix+"re-" # dont overwrite

if eval_params['agg_mode']==0:
    pass
elif eval_params['agg_mode']==1:
    prefix = prefix+"max-"
elif eval_params['agg_mode']==2 or eval_params['agg_mode']==-1:
    prefix = prefix+"min-"
elif eval_params['agg_mode']==3:
    prefix = prefix+"relone-"
elif eval_params['agg_mode']==4:
    prefix = prefix+"time-"
elif eval_params['agg_mode']==5:
    prefix = prefix+"median-"
else:
    prefix = prefix+"unknown-mode-"

# Enable relative L2?
USE_RELL2 = False
if eval_params['metric']=='rellinf':
    USE_RELL2 = True
    eval_params['metric']='l2' # otherwise treat as l2

SN = 48

CWD = os.path.abspath(os.getcwd())

# ---

import os
import numpy as np
import scipy as scp
from phi.torch.flow import * 

DISABLE_GCN = int(os.environ.get("KS_DISABLE_GCN", "0"))>0
if not DISABLE_GCN:
    import torch_geometric

TORCH.set_default_device("GPU")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if eval_params['cpu'] > 0:
    device = 'cpu' ; TORCH.set_default_device("CPU") # for debugging
device = torch.device(device)

print("pytorch device: "+format(device))

torch.manual_seed(16) # fixed for now...
np.random.seed(16)

# plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# "list-append", create list if it doesnt exist yet in dict a with key k, then append b
def lappend(a, k, b):
    if k in a.keys():
        a[k].append(b)
    else:
        a[k] = [b]


# ---------------

from ks_solver import DifferentiableKS, test_predict, run_sim_eval

# solver instance , init upon first model load
diff_ks = None 
doublestep = 1 # always on
domain_size = 8 # default for now
domain_size_base = domain_size # for varying DS runs



from ks_networks import ConvResNet1D, GCN, stack_prediction_inputs, stack_prediction_targets, direction_feature

# build edge list
def I(i):
    return (i % SN)  # periodic

neighbour_range = 1
edge_index = []
for i in range(SN):
    if eval_params['self_connect_nodes']:
        edge_index.append([I(i), I(i)])
    for j in range(1, neighbour_range + 1):
        edge_index.append([I(i), I(i - j)])
        edge_index.append([I(i), I(i + j)])

edge_index = np.array(edge_index)
edges_torch = torch.from_numpy(edge_index.transpose()).to(device)
edges_torch = edges_torch.long()

# -----------------------------------------------------------------------------------------

# interval size corresponds to step from calculating the correlations, only used to calculate original index
# warning - the thresholds of this classification are hard coded for the KS setup
def result_type(corr_val_array, interval_size=5):
    corr_vals = np.array(corr_val_array)
    
    # Check if any corr val is beyond limit
    for i,corr_val in enumerate(corr_val_array):
        if corr_val > 5E2:
            return 'unstable', i * interval_size
    
    ## New Logic: Use Correlation plot peaks to determine result type
    peaks_arr = []
    peaks_arr_ind = []
    max_peak = -1
    for i in range(1,len(corr_val_array)-1):
        if abs(corr_val_array[i])>abs(corr_val_array[i+1]) and abs(corr_val_array[i])> abs(corr_val_array[i-1]):
            peaks_arr.append(corr_val_array[i])
            peaks_arr_ind.append(i)
            if corr_val_array[i] > max_peak:
                max_peak = corr_val_array[i]

    
    static_count = 0
    if len(peaks_arr) ==0 or max_peak<5:
        return "static", 0
    for i in range(len(peaks_arr)-1):
        if abs(peaks_arr[i]) > 1E3:
            return "unstable", peaks_arr_ind[i] * interval_size
        #  old detection... obsolete
        # if abs(peaks_arr[i])>90:
        #     diff = abs(peaks_arr[i] - peaks_arr[i+1])
        #     if diff>170:
        #         #start_corr_val = peaks_arr[i]
        #         start_index = peaks_arr_ind[i]
        #         return "recurring", start_index * interval_size
        #     elif diff<10:
        #         if static_count==0:
        #             start_index = peaks_arr_ind[i]
        #         static_count += 1
        #         if static_count == 3:
        #             return "static", start_index * interval_size
            
    return "regular", len(corr_val_array) * interval_size


def write_image(tf, model_type, plot_increment, showMin, showMax, result_txt, model_test_num, channels, depth, network_type, learning_task, metric, is_ref=False):
    fig = plt.figure(figsize=(50,10))
    if not is_ref:
        plt.imshow( np.clip(np.transpose(tf[::plot_increment,0,:]) , showMin,showMax))
    else:
        plt.imshow( np.clip(np.transpose(tf[200+APPEND_START:201+HORIZON:plot_increment,:]) , showMin,showMax))
    result = result_txt.split(' ')[0]
    plt.title(f'{model_type}')
    if metric=='divtime':
        plt.text(HORIZON/3 , HORIZON/6, result_txt, fontsize=75) 
        plt.savefig(EVALDIR+"/test"+str(model_test_num)+f"_{model_type}_{channels},{depth},{network_type},{learning_task},{result_txt}.pdf", format='pdf')
    else:
        plt.savefig(EVALDIR+"/test"+str(model_test_num)+f"_{model_type}_{channels},{depth},{network_type},{learning_task}.pdf", format='pdf')
    plt.close('all')


def compute_metric(lossp, loss2, corr_vals, horizon, THRESH, 
            EVALDIR, test_num, model_count, model_type, channels, depth, network_type, learning_task, enable_type):

    # compute aggregated physics loss for classification (only needed for "divtime")
    lossp = np.clip(lossp,-1e12,1e12)
    np.nan_to_num( lossp , copy=False, nan=1e13, posinf=1e14, neginf=-1e14)  
    lpd = np.sum( lossp**2 , axis=-1 ) 

    # only necessary for divergence time metrics:
    if enable_type=="divtime" or enable_type=="divergence-time-simple": 
        last_stable_frame = horizon
        for j in range(horizon):
            if lpd[j]>THRESH: 
                last_stable_frame = j
                break
    
    if enable_type=="divtime":

        # Plotting Correlation Graphs
        if eval_params['print_corr_plots']:
            log(f"Length of Corrval Array: {len(corr_vals)}")
            plt.plot(range(len(corr_vals)),corr_vals)
            plt.text(10, 10, ' '.join(str(np.round(x,1)) for x in corr_vals), fontsize=2)
            plt.xlabel('Time frame')
            plt.ylabel('Correlation Value w.r.t last frame')
            plt.title('Correlation Values')
            plt.savefig(EVALDIR+"/test"+str(100*model_count + test_num)+f"corr_plot_{model_type}_{channels},{depth},{network_type},{learning_task}.pdf", format='pdf')
            
            plt.close('all')

        # Plotting Loss ** 2 plot
        if eval_params['print_lpd_plot']:
            plt.plot(range(len(lpd)),lpd)
            plt.text(10, 10, ' '.join(str(np.round(x,1)) for x in lpd), fontsize=2)
            plt.xlabel('Time frame')
            plt.ylabel('lpd')
            plt.title('lpd = lossp**2 Values')
            plt.savefig(EVALDIR+"/test"+str(100*model_count + test_num)+f"lpd_plot_{model_type}_{channels},{depth},{network_type},{learning_task}.pdf", format='pdf')

        if last_stable_frame < horizon: 
            return "unstable", last_stable_frame
        else:
            result_label, loc = result_type(corr_vals)
            if result_label == 'unstable':
                return 'unstable', last_stable_frame
            else:
                return result_label, loc

    elif enable_type=="divergence-time-simple": 
        if last_stable_frame < horizon:
            return 'unstable', last_stable_frame
        else: return 'regular', last_stable_frame

    elif enable_type=="correlation": 
        # exceed 0.8 for n frames
        decorr_thresh = eval_params['decorr_thresh']
        decorr_N = eval_params['decorr_N']
        last_corr_frame = -1
        decorr_count = -1
        for j in range(horizon):
            if corr_vals[j]<decorr_thresh and decorr_count>=decorr_N: 
                break # done!
            if corr_vals[j]<decorr_thresh and last_corr_frame>=0: 
                decorr_count += 1
            if corr_vals[j]<decorr_thresh and last_corr_frame<0: 
                last_corr_frame = j
            if corr_vals[j]>=decorr_thresh and last_corr_frame>=0: 
                last_corr_frame = -1
                decorr_count = -1
        if last_corr_frame<0: 
            last_corr_frame = horizon
            return 'unstable', last_corr_frame
        return 'regular', last_corr_frame

    elif enable_type=="l2": 
        if not USE_RELL2:
            lp2 = np.mean( loss2**2 , axis=-1 ) 
        else:
            lp2 = np.mean( loss2 , axis=-1 ) # square done above now!
        dtl2 = np.mean( np.clip(lp2, -30, 30) ) # clamp more
        return 'regular', dtl2

    elif enable_type=="lphys": 
        # lossp has shape [time step horizon, samples]
        lp2 = np.mean( lossp**2 )  # lossp is already clamped above
        return 'regular', lp2

    else: 
        log("Error, ubknown metric "+enable_type); exit(1)


def print_stats(a_in,name=""):
    a = np.asarray(a_in).flatten()
    print("stats "+name+" "+format(np.asarray(a_in).shape)+" "+format([np.min(a),np.max(a),np.mean(np.abs(a)),np.std(np.abs(a))]))


# ------------------------------------------------------------------------------------------------------------------------------


# eval: write to evalXXX
if eval_params['evaldir']<0:
    for i in range(1000):
        EVALDIR = CWD+"/"+"eval{:03d}".format(i)
        if not os.path.exists(EVALDIR):
            if eval_params['run_eval']:
                os.mkdir(EVALDIR)
                print("eval output dir "+EVALDIR)
            else:
                EVALDIR = CWD+"/"+"eval{:03d}".format(i-1) # re-run in last dir
                print("using last eval output dir "+EVALDIR)

            break
else:
    EVALDIR = CWD+"/"+"eval{:03d}".format(eval_params['evaldir'])

logname = prefix+'run.log'

# start log
mylog = logging.getLogger()
mylog.addHandler(logging.StreamHandler())
mylog.setLevel(logging.INFO)
mylog.addHandler(logging.FileHandler(EVALDIR+'/'+logname))
mylog.info("Eval params for log: "+str(eval_params))

# print and log
def log(s):
    mylog.info(s)

from datetime import datetime
log("Eval in "+EVALDIR+" at "+str(datetime.now()) )

# dont use print() from now on! only log() ...

# ---


# init test case data

NUM_TESTS = eval_params['numtests']
if NUM_TESTS>20:
    print("Error, max 20 tests per DS for now")
    exit(1)
NUM_DS  = eval_params['numds']

# global test data array, either load or compute
u_tests_all    = [] # all tests together, for convenience , NUMDS_*NUM_TESTS
u_tests_all_ds = [] # per test domain sizes for all tests together
u_tests_perds  = [] # per DS list , each NUM_TESTS (for batched eval)

testset_dssizes  = [1.40,   0.85,   1.15,   0.93   ] # tougher extrapolation cases first
testset_dssuffix = ["1_40", "0_85", "1_15", "0_93" ]

def init_testdata(diff_ks):
    global NUM_TESTS, domain_size, eval_params, u_tests_perds, u_tests_all, u_tests_all_ds
    if len(u_tests_all)>0:
        return # just do once

    for di in range(NUM_DS):
        domain_size = domain_size_base * testset_dssizes[di] 
        torch.manual_seed(di) # create repeatable conditions

        # note, not batched for now! a bit slow...
        fname = './ks-testdata'+testset_dssuffix[di]+'.npz'
        if not os.path.exists(fname):
            log("Generating testset "+str(di)+", for domain size "+str(domain_size)+" for "+fname)
            u_tests1 = [] 
            for i in range(20): # use max 20 , reduce to NUM_TESTS later on
                with math.precision(64):
                    x = domain_size*math.tensor(np.arange(0,diff_ks.resolution),spatial('x'))/diff_ks.resolution 
                    cosfac = -0.1 + 0.2*np.random.sample() # -0.1 to 0.1 
                    sinfac = -2.0 + 4.0*np.random.sample() # -4 to 4 
                    u_test_n = [ math.expand( math.cos(math.PI*x) + cosfac*math.cos(2*math.PI*x/domain_size) * (1-sinfac*math.sin(2*math.PI*x/domain_size) ), batch=1) ] 
                    u_iter = u_test_n[0]
                    for i in range(5200):
                        u_iter = diff_ks.etrk2(u_iter,domain_size=domain_size)
                        u_test_n.append(u_iter)
                u_tests1.append( tensor(u_test_n,instance('time') , u_iter.shape).numpy(['b','time', 'x']).astype(np.single) )
            u_tests1 = np.asarray(u_tests1)
            u_tests1 = u_tests1[:,200:,:] # crop first 200 warmup steps, note code below could be simplified to start at #0, starting at #200 not "wrong" though
            log("generated testdata "+format([u_tests1.shape, domain_size])+' at '+fname)
            np.savez(fname,u_tests1)
            print_stats(u_tests1, "final"); #exit(1)
        else:
            u_tests1 = np.load(fname)['arr_0']
            log("testdata "+fname+" loaded for "+format([u_tests1.shape, domain_size]) )

        u_tests1 = u_tests1[0:NUM_TESTS,:] # reduce number of tests if necessary
        #log("testdata shape "+format([u_tests1.shape]) )

        u_tests_perds.append(u_tests1)
        print(u_tests1.shape)
        for i in range(NUM_TESTS):
            u_tests_all.append( u_tests1[i])
            u_tests_all_ds.append( domain_size )
        #log("testdata len "+format(len(u_tests_all)) )

    domain_size = domain_size_base
    return None


# generate test data and references, solver only needed for this, could be skipped if data is loaded
diff_ks_testdata = DifferentiableKS(resolution=SN, dt=eval_params['timestep'] )
#log("KS solver initialized using dt="+str(eval_params['timestep'])+" ")
init_testdata(diff_ks_testdata) 



# ------------------------------------------------------------------------------------------------------------------------------


def load_net(fn, modelnames, params):
    global device
    net = None
    network_type = params['network_type']

    channels = params['channels']
    depth = params['depth']
    if os.path.exists(fn):
        if network_type == 'GCN':
            if params['compile']:
                edgeft_channels = 4 # needed for pytorch JIT compilation bugs, remove at some point
                direction_ft = direction_feature(edges_torch, edge_dim_channels=edgeft_channels, device=device)
                net = torch_geometric.compile(GCN(channels, depth, device=device, edgeconv=1, direction_ft=direction_ft)) 
            else:
                edgeft_channels = 1
                direction_ft = direction_feature(edges_torch, edge_dim_channels=edgeft_channels, device=device)
                net = GCN(channels, depth, device=device, edgeconv=1, direction_ft=direction_ft)
        elif network_type == 'CNN':
            if params['compile']:
                net = torch.compile(ConvResNet1D(channels, depth, device=device))
            else:
                net = ConvResNet1D(channels, depth, device=device) 
        else:
            log("Unknown network type! "+params['network_type'])
            exit(1)
        
        net.load_state_dict(torch.load(fn, map_location=device ))
        modelnames += fn+" "
    return net, modelnames



# compute correlation values between ref and the steps of in_array
# note, step here should match interval_size of result_type()
# (ref is the very last frame of in_array for divergence time, while correlation as metric uses the GT values)
def compute_corr(horizon, ref, in_array, step=5, normalize=False):
    corrs = []
    for horizon_val in range(0,horizon,step):
        u = in_array[horizon_val,0,:]
        if len(ref.shape)==2: # if sequence is given, compare step by step
            r = ref[horizon_val,:]
        else:
            r = ref # single field
        if normalize:
            r = (r - np.mean(r)) / (np.std(r) * len(r) + 1e-20)
            u = (u - np.mean(u)) / (np.std(u) + 1e-20)
        corrs.append( np.correlate(r, u) )
    return corrs


# setup manual limit string model selection
limit_models=[]
limit_models=[]
if len(eval_params['limitstring'])>0:
    limit_models1 = eval_params['limitstring'].split('-') 
    for m in limit_models1:
        limit_models.append( m ) 
    print("limitstring active "+format(limit_models)); # exit(1)

# ------------------------------------------------------------------------------------------------------------------------------

# starting main eval
# collect models from all out directories

losses = {}
divtime = {} # times until divergence
divtype = {} # Type of divergence {Unstable, Recurring, Static or Regular (No divergence)}
runparams_all = {} # keep around all parameters
model_count = 1
dtpartial = -2.0 # initialized upon first diff_ks solver init (can be -1 from default ks-train init)

different_archs = {} # check whether we have a mix of CNN and GCNs
different_tasks = {} # check whether we have a mix of correction and prediction tasks
different_bss   = {} # check whether we have a mix of batch sizes

loaded_model_count = 0 

# Function defined to read and evaluate models from outXXX 
# and minimize code redundancy 
def eval_in_outdirs():
    global dtpartial, diff_ks, loaded_model_count
    model_count=1
    CWDo = os.path.abspath(os.getcwd()) # main directory for outNNN search

    for i in range(eval_params['outstart'],eval_params['outend']): 
        OUTDIR = "out{:03d}".format(i)
        if not os.path.exists(OUTDIR):
            continue

        if not os.path.exists(OUTDIR+'/params.pickle'):
            continue

        use_dir = 0
        if  os.path.exists(OUTDIR+'/model-nog.pickle') or \
            os.path.exists(OUTDIR+'/model-one.pickle') or \
            os.path.exists(OUTDIR+'/model-phy.pickle') or \
            os.path.exists(OUTDIR+'/model-wgr.pickle'): use_dir = 2

        if use_dir==0:
            log("Warning - incomplete dir "+OUTDIR)
            continue

        time_start = time.time()
        log("checking "+OUTDIR)
        os.chdir(OUTDIR)

        # load settings
        with open('params.pickle', 'rb') as f: run_params = pickle.load(f)
        
        if 'network_type' not in run_params.keys():
            run_params['network_type'] = 'CNN' # use CNN as default , for old runs
        if 'learning_task' not in run_params.keys():
            if eval_params['default_learning_task'] == "predict":
                run_params['learning_task'] = 'predict' 
            elif eval_params['default_learning_task'] == "correct":
                run_params['learning_task'] = 'correct'
            else:
                log("error: model has no learning task in run_params, manually specify via --default_learning_task ")
                exit(1)
        if 'batch_size' not in run_params.keys():
            run_params['batch_size'] = 128

        # partial steps
        if 'dtpartial' not in run_params.keys():
            run_params['dtpartial']=-1.0

        if run_params['dtpartial']!=-1. and run_params['learning_task']!='corr-pred-transition':
            log("Error, dtpartial settings require learning task 'corr-pred-transition' "); exit(1)

        if 'compile' not in run_params.keys():
            run_params['compile']=0

        channels = run_params['channels']
        depth    = run_params['depth']
        network_type  = run_params['network_type']
        learning_task = run_params['learning_task']
        
        tagstr = '' # use tags?
        if eval_params['tag_sort']:
            if 'tag' not in run_params.keys():
                run_params['tag'] = eval_params['tag_default']
            elif run_params['tag']==0: # also replace 0
                run_params['tag'] = eval_params['tag_default']

            # manual fix for certain "old" tags... eg default runs for lessdata2 with tag 12 -> set to 0
            if eval_params['tag_replacefix']==1:
                if run_params['tag']==12:
                    run_params['tag'] = 0

            tagstr = ","+str(run_params['tag'])

        strIDnog = "%s,%d,%d,nog,%s,%d%s" % (network_type,channels,depth,learning_task,run_params['batch_size'],tagstr)
        strIDwgr = "%s,%d,%d,wgr,%s,%d%s" % (network_type,channels,depth,learning_task,run_params['batch_size'],tagstr)
        strIDphy = "%s,%d,%d,phy,%s,%d%s" % (network_type,channels,depth,learning_task,run_params['batch_size'],tagstr)
        strIDone = "%s,%d,%d,one,%s,%d%s" % (network_type,channels,depth,learning_task,run_params['batch_size'],tagstr)
        different_archs[network_type] = 1
        different_tasks[learning_task] = 1
        different_bss[run_params['batch_size']] = 1

        found_limstrIDone = found_limstrIDnog = found_limstrIDwgr = found_limstrIDphy = False
        if len(eval_params['limitstring'])>0:
            for m in limit_models:
                if m in strIDone: found_limstrIDone=True
                if m in strIDnog: found_limstrIDnog=True
                if m in strIDwgr: found_limstrIDwgr=True
                if m in strIDphy: found_limstrIDphy=True
        else:
            # activate all if unused
            found_limstrIDone = found_limstrIDnog = found_limstrIDwgr = found_limstrIDphy = True


        # store eval_params
        runparams_all[strIDwgr] = run_params
        runparams_all[strIDnog] = run_params
        runparams_all[strIDphy] = run_params
        runparams_all[strIDone] = run_params
        
        losses2_nog, lossesp_nog = [],[] # l2 & lp losses
        losses2_wgr, lossesp_wgr = [],[] # l2 & lp losses
        losses2_phy, lossesp_phy = [],[] # l2 & lp losses
        losses2_one, lossesp_one = [],[] # l2 & lp losses

        # for divtk split later on
        if not strIDwgr in divtime.keys() and found_limstrIDwgr:
            log("  New case "+strIDwgr)
            divtime[strIDwgr] = []
            divtype[strIDwgr] = []
        if not strIDnog in divtime.keys() and found_limstrIDnog:
            log("  New case "+strIDnog)
            divtime[strIDnog] = []
            divtype[strIDnog] = []
        if not strIDphy in divtime.keys() and found_limstrIDphy:
            log("  New case "+strIDphy)
            divtime[strIDphy] = []
            divtype[strIDphy] = []
        if not strIDone in divtime.keys() and found_limstrIDone:
            log("  New case "+strIDone)
            divtime[strIDone] = []
            divtype[strIDone] = []

        if (not found_limstrIDwgr) and (not found_limstrIDnog) and (not found_limstrIDphy) and (not found_limstrIDone):
            log("  Skipping dir "+OUTDIR+" due to limitstr")
            os.chdir(CWDo) 
            continue

        # TODO, count per individual variant?
        if eval_params['limitmodels']>0 and len(divtime[strIDwgr])>=(eval_params['limitmodels']*NUM_TESTS):
            log("  Skipping dir "+OUTDIR+" due to # limitmodels")
            os.chdir(CWDo) 
            continue

        log(OUTDIR + " with "+format(run_params))
        #exit(0)

        # initialize KS solver for NNs, just re-init & overwrite
        if dtpartial!=run_params['dtpartial']:
            dt = eval_params['timestep']
            if run_params['dtpartial']>=0.:
                log("Partial time step active! "+str(run_params['dtpartial']) )
                dt = dt * run_params['dtpartial']
            diff_ks = DifferentiableKS(resolution=SN, dt=dt )
            dtpartial = run_params['dtpartial']
            log("KS solver re-initialized using dt="+str(dt) )

        # allocate and load models
        netNOG = None; netWGR = None
        netPHY = None; netONE = None

        modelnames = ""
        netNOG, modelnames = load_net('model-nog.pickle', modelnames, run_params)
        netWGR, modelnames = load_net('model-wgr.pickle', modelnames, run_params)
        
        netONE, modelnames = load_net('model-one.pickle', modelnames, run_params)
        if eval_params['load_phy_models']: # skip by default now
            netPHY, modelnames = load_net('model-phy.pickle', modelnames, run_params)

        log("  loaded models ["+modelnames+"] in "+OUTDIR)
        loaded_model_count += len(modelnames)

        # if limit strings were given, remove loaded models again if not matching...
        if len(eval_params['limitstring'])>0:
            if not found_limstrIDone: 
                netONE = None; log("Unloaded ONE")
            if not found_limstrIDnog: 
                netNOG = None; log("Unloaded NOG")
            if not found_limstrIDwgr: 
                netWGR = None; log("Unloaded WGR")
            if not found_limstrIDphy: 
                netPHY = None; log("Unloaded PHY")

        if 0: # DEBUG , test only until here
            del netNOG,netWGR,netPHY,netONE
            os.chdir(CWDo) 
            continue 

        tfs_nog = []; tfs_wgr  = []; tfs_phy  = []; tfs_one  = [] # results append to these
        time_run = time.time()

        # check batch split
        split = 1 # off by default
        # warning: split settings are hard-coded for 11GB GPUs and typical NN sizes for now...

        # heuristics for different cases, 
        if eval_params['horizon']>2400:
              if(run_params['network_type']=='CNN'):
                  # split more for divtime
                  if(run_params['numparams']>999000000):
                      split = int(NUM_TESTS/2) # only 2 at once, unused
                  elif(run_params['numparams']>1000000): # needed?
                      split = 4
                  elif(run_params['numparams']>100000):
                      split = 2
        elif eval_params['horizon']>400:
            if(run_params['network_type']=='CNN'):
                # split more for divtime
                if(run_params['numparams']>10000000):
                    split = int(NUM_TESTS/2) # only 2 at once?
                elif(run_params['numparams']>5000000):
                    split = 4
                elif(run_params['numparams']>1000000):
                    split = 2
            else: # GCNs
                if(run_params['numparams']>1000000):
                    split = int(NUM_TESTS/2) # only 2 at once , warning , slow!
                elif(run_params['numparams']>100000):
                    split = 4 # safe margin?
            #split=NUM_TESTS # one by one
        elif eval_params['horizon']>100:
            # only for 416,24 model
            if(run_params['numparams']>10000000):
                split = 2
        if split>1:
            log("split for "+str(run_params['numparams'])+" is "+str(split))

        for di in range(NUM_DS):
            domain_size = domain_size_base * testset_dssizes[di] 
            # create list of initial states for all tests, by default, the test runs start with time step 200
            test_frame_in = []
            for i in range(NUM_TESTS):
                test_frame_in.append( u_tests_perds[di][i][200:201,:] )
            test_frame_in = torch.tensor( np.stack(test_frame_in) ).to(device) # eval expects one new dimension...

            with torch.no_grad():
                run_sim_eval(device, diff_ks, network_type, learning_task,  test_frame_in , netNOG , horizon=HORIZON, domain_size=domain_size, edges_torch=edges_torch ,out_ret=tfs_nog , split=split, horizon_append_start=APPEND_START)
                run_sim_eval(device, diff_ks, network_type, learning_task,  test_frame_in , netWGR , horizon=HORIZON, domain_size=domain_size, edges_torch=edges_torch, out_ret=tfs_wgr , split=split, horizon_append_start=APPEND_START)
                run_sim_eval(device, diff_ks, network_type, learning_task,  test_frame_in , netPHY , horizon=HORIZON, domain_size=domain_size, edges_torch=edges_torch ,out_ret=tfs_phy , split=split, horizon_append_start=APPEND_START)
                run_sim_eval(device, diff_ks, network_type, learning_task,  test_frame_in , netONE , horizon=HORIZON, domain_size=domain_size, edges_torch=edges_torch ,out_ret=tfs_one , split=split, horizon_append_start=APPEND_START)
        domain_size = domain_size_base 

        # sanity check, plot all tests as images, mostly debugging
        showMin = -1. ; showMax = 1.
        showMin = -3. ; showMax = 3. # larger?
        plot_increment = 1
        h = eval_params['horizon']
        corr_val_nogs = []
        corr_val_ones = []
        corr_val_phys = []
        corr_val_wgrs = []
        for i in range(NUM_DS*NUM_TESTS):
            u_test = u_tests_all[i] # select case i

            if eval_params['metric']=="divtime" or eval_params['metric']=="divergence-time-simple": 
                if netNOG:
                    test_frame_out_nog = tfs_nog[i]                
                    corr_val_nogs.append( compute_corr(h, test_frame_out_nog[-1, 0, :], test_frame_out_nog) )
                if netWGR:
                    test_frame_out_wgr = tfs_wgr[i]                
                    corr_val_wgrs.append( compute_corr(h, test_frame_out_wgr[-1, 0, :], test_frame_out_wgr) )
                if netPHY:
                    test_frame_out_phy = tfs_phy[i]
                    corr_val_phys.append( compute_corr(h, test_frame_out_phy[-1, 0, :], test_frame_out_phy) )
                if netONE:
                    test_frame_out_one = tfs_one[i]
                    corr_val_ones.append( compute_corr(h, test_frame_out_one[-1, 0, :], test_frame_out_one) )

            elif eval_params['metric']=="correlation": 
                if netNOG:
                    corr_val_nogs.append( compute_corr(h, u_tests_all[i][200+APPEND_START:201+HORIZON,:], tfs_nog[i], step=1, normalize=True) )
                if netWGR:
                    corr_val_wgrs.append( compute_corr(h, u_tests_all[i][200+APPEND_START:201+HORIZON,:], tfs_wgr[i], step=1, normalize=True) )
                if netPHY:
                    corr_val_phys.append( compute_corr(h, u_tests_all[i][200+APPEND_START:201+HORIZON,:], tfs_phy[i], step=1, normalize=True) )
                if netONE:
                    corr_val_ones.append( compute_corr(h, u_tests_all[i][200+APPEND_START:201+HORIZON,:], tfs_one[i], step=1, normalize=True) )

            elif eval_params['metric']=="l2" or  eval_params['metric']=="lphys": 
                corr_val_nogs.append( [] ) # not used
                corr_val_wgrs.append( [] )
                corr_val_phys.append( [] )
                corr_val_ones.append( [] )

            else:
                log("\nError - unknown metric {}".format(eval_params['metric']) ); exit(1)

            if eval_params['write_images']>0 and eval_params['horizon_append_start']<=0: # only write reference if we're looking at the full data, ie no cutoff
                write_image(tf=u_test, model_type='ref', 
                    plot_increment=plot_increment, showMin=showMin, showMax=showMax, 
                    result_txt="   ", model_test_num=100*model_count+i, channels=channels, depth=depth, network_type = network_type, learning_task = learning_task,
                    metric=eval_params['metric'], is_ref=True)

        #exit(1) # debug only! write only once

        time_eval = time.time()

        # compute losses, and print stats
        # warning - losses are computed via sum, no normalization by cells 
        if netNOG:
            for i in range(NUM_DS*NUM_TESTS):
                u_test = u_tests_all[i]
                test_frame_out_nog = tfs_nog[i]
                stepped_prediction_nog = test_predict(diff_ks, test_frame_out_nog, domain_size=u_tests_all_ds[i])

                lp = test_frame_out_nog[1:,:][:,0,:]-stepped_prediction_nog
                lossesp_nog.append(lp)

                if not eval_params['metric']=="lphys": # skip L2 eval with GT for physics loss
                    if not USE_RELL2:
                        l2 = test_frame_out_nog[:,0,:]-u_test[200+APPEND_START:201+HORIZON,:]
                    else:
                        l2 = ((test_frame_out_nog[:,0,:]-u_test[200+APPEND_START:201+HORIZON,:] )**2) / (np.max(u_test[200+APPEND_START:201+HORIZON,:]**2) )
                else:
                    l2 = -1
                losses2_nog.append(l2)

                #log("Case "+format(i)+", NOG losses: "+format( [np.sum((l2)**2) , np.sum((lp)**2)] ))

                result, loc = compute_metric(lossp=lossesp_nog[i],loss2=losses2_nog[i],corr_vals=corr_val_nogs[i], horizon=HORIZON, THRESH=THRESH,
                                                EVALDIR=EVALDIR, test_num=i, model_count = model_count, model_type="nog", channels=channels, depth=depth, network_type = network_type, learning_task = learning_task,
                                                enable_type=eval_params['metric'])
                divtime[strIDnog].append( loc )
                divtype[strIDnog].append( result)

                if result == 'static' or result == 'recurring' or result == 'unstable':
                    result_txt = f"{result} from {loc} onwards"
                else:
                    result_txt = f"{result} solution"
                
                if eval_params['write_images']>0: 
                    write_image(tf=tfs_nog[i], model_type='NOG', 
                            plot_increment=plot_increment, showMin=showMin, showMax=showMax, 
                                result_txt=result_txt, model_test_num=100*model_count+i, channels=channels, depth=depth, network_type = network_type, learning_task = learning_task,
                                metric=eval_params['metric'])

                log("    "+strIDnog+" "+" Test "+format(i)+"   ->  "+format(loc))

            lappend( losses,strIDnog+"_l2", losses2_nog) # warning, not yet L2! just difference... can be negative
            lappend( losses,strIDnog+"_lp", lossesp_nog)

        if netWGR:
            for i in range(NUM_DS*NUM_TESTS):
                u_test = u_tests_all[i]
                test_frame_out_wgr = tfs_wgr[i]
                stepped_prediction_wgr = test_predict(diff_ks, test_frame_out_wgr, domain_size=u_tests_all_ds[i])

                lp = test_frame_out_wgr[1:,:][:,0,:]-stepped_prediction_wgr
                lossesp_wgr.append(lp)

                if not eval_params['metric']=="lphys": # skip L2 eval with GT for physics loss
                    if not USE_RELL2:
                        l2 = test_frame_out_wgr[:,0,:]-u_test[200+APPEND_START:201+HORIZON,:]
                    else:
                        l2 = ((test_frame_out_wgr[:,0,:]-u_test[200+APPEND_START:201+HORIZON,:] )**2) / (np.max(u_test[200+APPEND_START:201+HORIZON,:]**2) )
                else:
                    l2 = -1
                losses2_wgr.append(l2)

                #log("Case "+format(i)+", WGR losses: "+format( [np.sum((l2)**2) , np.sum((lp)**2)] ))
                result, loc = compute_metric(lossp=lossesp_wgr[i],loss2=losses2_wgr[i], corr_vals=corr_val_wgrs[i], horizon=HORIZON, THRESH=THRESH,
                                                  EVALDIR=EVALDIR, test_num=i, model_count=model_count, model_type="wgr", channels=channels, depth=depth, network_type = network_type, learning_task = learning_task, 
                                                  enable_type=eval_params['metric'])
                divtime[strIDwgr].append( loc )
                divtype[strIDwgr].append( result)
                
                if result == 'static' or result == 'recurring' or result == 'unstable':
                    result_txt = f"{result} from {loc} onwards"
                else:
                    result_txt = f"{result} solution"
                
                if eval_params['write_images']>0: 
                    write_image(tf=tfs_wgr[i], model_type='WGR', 
                            plot_increment=plot_increment, showMin=showMin, showMax=showMax, 
                                result_txt=result_txt, model_test_num=100*model_count+i, channels=channels, depth=depth, network_type = network_type, learning_task = learning_task,
                                metric=eval_params['metric'])

                log("    "+strIDwgr+" "+" Test "+format(i)+"   ->  "+format(loc))
                

            lappend( losses,strIDwgr+"_l2", losses2_wgr) # warning, not yet L2! just difference... can be negative
            lappend( losses,strIDwgr+"_lp", lossesp_wgr)

        if netPHY:
            for i in range(NUM_DS*NUM_TESTS):
                u_test = u_tests_all[i]
                test_frame_out_phy = tfs_phy[i]
                stepped_prediction_phy = test_predict(diff_ks, test_frame_out_phy, domain_size=u_tests_all_ds[i])

                lp = test_frame_out_phy[1:,:][:,0,:]-stepped_prediction_phy
                lossesp_phy.append(lp)

                if not eval_params['metric']=="lphys": # skip L2 eval with GT for physics loss
                    if not USE_RELL2:
                        l2 = test_frame_out_phy[:,0,:]-u_test[200+APPEND_START:201+HORIZON,:]
                    else:
                        l2 = ((test_frame_out_phy[:,0,:]-u_test[200+APPEND_START:201+HORIZON,:] )**2) / (np.max(u_test[200+APPEND_START:201+HORIZON,:]**2) )
                else:
                    l2 = -1
                losses2_phy.append(l2)

                #log("Case "+format(i)+", PHY losses: "+format( [np.sum((l2)**2) , np.sum((lp)**2)] ))

                result, loc = compute_metric(lossp=lossesp_phy[i],loss2=losses2_phy[i], corr_vals=corr_val_phys[i], horizon=HORIZON, THRESH=THRESH,
                                                  EVALDIR=EVALDIR, test_num=i, model_count=model_count, model_type="phy", channels=channels, depth=depth, network_type = network_type, learning_task = learning_task, 
                                                  enable_type=eval_params['metric'])
                divtime[strIDphy].append( loc )
                divtype[strIDphy].append( result)

                if result == 'static' or result == 'recurring' or result == 'unstable':
                    result_txt = f"{result} from {loc} onwards"
                else:
                    result_txt = f"{result} solution"
                
                if eval_params['write_images']>0: 
                    write_image(tf=tfs_phy[i], model_type='PHY', 
                            plot_increment=plot_increment, showMin=showMin, showMax=showMax, 
                                result_txt=result_txt, model_test_num=100*model_count+i, channels=channels, depth=depth, network_type = network_type, learning_task = learning_task, 
                                metric=eval_params['metric'])

                log("    "+strIDphy+" "+" Test "+format(i)+"   ->  "+format(loc))

            lappend( losses,strIDphy+"_l2", losses2_phy) # warning, not yet L2! just difference... can be negative
            lappend( losses,strIDphy+"_lp", lossesp_phy)


        if netONE:
            for i in range(NUM_DS*NUM_TESTS):
                u_test = u_tests_all[i]
                test_frame_out_one = tfs_one[i]
                stepped_prediction_one = test_predict(diff_ks, test_frame_out_one, domain_size=u_tests_all_ds[i])

                lp = test_frame_out_one[1:,:][:,0,:]-stepped_prediction_one
                lossesp_one.append(lp)

                if not eval_params['metric']=="lphys": # skip L2 eval with GT for physics loss
                    if not USE_RELL2:
                        l2 = test_frame_out_one[:,0,:]-u_test[200+APPEND_START:201+HORIZON,:]
                    else:
                        l2 = ((test_frame_out_one[:,0,:]-u_test[200+APPEND_START:201+HORIZON,:] )**2) / (np.max(u_test[200+APPEND_START:201+HORIZON,:]**2) )
                else:
                    l2 = -1
                losses2_one.append(l2)

                result, loc = compute_metric(lossp=lossesp_one[i],loss2=losses2_one[i], corr_vals=corr_val_ones[i], horizon=HORIZON, THRESH=THRESH,
                                                  EVALDIR=EVALDIR, test_num=i, model_count=model_count, model_type="one", channels=channels, depth=depth, network_type = network_type, learning_task = learning_task, 
                                                  enable_type=eval_params['metric'])
                divtime[strIDone].append( loc )
                divtype[strIDone].append( result)
                if result == 'static' or result == 'recurring' or result == 'unstable':
                    result_txt = f"{result} from {loc} onwards"
                else:
                    result_txt = f"{result} solution"
                
                if eval_params['write_images']>0: 
                    write_image(tf=tfs_one[i], model_type='ONE', 
                            plot_increment=plot_increment, showMin=showMin, showMax=showMax, 
                                result_txt=result_txt, model_test_num=100*model_count+i, channels=channels, depth=depth, network_type = network_type, learning_task = learning_task,
                                metric = eval_params['metric'])

                log("    "+strIDone+" "+" Test "+format(i)+"   ->  "+format(loc))

            lappend( losses,strIDone+"_l2", losses2_one) # warning, not yet L2! just difference... can be negative
            lappend( losses,strIDone+"_lp", lossesp_one)

        # some profiling
        time_end = time.time()
        #log("    runtimes: run {:.2f}s , eval {:.2f}s , tot {:.2f}s".format(time_eval-time_run, time_end-time_eval, time_end-time_start ))

        model_count += 1
        # free model?
        del netNOG, netWGR, netPHY, netONE
        os.chdir(CWDo) 
        log("\nDone with runs for "+format(network_type)+", "+format(channels)+", "+format(depth)+", "+format(learning_task) ) # +", "+format(run_params['batch_size'])) 

# turn label into int
def label2int(l):
    l = l.split(",") ; ret = 0
    
    if eval_params['tag_sort']: # tags active?
        ret += 1000000000 * int(l[-1][1:])
        l = l[:-1] # remove tag
        
    if len(different_bss.keys())>1: # we have different batch sizes?
        # then remove last entry as BS , treat like arch (never vary BS and arch at same time...)
        ret += 10000000 * int(l[-1][1:])
        l = l[:-1] # remove

    if len(l)==2: # turn "NETWORKTYPE, CHANNELS, DEPTH" label into int
        ret += int(l[1]) * 10000 + int(l[0]) * 10 # depth & channels
    elif len(l)==3: # turn "NETWORKTYPE, CHANNELS, DEPTH" label into int
        ret += int(l[2]) * 10000 + int(l[1]) * 10 # depth & channels
        if l[0]=='G': # graph nets later
            ret += 10000000
    elif len(l)==4: # turn "TASK, NETWORKTYPE, CHANNELS, DEPTH" label into int
        ret += int(l[3]) * 10000 + int(l[2]) * 10 # depth & channels
        if l[1]=='G': # graph nets after CNNs
            ret += 10000000
        if l[0]=='p': # predict/corr last
            ret += 100000000
    else:
        log("invalid label "+format(l)); exit(1)

    # if l[3]=='nog': # NOG/WGR etc. not distinguished here, shared label
    #     ret += 1
    return(ret)



# actually run eval , or just load existing eval results?
if eval_params['run_eval']:

    if eval_params['evaluate_multiple_configs']:
        org_cwd = os.getcwd()
        for config_folder in os.listdir(eval_params['configs_search_dir']):
            label = config_folder
            print(label)
            try:
                [channels_, depth_] = label.split(",")
            except:
                continue
            print(f"Currently Evaluating for Model Channels, Depth:{channels_},{depth_}")

            os.chdir(config_folder)
            eval_in_outdirs()
            os.chdir(org_cwd)
    else:
        eval_in_outdirs()

    if loaded_model_count==0:
        log("\nError - no models found in search range out {} to {}...".format(eval_params['outstart'] , eval_params['outend']) ); exit(1)

    

    # ------------------------------------------------------------------------------------------------------------------------------

    # collect and sort

    have_multiple_tasks = len(different_tasks.keys())>1
    have_multiple_archs = len(different_archs.keys())>1

    divt_out_nog = []; divt_out_nog_std = []
    divt_out_wgr = []; divt_out_wgr_std = []; 
    divt_out_phy = []; divt_out_phy_std = []; 
    divt_out_one = []; divt_out_one_std = []; 
    divt_out_labels = [] # just for plotting

    divtype_nog = []; divtype_wgr = []
    divtype_phy = []; divtype_one = []
    for k in divtime.keys():

        # aggregate results, note: main value is "divt", old name, this can also be an L2 loss value
        result_type = {'unstable':0, 'regular': 0, 'recurring': 0, 'static':0}
        if len(divtime[k])>0:
            divt = divtime[k] # store for aggregation later 
            divt_std = 0. # deprecated, computed later on now, remove at some point

            for div_type in divtype[k]:
                    result_type[div_type] += 1
        else:
            # Neccesary to comment since it was causing the key error for untrained/unsaved model variants
            # while printing result_type plots
            divt = divt_std = 0.


        divtk = k.split(",")

        # create label and try to add if its new, old code relied on SUP & DP pairs... 
        label_prefix = '' 
        if have_multiple_archs:
            if runparams_all[k]['network_type'] == 'GCN':
                label_prefix = 'G,'
            elif runparams_all[k]['network_type'] == 'CNN':
                label_prefix = 'C,'

        new_label = f'{label_prefix}{runparams_all[k]["channels"]},{runparams_all[k]["depth"]}'

        if have_multiple_tasks:
            new_label = divtk[4][0:1]+','+new_label # pre-pend first letter only
        if len(different_bss.keys())>1: # we have different batch sizes?
            new_label = new_label+',b'+format(runparams_all[k]["batch_size"]) # append batch size, if it was varied
        
        if eval_params['tag_sort']:
            new_label = new_label+',t'+str(runparams_all[k]["tag"])
            
        if new_label not in divt_out_labels:
            divt_out_labels.append(new_label) 

        # split into different loss lists
        if divtk[3] == 'nog':
            divt_out_nog.append(divt)
            divt_out_nog_std.append(divt_std)
            divtype_nog.append(result_type) 
        elif divtk[3] == 'wgr':
            divt_out_wgr.append(divt)
            divt_out_wgr_std.append(divt_std)
            divtype_wgr.append(result_type) 
        elif divtk[3] == 'phy':
            divt_out_phy.append(divt)
            divt_out_phy_std.append(divt_std)
            divtype_phy.append(result_type) 
        elif divtk[3] == 'one':
            divt_out_one.append(divt)
            divt_out_one_std.append(divt_std)
            divtype_one.append(result_type) 
        else:
            log("unknown key? "+format(divtk) + " "+ format(divt) )
            exit(1)

    # sanity check lengths
    if len(divt_out_labels) != len(divt_out_nog) or \
        len(divt_out_labels) != len(divt_out_wgr) or \
        len(divt_out_labels) != len(divt_out_phy) or \
        len(divt_out_labels) != len(divt_out_one):
        log("Error - all variants have to have same count for sorting!..."); exit(1)

    # compute new index for ordering outputs along x
    # ugly, brute force ordering, assume numeric value for label!
    divt_order = []

    # sort by label, label2int() translates string into sort key
    if eval_params['sortgraph']>0:
        dtsmallest = label2int(divt_out_labels[0])
        for k in divt_out_labels:
            dtcurr = label2int(k)
            if dtcurr<dtsmallest:
                dtsmallest = dtcurr
        divt_order.append(dtsmallest)
        while len(divt_order)<len(divt_out_labels):
            dtsmallest = 1e30
            for k in divt_out_labels:
                dtcurr = label2int(k)
                #log("debug "+k+" "+ format([ dtsmallest, dtcurr,divt_order[-1] ]) ) # DEBUG
                if dtcurr>divt_order[-1] and dtcurr<dtsmallest:  
                    dtsmallest = dtcurr 
            divt_order.append(dtsmallest)
        # now we have the numbers, re-compute indices
        divt_order2 = divt_order; divt_order = []
        for k in divt_order2:
            for j in range(len(divt_out_labels)):
                if label2int(divt_out_labels[j])==k:
                    divt_order.append(j)
    else: # dont sort
        divt_order = range(len(divt_out_labels))

    # reordered field , suffix2
    divt_out_wgr2 = []; divt_out_nog2 = []; divt_out_labels2 = []; divt_out_wgr_std2 = []; divt_out_nog_std2 = [];
    divt_out_phy2 = []; divt_out_phy_std2 = []; divt_out_one2 = []; divt_out_one_std2 = [];
    divtype_out_wgr2 = []; divtype_out_nog2 = []; divtype_out_phy2 = []; divtype_out_one2 = []
    divt_out_labels3 = []

    for i in range(len(divt_order)):
        divt_out_labels2.append(divt_out_labels[divt_order[i]])

        divt_out_nog2.append(divt_out_nog[divt_order[i]])
        divt_out_nog_std2.append(divt_out_nog_std[divt_order[i]])
        divtype_out_nog2.append(divtype_nog[divt_order[i]])

        divt_out_wgr2.append(divt_out_wgr[divt_order[i]])  
        divt_out_wgr_std2.append(divt_out_wgr_std[divt_order[i]])
        divtype_out_wgr2.append(divtype_wgr[divt_order[i]])

        divt_out_phy2.append(divt_out_phy[divt_order[i]])  
        divt_out_phy_std2.append(divt_out_phy_std[divt_order[i]])
        divtype_out_phy2.append(divtype_phy[divt_order[i]])

        divt_out_one2.append(divt_out_one[divt_order[i]])  
        divt_out_one_std2.append(divt_out_one_std[divt_order[i]])
        divtype_out_one2.append(divtype_one[divt_order[i]]) 

    # optionally clean up, dont use old values from now on...
    #del divt_out_wgr,divt_out_nog,divt_out_labels
    #del divt_out_wgr_std,divt_out_nog_std


# ------------------------------------------------------------------------------------------------------------------------------

os.chdir(EVALDIR)

# ... else ...
if not eval_params['run_eval']: # ie only load!
    
    with open('plotdata.pickle', 'rb') as f: plotdata = pickle.load(f)
    [divt_out_one2,divt_out_one_std2] = plotdata['one']
    [divt_out_nog2,divt_out_nog_std2] = plotdata['nog'] 
    [divt_out_wgr2,divt_out_wgr_std2] = plotdata['wgr']  
    [divt_out_phy2,divt_out_phy_std2] = plotdata['phy']  
    divt_out_labels2 = plotdata['labels']

    [ different_archs, different_tasks, different_bss ] = plotdata['diffs']
    [ divtype_out_one2, divtype_out_nog2, divtype_out_wgr2, divtype_out_phy2 ] = plotdata['divtype'] 
    log("Plot data loaded from plotdata.pickle")

    with open('evaldata.pickle', 'rb') as f: evaldata = pickle.load(f)
    log("Eval data loaded from evaldata.pickle")

    if not os.path.exists('evalparams.pickle'):
        # older runs dont have it yet...
        if NUM_TESTS == 5 and  NUM_DS == 4:
            print("Error: make sure to specify num tests and data sets for older eval runs without evalparams.pickle")
            exit(1)
    else:
        with open('evalparams.pickle', 'rb') as f: eval_params_loaded = pickle.load(f)
        eval_params['metric'] = eval_params_loaded['metric']
        NUM_TESTS = eval_params['numtests'] = eval_params_loaded['numtests']
        NUM_DS    = eval_params['numds']    = eval_params_loaded['numds']
        log("Eval params loaded from evaldata.pickle using metric "+eval_params['metric']+" and NUM DS / TESTS: "+str([NUM_DS,NUM_TESTS]) )



# generate per model summaries, mean only for now 
# (potentially add agg_mode? might not be too useful though, mean is also used as basis for min/max)
if eval_params['per_model_summary']:
    for i in range(len(divt_out_one2)): # all same length!
        log("Per-model-means for "+divt_out_labels2[i])

        divt_out_oneT = divt_out_one2[i] # one results might be skipped, fill up with zeros
        if not type(divt_out_one2[i])==list:
            divt_out_oneT = np.zeros(np.asarray(divt_out_nog2[i]).shape)

        # from min/max , create single big array 
        # ignore for now: divt_out_phy2[i], "PHY-"+divt_out_labels2[i]
        a = np.reshape( np.asarray([divt_out_oneT,divt_out_nog2[i],divt_out_wgr2[i]]) , (3,-1,NUM_DS*NUM_TESTS) ) 
        am = np.mean(a,axis=-1)
        ast = np.std(a,axis=-1)
        for j in range( a.shape[1] ):
            log("    {:d}: ONE {:5.3f} , NOG {:5.3f} , WGR {:5.3f}".format(j, am[0,j], am[1,j], am[2,j])  )
            # also print std dev?




# find v in a and return the same index from b
def entry_for(v, a,b):
    for i in range(len(a)):
        if a[i]==v:
            return b[i]
    log("entry_for error: "+str(v)+" not found in "+str(a) +" , for "+str(b) )

agg_info = False

# aggregate results in a new pass
def agg(d,k):
    global agg_info
    if not type(d)==list:
        return 0.,0.

    # we want results for per randomseed & model, hence divide into sections for each model
    # then compute mean & std dev over tests (leaving per model results)
    a = np.reshape( np.asarray(d) , (-1,NUM_DS*NUM_TESTS) ) 
    am = np.mean(a,axis=1)
    ast = np.std(a,axis=1)

    if eval_params['agg_mode']==0 or eval_params['agg_mode']==3 or eval_params['agg_mode']==5: # mean , default=0, agg_mode 3 is relative to one, also uses mean , mode 5 is median
        if not agg_info:
            print("Aggregation mode: mean&std-dev for all models")
            agg_info = True

        # old:
        # divt     = np.mean( d )  # eval metric, mean, long until divergence is better
        # #divt     = np.median( divtime[k] )  # median, no better; more susceptible to outliers...
        # divt_std = np.std( d )     # eval metric, std dev

        divt     = np.mean( am )  # eval metric, mean, long until divergence is better
        #divt_std = np.std( ast )  # eval metric, std dev , wrong: std-dev of std-devs from above?
        #divt_std = np.mean( ast )  # mean of per model std-devs , non-standard...
        divt_std = np.std( am )  # more standard: std dev of means

        if eval_params['agg_mode']==5:
            divt = np.median( am )  

    elif eval_params['agg_mode']<3: # min max
    
        # note - error bar shows std dev for selected run

        # old: we want results for a single model (not just a single run) , hence divide into sections for each model then compute mean & std dev
        # a = np.reshape( np.asarray(d) , (-1,NUM_DS*NUM_TESTS) ) 
        # am = np.mean(a,axis=1)
        # ast = np.std(a,axis=1)
        # #log("Per model mean&std-dev for "+k+": "+str([am,ast])) # debug

        if eval_params['agg_mode']==1: # max
            if not agg_info:
                print("Aggregation mode: per model max")
                agg_info = True
            divt     = np.max( am ) 
            divt_std = entry_for(divt, am,ast) # np.std( 0 )  

        elif eval_params['agg_mode']==2: # min
            if not agg_info:
                print("Aggregation mode: per model min")
                agg_info = True
            divt     = np.min( am ) 
            divt_std = entry_for(divt, am,ast) # np.std( 0 )  
        else: # undefined?
            print("error NYI agg_mode "+str(eval_params['agg_mode']))
            exit(-1)

    elif eval_params['agg_mode']==4: # time graph
        divt = divt_std = -1. # skip here?

    else:
        log('Unknown agg mode {}'.format(eval_params['agg_mode'])); exit(1)

    #log(f'Aggregating {len(d)} results for model {k} = {divt:.5f} +/- {divt_std:.3f}')
    if USE_RELL2:
        relString = "rel. "
    else:
        relString = ""
    if not eval_params['agg_mode']==3:   
        log(f'    {len(d)} {relString}results for model {k} = {divt:.5f} +/- {divt_std:.3f}')
    return divt,divt_std

# small helper, also append...
def agg_append(a,b, d,k):
    _a,_b = agg(d,k)
    a.append(_a)
    b.append(_b)

# aggregated fields now have suffix 3
divt_out_one3=[];divt_out_one_std3=[]
divt_out_nog3=[];divt_out_nog_std3=[]
divt_out_wgr3=[];divt_out_wgr_std3=[]
divt_out_phy3=[];divt_out_phy_std3=[]
for i in range(len(divt_out_one2)): # all have to be same length!
    agg_append(divt_out_one3,divt_out_one_std3, divt_out_one2[i],"ONE-"+divt_out_labels2[i])
    agg_append(divt_out_nog3,divt_out_nog_std3, divt_out_nog2[i],"NOG-"+divt_out_labels2[i])
    agg_append(divt_out_wgr3,divt_out_wgr_std3, divt_out_wgr2[i],"WGR-"+divt_out_labels2[i])
    agg_append(divt_out_phy3,divt_out_phy_std3, divt_out_phy2[i],"PHY-"+divt_out_labels2[i])

    if eval_params['agg_mode']==3:
        one_val = divt_out_one3[-1] 
        divt_out_one3[-1] = 1.
        if one_val > 0.:
            divt_out_nog3[-1] /= one_val
            divt_out_wgr3[-1] /= one_val
            divt_out_one_std3[-1] /= one_val
            divt_out_nog_std3[-1] /= one_val
            divt_out_wgr_std3[-1] /= one_val
            log(f'    relative results for model {divt_out_labels2[i]}: \t\tNOG {divt_out_nog3[-1]:.2f} , WGR {divt_out_wgr3[-1]:.2f} ')


if eval_params['agg_mode']==4: # time graphs
    losses = evaldata['losses']
    gl = {}
    for k in losses.keys():
        #losses[k] = np.asarray(losses[k]) # shape ? (M, 8, 20, 48) -> M models, 4x2 tests, 20 steps, 48 space
        gl[k] = np.abs( np.asarray(losses[k]) )
        gl[k] = np.mean( gl[k] , axis=3) # avg space
        gl[k] = np.mean( gl[k] , axis=0) # avg tests
        glk_avg = np.mean( gl[k] , axis=0) # avg for seeds / models
        glk_std = np.std(  gl[k] , axis=0) # std dev for seeds / models
        gl[k]   = [glk_avg,glk_std] # [avg,stddev]

    fig,ax = plt.subplots()
    filt = eval_params['timeerr_filter_name']
    pltcnt = 0
    for k in losses.keys():
        if "_l2" in k and filt in k:
            if not filt == "": 
                log("plotting "+k+" ")
            ax.plot( np.arange(0.0, len(gl[k][0]), 1.) , gl[k][0], label=k)
            ax.fill_between( np.arange(0.0, len(gl[k][0]), 1.) , gl[k][0]-gl[k][1],  gl[k][0]+gl[k][1] , alpha=0.2)
            pltcnt+=1
        else: 
            if not filt == "": 
                log("[ skipping "+k+" , filt '"+filt+"' ]")

    if pltcnt==0:
        log("Error , no data found for filter "+eval_params['timeerr_filter_name']); exit(1)

    plt.title("Error over time")
    plt.xlabel('Time')
    plt.ylabel('L1') # we're computing L1 here!
    #plt.ylim(bottom=0.,top=1.8) # hard coded for now , todo make parameter...
    bottom=0.; top=1.8
    if eval_params['set_ylim_min']>-1.: 
        bottom=eval_params['set_ylim_min']
    if eval_params['set_ylim_max']>-1.: 
        top=eval_params['set_ylim_max']
    plt.ylim(bottom=bottom,top=top)
    #plt.xticks(index + bar_width, divt_out_labels2)
    plt.legend()
    plt.tight_layout()
    plotname = "l1.pdf" 
    plt.savefig(prefix+plotname, format='pdf'); log("Wrote "+prefix+plotname)

    exit(1)

# generate and save plots

# todo, later on use for 2D plots
def plotToAxis(axis, data):
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)
    im = axis.imshow(data, norm = LogNorm(vmin=vmin,vmax=vmax))     
    
    for (j,i),label in np.ndenumerate(data):
        txtLabels.append(axis.text(i,j,'%e' % label,ha='center',va='center', color = 'white', fontsize=4))
        
    ax1_divider = make_axes_locatable(axis)
    cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
    cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
    return im

#PLOT_THRESH = 1. # ignore values above , probably was an explosion...
def find_min_max(aIn,vmin,vmax):
    a = aIn.copy()
    #a[a > PLOT_THRESH] = np.median(a) # remove outliers
    vmin = np.min( np.append(a,vmin) )
    vmax = np.max( np.append(a,vmax) )
    return vmin,vmax



# ------------------------------------------------------------------------------------------------------------------------------

# generate graphs

fig,ax = plt.subplots()
index = np.arange(len(divt_out_wgr2)) # should all be same len
bar_width = 0.2
opacity = 1.
error_settings = {'ecolor': '0.3'}

rectsONE = plt.bar(index + 0*bar_width, divt_out_one3, bar_width, alpha=opacity, color='rosybrown',
                 yerr=divt_out_one_std3, error_kw=error_settings,
                 label='one step')

rectsNOG = plt.bar(index + 1*bar_width, divt_out_nog3, bar_width, alpha=opacity, color='steelblue',
                 yerr=divt_out_nog_std3, error_kw=error_settings,
                 label='no-grad')

rectsWGR = plt.bar(index + 2*bar_width, divt_out_wgr3, bar_width, alpha=opacity, color='darkorange',
                 yerr=divt_out_wgr_std3, error_kw=error_settings,
                 label='w grad')

# rectsPHY = plt.bar(index + 3*bar_width, divt_out_phy3, bar_width, alpha=opacity, color='lightgray',
#                  yerr=divt_out_phy_std3, error_kw=error_settings,
#                  label='full phys')

# compute only for shown values
if eval_params['compute_minmax']>0:
    vmin = 1e30; vmax = -1e30
    vmin,vmax = find_min_max(divt_out_one3,vmin,vmax)
    vmin,vmax = find_min_max(divt_out_nog3,vmin,vmax)
    vmin,vmax = find_min_max(divt_out_wgr3,vmin,vmax)
    if eval_params['load_phy_models']!=0:
        vmin,vmax = find_min_max(divt_out_phy3,vmin,vmax)

#plt.figure(figsize=(4*len(divt_out_labels2),16)) # scale size with entries
#plt.figure(figsize=(6,4)) # scale size with entries

title = 'Comparison'
title += ", tasks: "+format(list(different_tasks.keys()))
title += ", architectures: "+format(list(different_archs.keys()))
if eval_params['agg_mode']==1:
    title += ", maximum across models"
if eval_params['agg_mode']==2:
    title += ", minimum across models"
if eval_params['agg_mode']==3:
    title += ", relative to ONE"

#plt.rc('xtick', labelsize=6) # smaller, default 10
plt.rc('font', size=6)
plt.xticks(fontsize=4)

if eval_params['metric']=="l2" or  eval_params['metric']=="lphys":
    if eval_params['metric']=="l2": 
        plt.ylabel('L2')
        plotname = "l2-lin.pdf"
    elif eval_params['metric']=="lphys":
        plt.ylabel('LP')
        plotname = "lp-lin.pdf"

    if eval_params['compute_minmax']>0:
        plt.ylim(bottom=0, top=vmax*1.1)
    else:
        plt.ylim(bottom=0)

    if eval_params['agg_mode']==3:
        plt.ylabel('Factor (relative to ONE)')

elif eval_params['metric']=="correlation":
    plt.ylabel('Time until de-correlation')
    plt.ylim(0, HORIZON*2/3) # usually lower
    if eval_params['agg_mode']==3:
        plt.ylim(bottom=0, top=vmax*1.1) # around 1
        plt.ylabel('Time until de-correlation (relative to ONE)')
    plotname = "correlation.pdf"

else: # divergence
    plt.ylabel('Time until divergence')
    plt.ylim(top=HORIZON*1.1)
    if eval_params['agg_mode']==3:
        plt.ylim(bottom=0, top=vmax*1.1) # around 1
        plt.ylabel('Time until divergence (relative to ONE)')
    plotname = "divergence.pdf"

# manual override min max?
if eval_params['set_ylim_min']>-1.: 
    plt.ylim(bottom=eval_params['set_ylim_min'])
if eval_params['set_ylim_max']>-1.: 
    plt.ylim(top=eval_params['set_ylim_max'])


plt.title(title)
plt.xlabel('Model')
plt.xticks(index + bar_width, divt_out_labels2)
plt.legend()
plt.tight_layout()
plt.savefig(prefix+plotname, format='pdf'); log("Wrote "+prefix+plotname)


# also replot with log scale for L2 loss
if eval_params['metric']=="l2":
    plotname = "l2-log.pdf"
    plt.yscale('log')
    if eval_params['compute_minmax']>0:
        plt.ylim(bottom=vmin*0.75, top=vmax*1.2) # warning, vmin doesnt work if one variant is missing, eg ONE
    # keep previous , plt.ylabel('L2')
    plt.tight_layout() # otherwise it can crop...
    plt.draw()
    plt.savefig(prefix+plotname, format='pdf'); log("Wrote "+prefix+plotname)

plt.close('all')


#-------------------------------------------------------------------------------------
if eval_params['metric']=="divtime" and eval_params['run_eval']:
    fig,ax = plt.subplots()
    index = np.arange(len(divt_out_wgr2)) # should all be same len
    bar_width = 0.2
    opacity = 1.
    error_settings = {'ecolor': '0.3'}

    ## ONE 
    if len(divtype_out_one2)==1 and divtype_out_one2[0]['regular']==0:
        log(f"Skipping ONE")
    else:
        static_one = np.array([divtype_out_one2[i]['static'] for i in range(len(divtype_out_one2))]) 
        unstable_one = np.array([divtype_out_one2[i]['unstable'] for i in range(len(divtype_out_one2))])
        recurring_one = np.array([divtype_out_one2[i]['recurring'] for i in range(len(divtype_out_one2))])
        regular_one = np.array([divtype_out_one2[i]['regular'] for i in range(len(divtype_out_one2))])

        plt.bar(index + 0*bar_width, static_one, bar_width, alpha=opacity, color='rosybrown',
                        error_kw=error_settings, label='static')

        plt.bar(index + 0*bar_width, unstable_one, bar_width, alpha=opacity, color='steelblue',
                        error_kw=error_settings, label='unstable', bottom=static_one)

        plt.bar(index + 0*bar_width, recurring_one, bar_width, alpha=opacity, color='darkorange',
                        error_kw=error_settings, label='recurring', bottom=unstable_one+static_one)

        plt.bar(index + 0*bar_width, regular_one, bar_width, alpha=opacity, color='lightgray',
                        error_kw=error_settings, label='regular', bottom=recurring_one+unstable_one+static_one)

        log(f"Static ONE: {static_one}")
        log(f"Unstable ONE: {unstable_one}")
        log(f"Recurring ONE: {recurring_one}")
        log(f"Regular ONE: {regular_one}")

    ## NOG
    if len(divtype_out_nog2)==1 and divtype_out_nog2[0]['regular']==0:
        log(f"Skipping NOG")
    else:
        static_nog = np.array([divtype_out_nog2[i]['static'] for i in range(len(divtype_out_nog2))]) 
        unstable_nog = np.array([divtype_out_nog2[i]['unstable'] for i in range(len(divtype_out_nog2))]) 
        recurring_nog = np.array([divtype_out_nog2[i]['recurring'] for i in range(len(divtype_out_nog2))]) 
        regular_nog = np.array([divtype_out_nog2[i]['regular'] for i in range(len(divtype_out_nog2))])

        plt.bar(index + 1*bar_width, static_nog, bar_width, alpha=opacity, color='rosybrown',
                        error_kw=error_settings)

        plt.bar(index + 1*bar_width, unstable_nog, bar_width, alpha=opacity, color='steelblue',
                        error_kw=error_settings, bottom=static_nog)

        plt.bar(index + 1*bar_width, recurring_nog, bar_width, alpha=opacity, color='darkorange',
                        error_kw=error_settings, bottom=unstable_nog+static_nog)

        plt.bar(index + 1*bar_width, regular_nog, bar_width, alpha=opacity, color='lightgray',
                        error_kw=error_settings, bottom=recurring_nog+unstable_nog+static_nog)

        log(f"Static NOG: {static_nog}")
        log(f"Unstable NOG: {unstable_nog}")
        log(f"Recurring NOG: {recurring_nog}")
        log(f"Regular NOG: {regular_nog}")

    ## WGR
    if len(divtype_out_wgr2)==1 and divtype_out_wgr2[0]['regular']==0:
        log(f"Skipping WGR")
    else:
        static_wgr = np.array([divtype_out_wgr2[i]['static'] for i in range(len(divtype_out_wgr2))])
        unstable_wgr = np.array([divtype_out_wgr2[i]['unstable'] for i in range(len(divtype_out_wgr2))])
        recurring_wgr = np.array([divtype_out_wgr2[i]['recurring'] for i in range(len(divtype_out_wgr2))])
        regular_wgr = np.array([divtype_out_wgr2[i]['regular'] for i in range(len(divtype_out_wgr2))])

        plt.bar(index + 2*bar_width, static_wgr , bar_width, alpha=opacity, color='rosybrown',
                        error_kw=error_settings)

        plt.bar(index + 2*bar_width, unstable_wgr, bar_width, alpha=opacity, color='steelblue',
                        error_kw=error_settings, bottom=static_wgr)

        plt.bar(index + 2*bar_width, recurring_wgr, bar_width, alpha=opacity, color='darkorange',
                        error_kw=error_settings, bottom=unstable_wgr+static_wgr)

        plt.bar(index + 2*bar_width, regular_wgr, bar_width, alpha=opacity, color='lightgray',
                        error_kw=error_settings, bottom=recurring_wgr+unstable_wgr+static_wgr)

        log(f"Static WGR: {static_wgr}")
        log(f"Unstable WGR: {unstable_wgr}")
        log(f"Recurring WGR: {recurring_wgr}")
        log(f"Regular WGR: {regular_wgr}")

    ## PHY
    if len(divtype_out_phy2)==1 and divtype_out_phy2[0]['regular']==0 or eval_params['load_phy_models']==0:
        log(f"Skipping PHY")
    else:
        static_phy = np.array([divtype_out_phy2[i]['static'] for i in range(len(divtype_out_phy2))])
        unstable_phy = np.array([divtype_out_phy2[i]['unstable'] for i in range(len(divtype_out_phy2))])
        recurring_phy = np.array([divtype_out_phy2[i]['recurring'] for i in range(len(divtype_out_phy2))])
        regular_phy = np.array([divtype_out_phy2[i]['regular'] for i in range(len(divtype_out_phy2))])

        plt.bar(index + 3*bar_width, static_phy , bar_width, alpha=opacity, color='rosybrown',
                        error_kw=error_settings)

        plt.bar(index + 3*bar_width, unstable_phy, bar_width, alpha=opacity, color='steelblue',
                        error_kw=error_settings, bottom=static_phy)

        plt.bar(index + 3*bar_width, recurring_phy, bar_width, alpha=opacity, color='darkorange',
                        error_kw=error_settings, bottom=unstable_phy+static_phy)

        plt.bar(index + 3*bar_width, regular_phy, bar_width, alpha=opacity, color='lightgray',
                        error_kw=error_settings, bottom=recurring_phy+unstable_phy+static_phy)

        log(f"Static PHY: {static_phy}")
        log(f"Unstable PHY: {unstable_phy}")
        log(f"Recurring PHY: {recurring_phy}")
        log(f"Regular PHY: {regular_phy}")

    title = 'Result Type Stack Plot(ONE, NOG, WGR, PHY)'
    title += ", tasks: "+format(list(different_tasks.keys()))
    title += ", architectures: "+format(list(different_archs.keys()))

    #plt.rc('xtick', labelsize=6) # smaller, default 10
    plt.rc('font', size=4)
    plt.xticks(fontsize=4)
    plt.title(title)
    plt.ylabel('Number of results')
    plt.ylim(0., NUM_TESTS*15)
    plt.xlabel('Model')
    plt.xticks(index + 1.5*bar_width, divt_out_labels2)
    plt.legend()
    plt.tight_layout()
    plotname = "Result Type.pdf"
    plt.savefig(plotname, format='pdf'); log("Wrote "+plotname)
    plt.close('all')

#-----------------------------------------------------------------------------------------------------


# store only data for last plot to quickly recreate
if eval_params['run_eval']:
    plotdata = {}
    plotdata['one'] = [divt_out_one2,divt_out_one_std2]
    plotdata['nog'] = [divt_out_nog2,divt_out_nog_std2]
    plotdata['wgr'] = [divt_out_wgr2,divt_out_wgr_std2]
    plotdata['phy'] = [divt_out_phy2,divt_out_phy_std2]
    plotdata['labels'] = divt_out_labels2
    
    plotdata['diffs'] = [ different_archs, different_tasks, different_bss ]
    plotdata['divtype'] = [ divtype_out_one2, divtype_out_nog2, divtype_out_wgr2, divtype_out_phy2 ]
    
    with open('plotdata.pickle', 'wb') as f: pickle.dump(plotdata, f)
    log("Plot data saved in plotdata.pickle")


# collect and save all data
if eval_params['run_eval']:
    evaldata = {}
    evaldata['losses'] = losses
    evaldata['divtime'] = divtime 
    evaldata['divt_out_nog'] = [divt_out_nog , divt_out_nog_std] 
    evaldata['divt_out_wgr'] = [divt_out_wgr , divt_out_wgr_std]
    evaldata['divt_out_phy'] = [divt_out_phy , divt_out_phy_std]
    evaldata['divt_out_one'] = [divt_out_one , divt_out_one_std]
    evaldata['divt_out_labels'] = divt_out_labels

    with open('evaldata.pickle', 'wb') as f: pickle.dump(evaldata, f)
    log("Eval data saved in evaldata.pickle")

    with open('evalparams.pickle', 'wb') as f: pickle.dump(eval_params, f)
    log("Eval parameters saved in evalparams.pickle")


# collect and save all data
if eval_params['write_graph_data']>0:
    evaldata = {}
    #print(type(divt_out_one3)); print(len(divt_out_one3))
    evaldata['one-mean'] = divt_out_one3
    evaldata['one-std'] = divt_out_one_std3
    evaldata['nog-mean'] = divt_out_nog3
    evaldata['nog-std'] = divt_out_nog_std3
    evaldata['wgr-mean'] = divt_out_wgr3
    evaldata['wgr-std'] = divt_out_wgr_std3
    evaldata['labels'] = divt_out_labels2
    evaldata['run-info'] = "dir:"+ EVALDIR +", parameters: "+str(eval_params)+" "
    #print(evaldata['run-info'])
    #print(divt_out_labels2)
    fname = eval_params['rerun_eval_prefix']+'graphdata.pickle'
    with open(fname, 'wb') as f: pickle.dump(evaldata, f)
    log("Graph data saved in "+ fname )


# all done
logging.shutdown()

print("Done w eval in "+EVALDIR)
