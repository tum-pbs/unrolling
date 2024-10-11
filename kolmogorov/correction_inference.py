from matplotlib import animation
import sys
sys.path.append('../PhiFlow')
sys.path.append('..')
from diffpiso import *
from diffpiso.evaluation_helpers import EK_spectrum_2D
from kolmogorov_flow.networks import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from kolmogorov_flow.evaluation_functions import *
import socket
from helpers import *

base_path = '../kolm_tests'


models = ["enter paths to model ckpts here"]  

resnet_features, feature_selector = get_features(models)

frame_count = 301
initial_frames = [np.load('reference_data_1000.npy')[0], np.load('reference_data_600.npy')[0]]
Res = [1000, 600]

for Re, initial_frame in zip(Res, initial_frames):
  dataset_timestep = 0.005
  momForce = True
  dataset_wavenumber = 6
  resolution = [32,32]
  frame_increment = 4
  storing_interval = 1

  storing_frames = range(0, frame_count, storing_interval)

  domain, sim_physics, _, _, _ = periodic_phiflow_domain(resolution, 1, set_viscosity=True, Re=Re)
  
  for iter, (model, feature) in enumerate(zip(models, feature_selector)):
      print('Processing model Nbr: ', iter)
      trajectory = run_testsim(model, lambda: convResNet_2D_centered(resnet_features[feature], 2, conv_skips=True), domain, dataset_wavenumber,
                              frame_count, initial_frames, sim_physics, frame_increment * dataset_timestep, storing_frames, momForce)
      feature_count = int(model.split('nFeat')[1].split('_')[0])
      np.savez("/home/blist/datadrive/visRe"+str(Re)+"/"+str(feature_count)+"_"+model.split('.ckpt')[0].split('rand')[1].split('_')[0]+
                "_inference_traj_Re"+str(Re), trajectory) 
