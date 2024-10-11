import argparse
import sys
sys.path.append(".")
from kolm_correction import correct

parser = argparse.ArgumentParser(description='CmdLine Parse', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--nogradient', action='store_true')
parser.add_argument('--gradient',  dest='nogradient', action='store_false')
parser.set_defaults(nogradient=False)
parser.add_argument('--seeds', default=42, type=int, nargs='+', help='random seeds, multiple inputs like 53 64 ...')
parser.add_argument('--gpu', default=0, type=int, help='gpu id from nvidia-smi')
parser.add_argument('--feat_selector', default=0, type=int, help='feat_selector: 0,1,2,3')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--step_counts', default=1, type=int, nargs='+', help='unrolling curriculum, multiple inputs like 1 2 4 ...')
parser.add_argument('--start_learning_rate', default=1e-4, type=float, nargs='+', help='start learning rate, multiple inputs like : 1e-4 1e-5 1e-6 ...')
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--directory_name', default="", type=str)

params = vars(parser.parse_args())
correct(params)
