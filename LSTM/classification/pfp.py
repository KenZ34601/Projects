import tensorflow as tf
import sys, os, argparse
import time, signal
import pickle
import numpy as np

sys.path.append('..')

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import standard_ops

import tensorflow_trainer as tft

from pfp_network import PfpNetwork
from pfp_data_loader import PfpDataLoader


#def signal_handler(signal, frame):
#    global interrupted
#    interrupted = True

def listToPrint(l):
    s = ""
    for i,el in enumerate(l):
        if i != 0:
            s+='_'
        s+=str(el)
    return s

parser = argparse.ArgumentParser(conflict_handler='resolve')

#create optimizer
optimizer = tft.Optimizer([tft.MomentumWrapper,tft.AdadeltaWrapper,tft.AdamWrapper])

#add arguments to parser
optimizer.add_args(parser)
tft.Trainer.add_args(parser)
PfpNetwork.add_args(parser)
PfpDataLoader.add_args(parser)

parser.add_argument('--gpu_id',metavar='G',type=int,help='which gpu to use',required=True)
parser.add_argument('--run_name',metavar='R',type=str,help='name of run, used when generating checkpoint_dir',default=None)
parser.add_argument('--output_size',type=int,help='number of classes',required=True)

##################################
#DEFAULTS

parser.set_defaults(log_output=True)
parser.set_defaults(print_every=10)
parser.set_defaults(adam=True, epsilon=1e-8)
parser.set_defaults(learning_rate_decay_every=1, learning_rate_decay=1.0, learning_rate=0.0001)
parser.set_defaults(epochs=100)
parser.set_defaults(gradient_clip=1)
parser.set_defaults(checkpoint_every=1)

parser.set_defaults(batch_size=400)
if not '--test' in sys.argv:
    parser.set_defaults(dont_resume=True)
#*****************************

args = parser.parse_args()
print(args.dropout_rate)
#specify run name
if args.run_name == None:
    try:
        slash_index = args.file.rindex('/')+1
    except ValueError:
        slash_index = 0
    try:
        dot_index = args.file.index('.', slash_index)
    except ValueError:
        dot_index = len(args.file)
    args.run_name = args.file[slash_index:dot_index]

print(args.run_name)

#GPU
if args.gpu_id == -1:
    device = '/cpu:0'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = '/gpu:0'

#Globals
N_LABELS=args.output_size

#signal.signal(signal.SIGINT, signal_handler)

with tf.device(device):
    print(tft.OKGREEN + 'Creating DataLoader' + tft.ENDC)
    data_loader = PfpDataLoader(args)

    try:
        print(tft.OKGREEN + 'Creating Network' + tft.ENDC)
        network = PfpNetwork(args,N_LABELS)

        accuracy_n = tft.softmax_accuracy(
            network.labels[0], network.network[0])

        print(tft.OKGREEN + 'Initalizing Trainer' + tft.ENDC)
        trainer = tft.Trainer(args, args.run_name, data_loader, network, optimizer,
                              shared_output_metrics=[accuracy_n], eval_output_metrics=[])

        trainer.train()
        print(tft.OKGREEN + 'Finished' + tft.ENDC)
    except KeyboardInterrupt:
        pass
    except Exception:
	data_loader.shutdown()
        raise sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2]

# terminate dataloader threads
data_loader.shutdown()

