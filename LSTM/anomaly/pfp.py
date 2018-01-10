import tensorflow as tf
import argparse
import sys
import os
import time
import pickle
import numpy as np
import sklearn
from sklearn.metrics import roc_curve
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('..')

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import standard_ops

import tensorflow_trainer as tft

from pfp_network import PfpNetwork
from pfp_data_loader import PfpDataLoader

def listToPrint(l):
    s = ""
    for i,el in enumerate(l):
        if i != 0:
            s+='_'
        s+=str(el)
    return s

parser = argparse.ArgumentParser(conflict_handler='resolve')

#create optimizer
optimizer = tft.Optimizer([tft.MomentumWrapper,tft.RMSPropWrapper,tft.AdamWrapper])

#add arguments to pareser
optimizer.add_args(parser)
tft.Trainer.add_args(parser)
PfpNetwork.add_args(parser)
PfpDataLoader.add_args(parser)

parser.add_argument('--gpu_id',metavar='G',type=int,help='which gpu to use',required=True)
parser.add_argument('--run_name',metavar='R',type=str,help='name of run, used when generating checkpoint_dir',default=None)

#*****************************
#DEFAULTS

parser.set_defaults(log_output=True)
parser.set_defaults(print_every=10)
parser.set_defaults(eval_every=50)

parser.set_defaults(adam=True, epsilon=1e-8)
parser.set_defaults(learning_rate_decay_every=1, learning_rate_decay=0.8, learning_rate=0.0001)
parser.set_defaults(epochs=200)
parser.set_defaults(gradient_clip=0.01)
parser.set_defaults(checkpoint_every=1)

parser.set_defaults(batch_size=400)
#if not '--test' in sys.argv:
#    parser.set_defaults(dont_resume=True)
#*****************************

args = parser.parse_args()

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
    #args.run_name = 'pfp_hs_'+listToPrint(args.hidden_size)+'_fs_'+listToPrint(args.fc_size)
    args.run_name = args.file[slash_index:dot_index]

print(args.run_name)

#GPU
if args.gpu_id == -1:
    device = '/cpu:0'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = '/gpu:0'

pred = []
lbl = []
raw_pred = []
raw_label = []
def eval_write_func(res, log, epoch_per):
    pred.extend(res[0])
    lbl.extend(res[1])

    raw_pred.extend(np.transpose(res[2], (1,0,2)))
    raw_label.extend(np.transpose(res[3], (1,0,2)))

with tf.device(device):
    print(tft.OKGREEN + 'Creating DataLoader' + tft.ENDC)
    data_loader = PfpDataLoader(args)

    try:
        print(tft.OKGREEN + 'Creating Network' + tft.ENDC)
        network = PfpNetwork(args)

        error = tft.l1_error(network.label_pred,network.prediction)

        if args.test:
            ewf = eval_write_func
            ewo = [network.loss,network.label, network.prediction, network.label_pred]
        else:
            ewf = None
            ewo = None

        print(tft.OKGREEN + 'Initalizing Trainer' + tft.ENDC)
        trainer = tft.Trainer(args, args.run_name, data_loader, network, optimizer,
                              shared_output_metrics=[error], eval_output_metrics=[],
                              eval_write_func=ewf, eval_write_op=ewo)

        trainer.train()
        print(tft.OKGREEN + 'Finished' + tft.ENDC)
    except KeyboardInterrupt:
        pass
    except Exception:
        data_loader.terminate()
        raise sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2]

# terminate dataloader threads
data_loader.terminate()

if args.test:
    
    #ROC
    roc_path = '../Results/' + args.run_name + '/roc.png'
    err0_path= '../Results/' + args.run_name + '/error_0.png'
    err1_path= '../Results/' + args.run_name + '/error_1.png'
    pred = (pred-min(pred))/(max(pred)-min(pred))
    #print([l for p,l in zip(pred, lbl) if l == 1.0])
    mean_n = np.mean([p for p,l in zip(pred, lbl) if l == 1.0])
    mean_a = np.mean([p for p,l in zip(pred, lbl) if l != 1.0])

    print('Mean Data:'+str(mean_n))
    print('Mean Anomaly:'+str(mean_a))


    bad = [p for p,l in zip(pred, lbl) if l != 1.0]
    good = [p for p,l in zip(pred, lbl) if l == 1.0]

    print('bad '+str(len(bad)))
    print(bad[:100])
    print('good '+str(len(good)))
    print(good[:100])


    #tpr, fpr, thresholds = roc_curve(lbl, pred, pos_label=1)
    tpr, fpr, thresholds = roc_curve(lbl, pred)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    lw = 2
    ax.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve')
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    fig.savefig(roc_path)


    #predictions
    
    print(np.shape(raw_pred))
    
    print(next(i for i,l in enumerate(lbl) if l==1))
    print(next(i for i,l in enumerate(lbl) if l==0))
    

    fig = plt.figure()
    ax = fig.add_subplot(111)

    index = next(i for i,l in enumerate(lbl) if l==1)
    ax.plot(raw_pred[index])
    ax.plot(raw_label[index]) 

    fig.savefig(err1_path)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    index = next(i for i,l in enumerate(lbl) if l==0)
    ax.plot(raw_pred[index])
    ax.plot(raw_label[index]) 

    fig.savefig(err0_path)
