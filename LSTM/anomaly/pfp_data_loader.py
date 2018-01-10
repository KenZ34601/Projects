import argparse
import tensorflow as tf
import h5py
import numpy as np
import sys
from multiprocessing import Process, Queue
import math
import matplotlib.pyplot as plt
import sklearn
import os
import scipy.io as sio
from scipy.interpolate import interp1d

import tensorflow_trainer as tft

class PfpDataLoader(tft.DataLoader):

    class batcher():
        
        @staticmethod
        def process_batch(q, data, label, batch_size):
            data_points = np.shape(data)[0]

            while(True):
                for i in range(0,data_points-batch_size+1,batch_size):
                    q.put((np.array(data[i:i+batch_size]), np.array(label[i:i+batch_size])))

        def __init__(self, data, label, batch_size):
            self._data = data;
            self._label = label;
            self._batch_size = batch_size
            self.q = Queue(10)
            self.p = Process(target=self.process_batch, args=(self.q, data, label, batch_size))
            self.p.start()
            self.n_iter = np.shape(data)[0]/batch_size

        def __call__(self):
            return self.q.get()

        def shape(self):
            return (self._batch_size,) + np.shape(self._data)[1:]

        def terminate(self):
            self.p.terminate()
        

    @staticmethod
    def add_args(parser):
        parser.add_argument('--seed', metavar='S', default=1, type=int, help='np random seed')
        parser.add_argument('--file',metavar='F',required=True,help='File Containing Training/Testing Data')
        parser.add_argument('--eval_file',metavar='F',default=None,help='File Containing Testing Data')
        parser.add_argument('--batch_size',metavar='B',type=int,default=64,help='Batch Size')
        parser.add_argument('--data_labels',metavar='dl',type=int, nargs='+',default=[-1],help='which labels to cound as data, -1 means all that arnt abnormal')
        parser.add_argument('--anomaly_labels',metavar='R',type=str,required=True,help='which labels to count as anomalis')
        parser.add_argument('--min_iters',metavar='mi',type=int,default=100,help='min iters per epoch')
        parser.add_argument('--max_iters',metavar='mi',type=int,default=500,help='min iters per epoch')
        parser.add_argument('--raw_data',action='store_true',help='is in raw data format')
        parser.add_argument('--format',metavar='f',default='hdf5',help='format of data, one of hdf5, raw')
        
        #raw format args
        parser.add_argument('--raw_test_per',metavar='tts',type=float,default=0.25)
        parser.add_argument('--raw_stride', metavar='S', type=int, default=1000)
        parser.add_argument('--raw_window_size', metavar='WS',type=int, default=1000)
        parser.add_argument('--raw_subsample_fact', metavar='SF',type=float,default=1.0)

    def __init__(self, args):
        self.args = args

        np.random.seed(args.seed)

        if args.format == 'hdf5':

            print('reading file '+args.file)
            f = h5py.File(args.file,'r')

            print(f['input'])
            print(f['labels'])
            tmp = args.anomaly_labels
            a_lbl = [int(tmp[i]) for i in range(len(tmp))]
            if not args.test:
                data = []
                anomaly = []
                for inp, label in zip(f['input'],f['labels']):
                    if label in a_lbl:
                        anomaly.append(inp)
                    elif -1 in args.data_labels or label in args.data_labels :
                        data.append(inp)

                data = np.array(data)
                if args.raw_data:
                    data = np.reshape(data, (np.shape(data)[0]*np.shape(data)[1],np.shape(data)[2],1))
                data = np.array([data[i] for i in np.random.permutation(np.shape(data)[0])])
                print(np.shape(data))
                anomaly = np.array(anomaly)
                if args.raw_data:
                    anomaly = np.reshape(anomaly, (np.shape(anomaly)[0]*np.shape(anomaly)[1],np.shape(anomaly)[2],1))
                anomaly = np.array([anomaly[i] for i in np.random.permutation(np.shape(anomaly)[0])])
                print(np.shape(anomaly))

                data_labels = np.ones((np.shape(data)[0]))
                ano_labels = np.zeros((np.shape(anomaly)[0]))

                self.data = self.batcher(data, data_labels, self.args.batch_size)
                self.anomaly = self.batcher(anomaly, ano_labels, self.args.batch_size)                         

                print('n data points: '+str(self.data.n_iter))
                print('n anomaly points: '+str(self.anomaly.n_iter))
                
            else:
                data = f['input'].value
                label = np.reshape(f['labels'],(np.shape(data)[0],))

                if args.raw_data:
                    label = np.reshape(np.tile(label, (1,np.shape(data)[1])),(-1))
                    data = np.reshape(data, (np.shape(data)[0]*np.shape(data)[1],np.shape(data)[2],1))
                    print(np.shape(data))
                    print(np.shape(label))

                label = np.array([l not in a_lbl for l in label])

                self.data = self.batcher(data, label, self.args.batch_size)

        elif args.format=='raw':

            mat_dir = args.file+'/mats'
            if not os.path.isdir(mat_dir):
                print(tft.FAIL+'no mats directory in '+args.file+tft.ENDC)
            sets = os.listdir(mat_dir)
            train_filenames = []
            test_filenames = []
            for set in sets:
                set_dir = os.path.join(mat_dir,set)
                fn = [os.path.join(set_dir,f) for f in os.listdir(set_dir) if f.endswith('.mat')]
                fn = [fn[i] for i in np.random.permutation(len(fn))]
                test_num = int(len(fn) * args.raw_test_per)
                test_filenames.append(fn[:test_num])
                train_filenames.append(fn[test_num:])
    
            def _to_data(filenames, stride, window_size):
                data = []
                for filename in filenames:
                    mat = sio.loadmat(filename)
                    raw_data = np.squeeze(np.array(mat['Data']))
                    if args.raw_subsample_fact != 1.0:
                        x = np.linspace(0,len(raw_data),num=len(raw_data),endpoint=False)
                        f = interp1d(x,raw_data, kind='cubic')
                        xnew = np.linspace(0,len(raw_data),num=int(len(raw_data)*args.raw_subsample_fact), endpoint=False)
                        raw_data = f(xnew)
                    for i in range(0,len(raw_data)-window_size+1,stride):
                        tmp_data = np.array(raw_data[i:i+window_size])
                        if tmp_data.ndim == 1:
                            tmp_data = np.expand_dims(tmp_data, axis=1)
                        data.append(tmp_data)
                return data
              
            if args.test:
                test_data = [_to_data(filenames, args.raw_stride, args.raw_window_size) for filenames in test_filenames]
                test_label = [np.ones((np.shape(d)[0]))*int(i in args.anomaly_labels) for i,d in enumerate(test_data)]
                #concat
                test_data = [d for d_i in test_data for d in d_i]
                test_label = [d for d_i in test_label for d in d_i]
                #test_data = np.concatenate(test_data, axis=0)
                #test_label = np.concatenate(test_label, axis=0)
                #permute
                perm = np.random.permutation(np.shape(test_data)[0])
                test_data = [test_data[i] for i in perm]
                test_label = [test_label[i] for i in perm]

                print(np.shape(test_data))

                self.data = self.batcher(test_data, test_label, self.args.batch_size)
            else:
                data = []
                ano = []
                for i in range(len(train_filenames)):
                    tmp_data = _to_data(train_filenames[i], args.raw_stride, args.raw_window_size)
                    if i in args.anomaly_labels:
                        ano.append(tmp_data)
                    else:
                        data.append(tmp_data)

                #concat
                data = [d for d_i in data for d in d_i]
                ano = [d for d_i in ano for d in d_i]
                #data = np.concatenate(data, axis=0)
                #ano = np.concatenate(ano, axis=0)
                #perm
                data = [data[i] for i in np.random.permutation(np.shape(data)[0])]
                ano = [ano[i] for i in np.random.permutation(np.shape(ano)[0])]
                #labels
                data_labels = np.zeros((np.shape(data)[0]))
                ano_labels = np.ones((np.shape(ano)[0]))

                print(np.shape(data))
                print(np.shape(data_labels))
                print(np.shape(ano))
                print(np.shape(ano_labels))

                self.data = self.batcher(data, data_labels, self.args.batch_size)
                self.anomaly = self.batcher(ano, ano_labels, self.args.batch_size)                         

                print('n data points: '+str(self.data.n_iter))
                print('n anomaly points: '+str(self.anomaly.n_iter))
                
        else:
            print(tft.FAIL + 'unrecognized format '+args.format+tft.ENDC)
            exit()

        self.data_ph = tf.placeholder(tf.float32, shape=self.data.shape(),name='input')
        tf.add_to_collection(tf.GraphKeys.INPUTS, self.data_ph)
        self.label_ph = tf.placeholder(tf.float32, shape=(args.batch_size,),name='label')
        tf.add_to_collection(tf.GraphKeys.INPUTS, self.label_ph)

    #returns placeholders for the data and label that are loaded from this class
    def next_train_batch(self):
        data = self.data()
        return {self.data_ph:data[0], self.label_ph:data[1]}

    def next_eval_batch(self):
        data = self.anomaly()
        return {self.data_ph:data[0], self.label_ph:data[1]}

    def get_data_placeholders(self):
        return [self.data_ph]

    def get_label_placeholders(self):
        return [self.label_ph]

    def n_iter(self):
        if self.args.test:
            return min(self.data.n_iter,100)
        else:
            return min(max(self.data.n_iter,self.args.min_iters),self.args.max_iters)

    def can_eval(self):
        return not self.args.test

    def terminate(self):
        self.data.terminate()
        if not self.args.test:
            self.anomaly.terminate()
