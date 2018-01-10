import argparse
import tensorflow as tf
import h5py
import numpy as np
import sys
from multiprocessing import Process, Queue, Event
import math

import tensorflow_trainer as tft

class PfpDataLoader(tft.DataLoader):

    @staticmethod
    def add_args(parser):
        parser.add_argument('--file',metavar='F',required=True,help='File Containing Training/Testing Data')
        parser.add_argument('--eval_file',metavar='F',default=None,help='File Containing Testing Data')
        parser.add_argument('--batch_size',metavar='B',type=int,default=64,help='Batch Size')

    def __init__(self, args):
        self.args = args

        print('reading file '+args.file)
        f = h5py.File(args.file,'r')

        #generate data placeholders, input
        
        self.data_id_dct, self.data_in_dct, data_links, _, _ = tft.h5_create_input_dicts(f,['input'],self.args.batch_size,is_input=True)

        #generate label placeholders, input
        self.label_id_dct,self.label_in_dct, label_links, _, _ = tft.h5_create_input_dicts(f,['labels'],self.args.batch_size,is_input=False)

        self.batch = tft.Batcher(data_links, label_links, self.args.batch_size)
        file_size = np.size(self.data_in_dct[self.data_in_dct.keys()[0]],0)
        self.n_iter_n = int(math.floor(file_size/self.args.batch_size))
        #print("you have started processing batches of data!!!")

    def next_train_batch(self):
        #return self.batch.get_next_batch()
        data_dct = self.batch.get_next_batch()
        feed_dct = {}
        for key,value in data_dct.iteritems():
            if key in self.data_id_dct:
                feed_dct[self.data_id_dct[key]] = value
            else:
                feed_dct[self.label_id_dct[key]] = value
        return feed_dct

    def get_data_placeholders(self):
        return self.data_id_dct

    def get_label_placeholders(self):
        return self.label_id_dct

    def n_iter(self):
        return self.n_iter_n

    def can_eval(self):
        return False
    def shutdown(self):
        #print("you have exited2!!!!")
        self.batch.stop_threads()
