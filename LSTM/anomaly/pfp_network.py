import argparse
import tensorflow as tf
import math
import numpy as np

from tensorflow.python.ops import rnn
import tensorflow_trainer as tft

class PfpNetwork(tft.Network):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--hidden_size',metavar='N',type=int,nargs='+',help='number of rnn neurons',required=True)
        parser.add_argument('--fc_size', metavar='N', type=int, nargs='+', help='number of fully connected neurons', required=True)
    def __init__(self, args):
        
        self.args = args
        
        data = tft.get_inputs_placeholder_by_name('input')
        dtype = data.dtype
        self.label=tft.get_inputs_placeholder_by_name('label')

        print(data)
                
        data = tf.transpose(data, [1,0,2])

        time_steps = data.get_shape()[0].value
        batch_size = data.get_shape()[1].value
        out_size = data.get_shape()[2].value
        seq_len = tf.ones((batch_size,),dtype=tf.int32)*time_steps
        
        cur_inp = data

        #RNN
        fwd_cells = []
        bwd_cells = []
        for hs in args.hidden_size:
            fwd_cells.append(tf.contrib.rnn.LSTMCell(hs))
            bwd_cells.append(tf.contrib.rnn.LSTMCell(hs))
        
        fwd_cell = tf.contrib.rnn.MultiRNNCell(fwd_cells)
        bwd_cell = tf.contrib.rnn.MultiRNNCell(bwd_cells)

        rnn_output, rnn_state = tf.nn.bidirectional_dynamic_rnn(fwd_cell, bwd_cell, cur_inp, dtype=data.dtype, time_major=True, sequence_length = seq_len, swap_memory=True, parallel_iterations=8)
    
        #FC
        cur_inp = tf.concat(rnn_output, axis=2)
        for i, fs in enumerate(args.fc_size):
            with tf.variable_scope("lin_"+str(i)):
                cur_inp = tf.map_fn(lambda x: tf.sigmoid(tft.linear(x, fs)), cur_inp)

        final_out = tf.map_fn(lambda x:tft.linear(x, out_size), cur_inp)

        self.network = [final_out]
        prediction = tf.slice(final_out, [0, 0, 0], [time_steps-1, batch_size, out_size])
        label = tf.slice(data, [1, 0, 0], [time_steps-1, batch_size, out_size])
        self.loss = tf.reduce_sum(tf.reduce_sum(tf.square(prediction-label), axis=2), axis=0)
        self.criterion = tf.reduce_mean(self.loss)
        #self.criterion = tf.nn.l2_loss(prediction-label)
        self.label_pred = label
        self.prediction = prediction

    def get_criterion(self):
        return self.criterion

    def debug_nan_callback(self, sess, epoch_per, feed_dict):
        out, label, crit, softmax = sess.run([self.network[0], self.labels[0], self.criterion, tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.network[0], labels=self.labels[0])], feed_dict = feed_dict)

        print(out)
        print(label)
        print(crit)
        print(softmax)
