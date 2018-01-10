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
        parser.add_argument('--dropout_rate', metavar='DR', nargs='+', type=float, help='Dropout_Rate', default=0.0)

    def __init__(self, args, N_LABELS):
        
        self.args = args
        
        #data = tf.transpose(tft.get_inputs_placeholder_by_name('input'),[2,0,1])
        data = tf.transpose(tft.get_inputs_placeholder_by_name('input'),[1,0,2])
        labels = tf.to_int32(tf.squeeze(tft.get_targets_placeholder_by_name('labels')))

        time_steps = data.get_shape()[0].value
        batch_size = data.get_shape()[1].value
        seq_len = tf.ones((batch_size,),dtype=tf.int32)*time_steps
        keep_prob = 1.0 - args.dropout_rate[0] if not args.test else 1.0
        #keep_prob = 0.5
        dtype = data.dtype
        cur_inp = data        
       
        # RNN
        fwd_cells = []
        bwd_cells = []
        for hs in args.hidden_size:
            #fwd_cells.append(tf.contrib.rnn.LSTMCell(hs))
            #bwd_cells.append(tf.contrib.rnn.LSTMCell(hs))
            cellfwd = tf.contrib.rnn.LSTMCell(hs)
            cellbwd = tf.contrib.rnn.LSTMCell(hs)
            fwd_cells.append(tf.contrib.rnn.DropoutWrapper(cellfwd, output_keep_prob=keep_prob))
            bwd_cells.append(tf.contrib.rnn.DropoutWrapper(cellbwd, output_keep_prob=keep_prob))

        fwd_cell = tf.contrib.rnn.MultiRNNCell(fwd_cells)
        bwd_cell = tf.contrib.rnn.MultiRNNCell(bwd_cells)

        rnn_output, rnn_state = tf.nn.bidirectional_dynamic_rnn(fwd_cell, bwd_cell, data, dtype=data.dtype, time_major=True, sequence_length = seq_len, swap_memory=True, parallel_iterations=8)
       
        rnn_out = tf.concat([rnn_state[0][-1][-1], rnn_state[1][-1][-1]], axis=1)
        rnn_out.set_shape((batch_size, rnn_out.get_shape()[1].value))
        
        #FC
        cur_inp = rnn_out
        for i, fs in enumerate(args.fc_size):
            with tf.variable_scope("lin_"+str(i)):
                #cur_inp = tf.sigmoid(tft.linear(cur_inp, fs))
                cur_inp = tf.sigmoid(tft.linear(tf.nn.dropout(cur_inp,keep_prob), fs))
        final_out = tft.linear(cur_inp, N_LABELS)

        self.network = [final_out]
        self.labels = [labels]

        self.criterion = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=final_out, labels=labels))

    def get_criterion(self):
        return self.criterion

    def debug_nan_callback(self, sess, epoch_per, feed_dict):
        out, label = sess.run([self.network[0], self.labels[0]], feed_dict = feed_dict)

        print(np.shape(out))
        print(np.shape(label))
