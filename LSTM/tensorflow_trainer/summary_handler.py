import tensorflow as tf
import sys
import os
import time
import numpy as np

from tensorflow_trainer.utils.header import bcolors



class SummaryHandler:

    @staticmethod
    def add_args(parser):
        parser.add_argument('--print_every', metavar='N', type=int, help='print loss every N', default=100)
        parser.add_argument('--dont_store_summaries', action='store_true', help='dont record summerys')
        parser.add_argument('--record_summary_every', metavar='N', type=int, default=None,
                            help='record summary information every N steps, defaults to 10 * --print_every')
        parser.add_argument('--log_output', action='store_true', help='write output to log file')

    def __init__(self, args, criterion, shared_ops, train_ops, eval_ops, summary_only_ops, checkpoint_dir):

        self.args = args
        self.record_sum = False

        if args.record_summary_every == None:
            args.record_summary_every = args.print_every

        #*******************
        # printing
        #*******************

        # list of ops we both print and take summarys for
        self.train_print_ops = []
        self.eval_print_ops = []

        for op in shared_ops:
            if len(op.get_shape()) == 0:
                # its a scalor so we can print it
                self.train_print_ops.append(op)
                self.eval_print_ops.append(op)

        for op in train_ops:
            if len(op.get_shape()) == 0:
                # its a scalor so we can print it
                self.train_print_ops.append(op)

        for op in eval_ops:
            if len(op.get_shape()) == 0:
                # its a scalor so we can print it
                self.eval_print_ops.append(op)

        # self.train_avg_store
        self.train_avg_store = np.zeros(len(self.train_print_ops)+1)
        self.train_avg_store_tot = np.zeros(len(self.train_print_ops)+1)

        #*******************
        # Summary
        #*******************

        if (not args.dont_store_summaries):

            if isinstance(criterion, list):
                self._add_summary_for_ops(criterion, False)
            else:
                self._add_summary_for_ops([criterion], False)
            self._add_summary_for_ops(shared_ops, False)
            self._add_summary_for_ops(train_ops, False)
            self._add_summary_for_ops(eval_ops, True)
            self._add_summary_for_ops(summary_only_ops,False)

            self.merge_summaries_op = tf.summary.merge_all()
        else:
            self.merge_summaries_op = None

        # summary writers
        summaries_dir = os.path.join(checkpoint_dir,'summaries')
        if not os.path.isdir(summaries_dir):
            os.mkdir(summaries_dir)

        def rmdir(d):
            for root, dirs, files in os.walk(d, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(d)

        train_writer_path = os.path.join(summaries_dir,('test' if args.test else 'train'))
        if os.path.isdir(train_writer_path):
            rmdir(train_writer_path)
        self.train_writer = tf.summary.FileWriter(train_writer_path)
        eval_writer_path = os.path.join(summaries_dir,('test' if args.test else 'train')+'_eval')
        if os.path.isdir(eval_writer_path):
            rmdir(eval_writer_path)
        self.eval_writer = tf.summary.FileWriter(eval_writer_path)

        # concatenate train op list
        if self.merge_summaries_op == None:
            self.train_sum_op = self.train_print_ops
            self.eval_ops = self.eval_print_ops
        else:
            self.train_sum_op = self.train_print_ops+[self.merge_summaries_op]
            self.eval_ops = self.eval_print_ops+[self.merge_summaries_op]

        # *******************
        # writers
        # *******************
        self.writers = []
        self.writers.append(sys.stdout)
        # output loging file
        if args.test:
            f_name = 'output_test.log'
        else:
            f_name = 'output_train.log'
        self.output_log = open(os.path.join(args.checkpoint_dir, f_name), 'w')
        if args.log_output:
            self.writers.append(self.output_log)

        #Write args
        for writer in self.writers:
            writer.write(str(args))
            writer.write('\n')
            writer.flush()

    def _add_summary_for_ops(self, ops,eval=False):
        for op in ops:
            op_name = ('eval_' if eval else '') + op.name
            if len(op.get_shape()) == 0:
                # scalar
                tf.summary.scalar(op_name,op)
            elif len(op.get_shape()) == 1:
                # not scalar, little trickier
                # vector give it a histogram
                tf.summary.histogram(op_name,op)
            elif len(op.get_shape()) == 2:
                # image
                dims = tf.shape(op)
                x_min = tf.reduce_min(op)
                x_max = tf.reduce_max(op)
                op_0_to_1 = (op - x_min) / (x_max - x_min)
                op_image = tf.reshape(op_0_to_1,[1,dims[0],dims[1],1])
                op_image = tf.image.convert_image_dtype (op_image, dtype=tf.uint8)
                tf.image_summary(op_name,op_image)
            else:
                print(bcolors.FAIL + 'ops with shape > 2 not currently supported' + bcolors.endc)
                os.exit(1)

    def add_graph(self,sess):
        if self.merge_summaries_op != None:
            self.train_writer.add_graph(sess.graph)
            self.eval_writer.add_graph(sess.graph)

    def get_train_ops(self,ite,max_ite):
        if self.merge_summaries_op != None\
                and (ite % self.args.record_summary_every == self.args.record_summary_every-1 or ite == max_ite-1):
            self.record_sum = True
            return self.train_sum_op
        else:
            self.record_sum = False
            return self.train_print_ops

    def get_eval_ops(self):
        return self.eval_ops

    def train_iter(self,res,iter,epoch_per,start_time):
        self.train_avg_store = np.add(self.train_avg_store, res[:len(self.train_print_ops)+1])

        if iter % self.args.print_every == self.args.print_every-1:
            # add averages into total store
            self.train_avg_store_tot = np.add(self.train_avg_store_tot,self.train_avg_store)

            # divide average by prints
            self.train_avg_store = np.divide(self.train_avg_store,self.args.print_every)
            self._print_status(self.train_avg_store,epoch_per,start_time=start_time)

            # clear out averages
            self.train_avg_store[:] = 0

        if self.record_sum:
            # record summary
            self.train_writer.add_summary(res[-1])

    def eval(self,res,epoch):
        # print results
        self._print_status(res[:len(self.eval_print_ops)+1],epoch,eval=True,PreCursor='EVAL: ')
        if self.merge_summaries_op != None:
            self.eval_writer.add_summary(res[-1])


    def print_epoch(self,epoch,epoch_iter,epoch_start_time):
        # end of epoch metrics
        self.train_avg_store_tot = np.add(self.train_avg_store_tot,self.train_avg_store)
        self.train_avg_store_tot = np.divide(self.train_avg_store_tot,epoch_iter)
        epoch_loss = self.train_avg_store_tot[0]
        self._print_status(self.train_avg_store_tot,epoch, PreCursor='EPOCH:', start_time=epoch_start_time, end_epoch=True)

        metrics = np.copy(self.train_avg_store_tot)

        # reset average store to 0
        self.train_avg_store_tot[:] = 0
        self.train_avg_store[:] = 0

        return metrics

    def _print_status(self, results, epoch_per, eval = False, start_time=None,PreCursor=None,end_epoch=False):
        for writer in self.writers:
            self._print_status_single(writer,results,epoch_per,eval,start_time,PreCursor,end_epoch)

    def _print_status_single(self, writer, results, epoch_per, eval = False, start_time=None,PreCursor=None, end_epoch=False):
        print_color = (bcolors.OKBLUE if not end_epoch else bcolors.OKGREEN)
        if PreCursor != None:
            writer.write(print_color + PreCursor + ' ' + bcolors.ENDC)
        if start_time != None:                                                            
            writer.write(print_color + 'Time: '+str(time.asctime()) + ' ' +bcolors.ENDC)
            writer.write('(%0.3f)' % (time.time()-start_time))
        writer.write(' Epoch %0.2f: Loss = %0.5f ' % (epoch_per,results[0]))
        print_ops = self.eval_print_ops if eval else self.train_print_ops
        for k,op in enumerate(print_ops):
            writer.write(' ' + op.name + '= %0.4f' % (results[k+1]))
            
        writer.write('\n')
        writer.flush()

    def eval_write_to_log(self,ret_val):
        self.output_log.write('Eval: ')
        if isinstance(ret_val,list):
            for ret_val_i in ret_val:
                self.output_log.write(np.array_str(ret_val_i))
        else:
            self.output_log.write(np.array_str(ret_val))


    def epoch_end(self):
         if self.merge_summaries_op != None:
             self.train_writer.flush()
             self.eval_writer.flush()

    def finish(self):
        if self.merge_summaries_op != None:
            self.train_writer.close()
            self.eval_writer.close()
        #for writer in self.writers:
        #    writer.close()
