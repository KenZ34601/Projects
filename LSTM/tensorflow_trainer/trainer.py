import tensorflow as tf
import os
import time
import argparse
import sys
import numpy as np
from tensorflow_trainer.utils import graph_common
from tensorflow_trainer.utils import data_utils
from tensorflow_trainer.summary_handler import SummaryHandler
from tensorflow_trainer.utils.header import bcolors

class Trainer:

    @staticmethod
    def add_args(parser):

        # run params
        parser.add_argument('--epochs', metavar='N', type=int, help='number of epochs to run for', default=50)
        parser.add_argument('--eval_every', metavar='N', type=int, help='run eval metrics ever N', default=500)

        parser.add_argument('--grad_by_scope', action='store_true',
                            help='if true, uses the scope of each criterion '
                                 'to determine which variables to update during backprop')
        #parser.add_argument('--threads', metavar='N', type=int, help='number of threads for tensorflow to use')

        # learning rate args included by default
        parser.add_argument('--learning_rate', metavar='LR', nargs='+', type=float,
                            help='Initial Learning Rate', default=0.01)
        parser.add_argument('--learning_rate_decay', metavar='LRD', type=float,
                            help='Factor to decay learning rate by', default=0.8)
        parser.add_argument('--learning_rate_decay_every', metavar='LRDE', type=float,
                            help='decay learning rate every this number of epochs', default=1.0)
        parser.add_argument('--staircase', action='store_true', help='uses staircase method of learning rate decay')

        # checkpoints
        parser.add_argument('--dont_resume', action='store_true',
                            help='dont read checkpoint from starting point to resume training')
        parser.add_argument('--resume', action='store_true',
                            help='read checkpoint from starting point to resume training, overrides dont resume')
        parser.add_argument('--start_over', action='store_true',
                            help='if true will read checkpoint from file but will start at epoch 0')

        parser.add_argument('--checkpoint_every', metavar='N', type=int,
                            help='checkpoint every given number of epochs', default=1)
        parser.add_argument('--checkpoint_dir', metavar='dir',
                            help='directory to store checkpoints to', default='./ckpts')
        parser.add_argument('--init_from', nargs='+',
                            help='initalize variables from checkpoint', default=[])

        parser.add_argument('--restore_only_trainable', action='store_true',
                            help='restore only trainable variables which existed previously in checkpoint')
        parser.add_argument('--restore_scope',type=str,default=None,nargs='+',
                            help='restore only variables with this scope as prefix')
        parser.add_argument('--store_scope',type=str,default=None,
                            help='save only variable with this scope as prefix')

        parser.add_argument('--test', action='store_true', help='run testing')
        parser.add_argument('--test_iters', metavar='I', type=int, default=None,
                            help='only run this number of iters for testing')

        SummaryHandler.add_args(parser)

    def __init__(self, args, run_name, data_loader, network, optimizer,
                 eval_write_op=None, eval_write_func=None,
                 shared_output_metrics=[], train_output_metrics=[], eval_output_metrics=[], summary_only_metrics=[]):
        """

        :param args: aggregated parser including those defined by get_arg_parser above
        :param run_name: string defining name of run, used to create checkpoint dir
        :param data_loader: instance of abstract_classes.data_loader.DataLoader
        :param network: instance of abstract_classes.network.Network
        :param optimizer: instance of optimizer.optimizer.Optimizer
        :param eval_write_op: tensorflow op to run duing eval and write result to log file
        :param eval_write_func: function to be called during eval, will be passed the result of eval_write_op and
        the handle to the log file
        :param shared_output_metrics: list of tf graph nodes to output when training and testing
        :param train_output_metrics: list of tf graph nodes to output when training
        :param eval_output_metrics: list of tf graph nodes to output when testing
        """

        if args.resume:
            args.dont_resume=False

        self.data_loader = data_loader
        self.network = network
        self.args = args

        self.eval_write_op = eval_write_op
        self.eval_write_func = eval_write_func

        # check for wrong parameters
        if self.args.test and (self.args.dont_resume or self.args.start_over):
            raise ValueError(bcolors.FAIL + '--test set but no checkpoint provided, please provice one with --init_from and remove dont_resume or start_over'
                  + bcolors.ENDC)

        # create directories
        if not os.path.isdir(args.checkpoint_dir):
            os.mkdir(args.checkpoint_dir)
        args.checkpoint_dir = self.checkpoint_dir = os.path.join(args.checkpoint_dir, run_name)
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        #make optimizers for criterions supplied by network
        if not args.test:
            self.train_ops, self.train_map, self.global_steps, self.learning_rate_op = self.make_learning_ops(network, optimizer)
        self.loss_op = self.make_loss_op(network)

        # create SummaryHandler
        # summary handler handles printing results to the screen and writing output files
        # as well as storing tensorboard summaries
        self.sum_han = SummaryHandler(args, self.network.get_criterion(), shared_output_metrics, train_output_metrics,
                                      eval_output_metrics, summary_only_metrics, self.checkpoint_dir)
        # epoch variable
        self.epoch = tf.Variable(0, name='epoch', trainable=False)

        # Add the variable initializer Op.
        try:
            self.init = tf.global_variables_initializer()
        except:
            self.init = tf.initialize_all_variables()

        #saving and restoring, many options are available, see the input arguments for more details
            
        #self.saver is used to save checkpoints
        self.saver = self.get_saver(network)
        #self.restorere is used to restore checkpoints
        self.restr = self.restorer(args, self.checkpoint_dir)

        # store update variables
        # if network.has_update() it will specify a set of placeholders given by update_ph
        # a set of default zero ops given by update_zero_ops and a set up update ops
        # given by update_ops. One the first iteration the value of each update_ph is fed with
        # the respective value from update_zero_ops. The value of each update_op is also returned
        # on each subsequent iteration the valuse of each ph is fed with the previous value from
        # its respective update_op
        if network.has_update():
            self.update_ph, self.update_zero_ops, self.update_ops = network.get_update()
            self.n_update_ops = len(self.update_ops)
        else:
            self.update_ops = []
            self.n_update_ops = 0

    def train(self):

        """
        runs training

        :return: None
        """

        if self.args.test:
            print(bcolors.OKGREEN + "Running Test" + bcolors.ENDC)
        else:
            print(bcolors.OKGREEN + "Running Training" + bcolors.ENDC)

        config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        #config.gpu_options.per_process_gpu_memory_fraction = 1.0
        config.allow_soft_placement = True
        # config.log_device_placement = True

        with tf.Session(config=config) as sess:
            self.sess = sess
            
            #randomly initialize all variables
            sess.run(self.init)

            #restore from checkpoint if necessary
            self.restr.restore(sess, self.epoch)
            
            # start over means we want to start from epoch 0, 
            # have to reset epoch counter and global steps
            if self.args.start_over:
                print(bcolors.OKGREEN + 'Setting epoch to 0' + bcolors.ENDC)
                sess.run(self.epoch.assign(0))
                for global_step in self.global_steps:
                    sess.run(global_step.assign(0))

            
            if self.args.test:
                starting_epoch = 0
            else:
                starting_epoch = sess.run(self.epoch)

            # set initial test/training
            self.network.set_test(self.args.test, sess)
            
            # run update zeros to get them into numpy arrays
            if self.network.has_update():
                update_np_zeros = sess.run(self.update_zero_ops)
                update_np = update_np_zeros

            #start data_loader
            self.data_loader.start(sess)

            # add graph to summary writers
            self.sum_han.add_graph(sess)

            # loop over all epochs
            for epoch in (range(1) if self.args.test else range(starting_epoch, self.args.epochs)):

                print(bcolors.BOLD + 'Starting Epoch: ' + str(epoch) + bcolors.ENDC)

                # store epoch in variable
                sess.run(self.epoch.assign(epoch))

                #call network callback to start epoch
                self.network.start_epoch(sess, epoch, self.sum_han.output_log)

                #number of iterations this epoch
                iters = self.data_loader.n_iter()
                if self.args.test and self.args.test_iters:
                    iters = self.args.test_iters

                epoch_start_time = time.time()

                #run iterations
                for ite in range(iters):
                    start_time = time.time()
                    epoch_per = epoch+(float(ite)/iters)

                    #call start callback
                    self.network.start_iter(sess, epoch_per, ite, iters)

                    # fill feed dict, do training
                    feed_dict = self.data_loader.next_train_batch()

                    # if we have to do an update then we need to add
                    # the update ph and there values into the feed_dict
                    if self.network.has_update():
                        feed_dict.update(dict(zip(self.update_ph, update_np)))

                    if (not self.args.test) and self.network.has_update_criterions():
                        self.train_ops = [self.train_map[c] for c in self.network.cur_criterions(epoch_per, ite)]

                    sum_train_ops = self.sum_han.get_train_ops(ite, iters)

                    #call debug callback
                    self.network.debug_callback(sess, epoch_per, feed_dict)

                    #run the network
                    if self.args.test:
                        ret_val = sess.run( [self.loss_op] + sum_train_ops + self.update_ops, feed_dict=feed_dict)
                    else:
                        ret_val = sess.run([self.loss_op] + sum_train_ops + self.update_ops + self.train_ops, feed_dict=feed_dict)

                    n_sum_ops = len(sum_train_ops)+1
                    loss = ret_val[0]
                    metrics = ret_val[:n_sum_ops]
                    self.sum_han.train_iter(metrics, ite, epoch_per, start_time)
                    if self.network.has_update():
                        update_np = ret_val[n_sum_ops:n_sum_ops+n_update_ops]
                    
                    #check for NAN loss
                    if np.isnan(loss):
                        self.network.debug_nan_callback(sess, epoch_per, feed_dict)
                        print(bcolors.FAIL + 'Loss is NAN - Exiting' + bcolors.ENDC)
                        raise ValueError('Loss is NAN - Exiting')

                    if self.args.test:
                        #write out eval data if it is provided during testing
                        self.write_eval(sess, feed_dict, epoch_per);

                    # EVAL
                    if ite % self.args.eval_every == self.args.eval_every-1 and ite != self.data_loader.n_iter() - 1:
                        self.do_eval(sess, epoch_per)

                    #ending callback
                    self.network.end_iter(sess, epoch_per, ite, iters, metrics)

                # end of epoch metrics
                epoch_metrics = self.sum_han.print_epoch(epoch, iters, epoch_start_time)
                epoch_loss = epoch_metrics[0]
                self.network.end_epoch(sess, epoch, epoch_metrics, self.sum_han.output_log)

                #flush summaries
                self.sum_han.epoch_end()

                #print learning rate
                if not self.args.test:
                    print(bcolors.BOLD+'Learning Rate: '+str(sess.run(self.learning_rate_op))+bcolors.ENDC)

                # eval
                if self.data_loader.can_eval():
                    self.do_eval(sess, epoch)

                # **********************
                # Checkpoint
                if not self.args.test:
                    if (epoch % self.args.checkpoint_every) == self.args.checkpoint_every - 1:
                        # checkpoint
                        checkpoint_file = os.path.join(self.checkpoint_dir, 'checkpoint.tf')
                        print('Saving checkpoint to file '+checkpoint_file)
                        self.saver.save(sess, checkpoint_file)

            
            self.data_loader.stop(sess)
            self.sum_han.finish()
            print(bcolors.OKGREEN + 'Finished' + bcolors.ENDC)

    #***********************************************************
    def make_learning_ops(self, network, optimizer):
        """

        pulls criterions from the network and compiles them
        return: train_ops - list training operations to be called by trainer, there are instances of tf.optimizer
                global_steps - list of global steps for each training op
                loss_op - sum of losses from each training_op
                learning_rate_print_op - the learning rate op for the first criterion, to be printed
        """

        def create_training_op(args, criterion, scope, learning_rate):
            # create training op
            global_step = tf.Variable(0, name='global_step', trainable=False)
            if args.learning_rate_decay == 1.0:
                learning_rate_op = tf.constant(learning_rate)
            else:
                learning_rate_op = tf.train.exponential_decay(learning_rate, global_step,
                                                          args.learning_rate_decay_every * self.data_loader.n_iter(),
                                                          args.learning_rate_decay, staircase=args.staircase,
                                                          name='learning_rate')

            # check for grad_by_scope
            if scope:
                #var_list = [v for v in tf.global_variables() if v.name.startswith(scope)]
                var_list = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name.startswith(scope)]
            else:
                var_list = None

            # get train opt
            return optimizer.minimize(criterion, network, self.args, learning_rate_op,
                                      global_step=global_step, var_list=var_list), learning_rate_op, global_step

        # create training ops
        train_ops = []
        global_steps = []
        train_map = {}
            
        #convert learning rates to list if its input is a singleton
        learning_rates = self.args.learning_rate
        if not isinstance(learning_rates, list):
            learning_rates = [learning_rates]
        criterions = network.get_criterion()
        if not isinstance(criterions, list):
            criterions = [criterions]
        if self.args.grad_by_scope:
            scopes = network.get_grad_scopes()
        else:
            scopes = [None] * len(criterions)
            
        #make sure learning rate is the same length as criterions
        if len(learning_rates) == 1:
            learning_rates = np.tile(learning_rates, len(criterions))

        #make sure everything is the same length
        if len(criterions) != len(scopes) \
           or len(criterions) != len(learning_rates):
            raise ValueError(bcolors.FAIL+"lengths of criterion, grad_scopes, and learning rates must match"+bcolors.ENDC)

        # since its a list we want to generate training ops for each criterion individually
        first = True
        for criterion, scope, learning_rate in zip(criterions, scopes, learning_rates):
            if(learning_rate == 0.0):
                #dont bother with a learning op then
                continue
            train_op, learning_rate_op, global_step = create_training_op(self.args, criterion, scope, learning_rate)
            if first:
                learning_rate_print_op = learning_rate_op
                first = False
            train_ops.append(train_op)
            global_steps.append(global_step)
            train_map[criterion] = train_op

        return train_ops, train_map, global_steps, learning_rate_print_op

    #***********************************************************
    def make_loss_op(self, network):
        criterions = network.get_criterion()
        if not isinstance(criterions, list):
            criterions = [criterions]
         # Compute loss ops
        if len(criterions) > 1:
            # set loss opp to be the addition of all of these ops
            loss_op = tf.reduce_sum(tf.stack(criterions))
        else:
            loss_op = criterions[0]

        return loss_op
    
    #***********************************************************
    def get_saver(self, network):
        """

        Creates a tf.train.Saver for saving checkpoints for this run
        
        Return: tf.train.Saver object

        """
        # Create a saver for writing training checkpoints.
        if not self.args.test:
            if self.args.store_scope:
                return tf.train.Saver([v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name.startswith(self.args.store_scope)])
            else:
                var_list = network.get_var_list(self.args.test)
                if var_list:
                    return tf.train.Saver(var_list)
                else:
                    #default
                    return tf.train.Saver()
        else:
            return None

    class restorer():

        #***********************************************************
        def __init__(self, args, checkpoint_dir):
            self.args = args

            # create a saver for restoring from past checkpoints
            if (not self.args.dont_resume) or self.args.init_from:
                
                self.init_from = self.args.init_from
                if not self.init_from:
                    #init from not specified, try to load checkpoint from checkpoint dir
                    if os.path.isdir(checkpoint_dir):
                        self.init_from = [checkpoint_dir]
                    else:
                        print(bcolors.WARNING + 'could not find checkpoint: ' + str(checkpoint_dir))
                elif not isinstance(self.init_from, list):
                    #make init from into a list
                    self.init_from = [self.init_from]
            
                #check for invalid inputs
                if len(self.init_from) > 1 and \
                   (not self.args.restore_scope or len(self.init_from) != len(self.args.restore_scope)):
                    print(bcolors.FAIL + 'number of restore_scope arguments must match init_from arguments' + bcolors.ENDC)
                    raise ValueError('number of restore_scope arguments must match init_from arguments')

                if self.args.restore_scope:
                    print(bcolors.OKGREEN + 'Restoring from given restore_scopes' + bcolors.ENDC)
                    #restore_scope specifies to only restore from checkpoint those cariables specified in the given scope
                    vars = [[v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.startswith(restore_scope)] for restore_scope in self.args.restore_scope]
                    self.saver = [tf.train.Saver(vars_i) for vars_i in vars]
                elif self.args.restore_only_trainable:
                    #save/restore only the trainable_variables
                    #this allows saving/restoring networks with additional variables
                    print(bcolors.OKGREEN + 'Restoring only trainable' + bcolors.ENDC)
                    self.saver = []
                    for checkpoint_file in self.init_from:
                        vars = tf.contrib.framework.list_variables(checkpoint_file)
                        vars = [var for var,size in vars]
                        vars = [var for var in tf.trainable_variables() if var.name[:-2] in vars]
                        self.saver.append(tf.train.Saver(vars))
                else:
                    #Default restorer, see tensorflow documentation
                    self.saver = [tf.train.Saver()] * len(self.init_from)
            else:
                self.init_from = None
                self.saver = None

        #***********************************************************
        def restore(self, sess, epoch):
            # restore from checkpoint if necessary
            if self.init_from:
                for init_from_i, saver_i in zip(self.init_from, self.saver):
                    # init from given checkpoint
                    print(bcolors.OKGREEN + 'loading from checkpoint ' + str(init_from_i) + bcolors.ENDC)
                    latest_checkpoint = tf.train.latest_checkpoint(init_from_i)
                    if(latest_checkpoint):
                        saver_i.restore(sess, latest_checkpoint)
                    else:
                        saver_i.restore(sess, init_from_i)
                sess.run(epoch.assign(epoch + 1))
            else:
                print(bcolors.OKGREEN + 'Randomly initalizing weights' + bcolors.ENDC)


    #***********************************************************
    def do_eval(self, sess, epoch_per):
        """

        Preforms evaluation by running the evaluation batch
        Only occurs during training and if network.can_eval returns true

        :param sess: tensorflow session
        :param epoch_per: percent of epoch which has been completed
        :return: None
        """
        if self.data_loader.can_eval():
            # during eval we want to test
            self.network.set_test(True, sess)
            # grab eval batch
            feed_dict = self.data_loader.next_eval_batch()
            # run eval batch
            eval_ops = self.sum_han.get_eval_ops()
            ret_val = sess.run([self.loss_op]+eval_ops, feed_dict=feed_dict)
            # print results
            self.sum_han.eval(ret_val, epoch_per)
            # restore testing status
            self.network.set_test(self.args.test, sess)

            #dont know why you would want to use eval during testing
            #but if you do we dont need to write out for both test data and eval
            #we already use eval_write_op up above during test
            if not self.args.test:
                self.write_eval(sess, feed_dict, epoch_per)

    #***********************************************************
    def write_eval(self, sess, feed_dict, epoch_per):
        """
        
        Runs eval_write_op and writes it output to a log file
        If eval_write_func is not None, the output of eval_write_op will be
        passed to eval_write_func
        :param sess: tensorflow session
        :param feed_dict: feed_dict to use with sess
        :param epoch_per: percent of epoch which has been completed
        :return: None
        """
        if self.eval_write_op != None:
            ret_eval = sess.run(self.eval_write_op, feed_dict=feed_dict)
            if self.eval_write_func != None:
                self.eval_write_func(ret_eval, self.sum_han.output_log, epoch_per)
            else:
                self.sum_han.eval_write_to_log(ret_eval)


