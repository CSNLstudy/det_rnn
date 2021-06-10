import copy, os, sys, yaml
import pickle

import numpy as np
from scipy.interpolate import interp1d
import tensorflow as tf

from tensorboard.plugins.hparams import api as tf_hpapi

sys.path.append('../')
import det_rnn.train as utils_train
from models.base.analysis import behavior_summary, estimation_decision
from utils.plotfnc import *
from utils.general import get_logger, Progbar, export_plot

class BaseModel(tf.Module):
    def __init__(self, hp,
                 hypsearch_dict=None,
                 dtype=None, logger=None):
        """
        hp: network hyperparameters
        hypsearch_dict: todo

        Creates
         - output directory
         - logger
         - tensorboard writer; consider initializing in the train section.
         - hyperparameter yaml file
        """


        super(BaseModel, self).__init__()
        self.dtype = dtype
        self.hp = copy.deepcopy(hp)

        #self.build()

        # set up directories for the model, logs and tb
        self.hp['output_path'] = self.hp['output_base'] + '{:03}'.format(self.hp['model_number'])
        while os.path.exists(self.hp['output_path']):
            # find a new folder, increment model number
            self.hp['model_number'] += 1
            self.hp['output_path'] = self.hp['output_base'] + '{:03}'.format(self.hp['model_number'])
        os.makedirs(self.hp['output_path'])
        self.hp['model_output'] = os.path.join(self.hp['output_path'], 'model')
        self.hp['log_path'] = os.path.join(self.hp['output_path'], 'logs')
        self.writer = tf.summary.create_file_writer(self.hp['output_path'])
        print('model output path = ' + self.hp['output_path'])

        if hypsearch_dict is not None:
            with self.writer.as_default():
                tf_hpapi.hparams(hypsearch_dict)  # record the values used in this trial in tensorboard

        # for saving model; called by self.save_model
        if not os.path.exists(self.hp['model_output']):
            os.makedirs(self.hp['model_output'])

        # logger
        self.logger = logger
        if logger is None:
            self.logger = get_logger(self.hp['log_path'])

        self.scheduler = None # implement some scheduler

        # dump all the hyperparameters into a readable YAML file
        with open(self.hp['output_path'] + os.path.sep + 'hp.yaml', 'w') as outfile:
            yaml.dump(hp, outfile, default_flow_style=False)

    # all the networks to train goes here.
    def build(self):
        raise NotImplementedError

    ''' Train operations'''
    # todo: implement train; add schedulers?

    def evaluate(self, trial_info):
        raise NotImplementedError

        # should return
        # (1) a loss struct; should contain the final aggregated loss (unregularized); loss_struct['loss']
            # regularization loss for submodules are calculated via
        # (2) output struct (i.e. logits)
        return 0

    def calc_loss(self, labels, logits):
        raise NotImplementedError

    ''' UTILS '''
    def _initialize_variable(self, hp):
        raise NotImplementedError

    def visualize_weights(self):
        raise NotImplementedError

    ''' Train operations'''
    def train(self, stim_train, stim_test, niter = None):
        """
        """
        if niter is None:
            niter = self.hp['nsteps_train']

        t = 0  # time control of nb of steps
        loss_train = []
        loss_test = []

        prog = Progbar(target=niter)

        # test on the same test set.
        test_data       = utils_train.tensorize_trial(stim_test.generate_trial(), dtype=self.dtype)
        self.optimizer = tf.optimizers.Adam(self.hp['learning_rate']) # use same optimizer across iterations

        # interact with environment
        while t < niter:
            t += 1

            # perform a training step
            train_data = utils_train.tensorize_trial(stim_train.generate_trial(), dtype = self.dtype)
            loss_struct, grad = self.update_step(train_data)
            loss_train += [loss_struct['loss']]

            if self.trainable_variables is None:
                print('No trainable variables. Done training! ')
                break

            # evaluate loss on the same test set
            train_lossStruct, train_outputs = self.evaluate(train_data)
            test_lossStruct, test_outputs   = self.evaluate(test_data)
            loss_test                       += [test_lossStruct['loss']]

            # output behavior summary
            train_est_summary, train_dec_summary = behavior_summary(train_data, train_outputs, stim_train)
            est_summary, dec_summary            = behavior_summary(test_data, test_outputs, stim_test)

            # add losses to tensorboard
            with self.writer.as_default():
                tf.summary.scalar('Test est cos performance', np.mean(np.abs(est_summary['est_perf'])),step=t)
                tf.summary.scalar('Test dec performance', np.mean(np.abs(dec_summary['dec_perf'])), step=t)

                tf.summary.scalar('Train est cos performance', np.mean(np.abs(train_est_summary['est_perf'])), step=t)
                tf.summary.scalar('Train dec performance', np.mean(np.abs(train_dec_summary['dec_perf'])), step=t)
                tf.summary.scalar('Gradient norm', grad, step = t)
                for (name, val) in loss_struct.items():
                    tf.summary.scalar('train ' + name, val, step=t)
                for (name, val) in test_lossStruct.items():
                    tf.summary.scalar('test ' + name, val, step=t)

                if self.hp.__contains__('tau_max'):
                    tf.summary.scalar('tau max', self.hp['tau_max'], step=t)
                    tf.summary.scalar('tau min', self.hp['tau_min'], step=t)

                if self.scheduler is not None:
                    tf.summary.scalar('(Schedule) off', self.scheduler.switch, step=t)
                    for (name, val) in self.scheduler.get_params().items():
                        tf.summary.scalar('(Schedule) params ' + name, val, step=t)

            # logging stuff
            prog.update(t, exact=[("Train Loss", loss_struct['loss']),
                                  ("Test Loss", test_lossStruct['loss']),
                                  ("Test est cos(error)", np.mean(np.abs(est_summary['est_perf']))),
                                  ("Test dec perf", np.mean(dec_summary['dec_perf'])),
                                  ("Grads", grad)])

            if (t % self.hp['saving_freq']) == 1:
                template = '\n Epoch {}, Train Loss: {}, Test Loss: {}'
                self.logger.info(template.format(t + 1,
                                                loss_struct['loss'],
                                                test_lossStruct['loss']))

                savestruct = {'test_lossStruct': test_lossStruct,
                              'test_outputs': test_outputs,
                              'loss_test': loss_test,
                              'est_summary': est_summary,
                              'dec_summary': dec_summary,
                              'stim_test': stim_test,
                              'test_data': test_data}

                self.save_model(t, savestruct) #Save Model
                #self.visualize_weights()

            # update scheduler
            if self.scheduler is not None:
                self.scheduler.update(t,
                                      np.mean(np.abs(est_summary['est_perf'])),
                                      np.mean(np.abs(dec_summary['dec_perf'])))

            # stop if the network is good enough
            if grad == 0:
                print('(rnn training) grad is zero. Terminating...')
                t = niter

            if (np.mean(np.abs(est_summary['est_perf'])) > 0.95 and \
                np.mean(np.abs(dec_summary['dec_perf'])) > 0.95):
                print('(rnn training) Reached good performance! Terminating')
                t = niter

        # last words
        self.logger.info("- Training done.")
        self.save_model('final', savestruct) # save and output behavior
        export_plot(loss_train, loss_test, 'loss' , self.hp['output_path'] + os.path.sep + 'losses')

        # output graph on tensorflow using trace (need to enable tf.function
        # tf.summary.trace_on(graph=True, profiler=False)
        # test_data       = utils_train.tensorize_trial(stim_test.generate_trial(), dtype = self.dtype)
        # test_input      = test_data['input_tuning']
        # test_Y          = self.__call__(test_input)
        # with self.writer.as_default():
        #     tf.summary.trace_export(name="loss",step=t)

    def update_step(self, trial_info):
        """
        trial_info  : from the stimulus struct
        t           : time
        """
        if self.scheduler is not None:
            # note: if using scheduler, may need to do SGD, since the loss function change and the lr changes
            # reset adam optimizer.
            self.optimizer = tf.optimizers.Adam(self.scheduler.get_params()['learning_rate'])
            # self.optimizer.lr.assign(self.scheduler.get_params()['learning_rate'])
            # otherwise, keep the same optimizer as before

        # calculate gradients
        with tf.GradientTape() as tape:
            loss_struct, outputs = self.evaluate(trial_info)

            # add regularization loss; todo: check that this wasn't added in the evalute (or elsewhere)
            reg_loss = []
            for smod in self.submodules:
                # todo: dataformat when we add activation or bias losses?
                if "losses" in dir(smod):
                    reg_loss += smod.losses # losses store the regularizer losses;
            alllosses = loss_struct['loss'] + tf.reduce_sum(tf.squeeze(reg_loss))
        grads = tape.gradient(alllosses, self.trainable_variables)  # vardict is in trainable_variables if they are variables

        # todo: check regularization in new networks
        '''
        l2 = lambda_l2_reg * sum(
            tf.nn.l2_loss(tf_var)
            for tf_var in tf.trainable_variables()
            if not ("noreg" in tf_var.name or "Bias" in tf_var.name)
        )
        '''

        # clip gradient
        grads_clipped = []  # gradient capping and clipping
        for grad in grads:
            if tf.math.reduce_any(tf.math.is_nan(grad)):
                print('wtf grads are NaNs')
                assert not tf.math.reduce_any(tf.math.is_nan(grads))
                # tf.softmax(trial_info['output_tuning'])
            grads_clipped.append(tf.clip_by_norm(grad, self.hp['clip_max_grad_val']))
        self.optimizer.apply_gradients(zip(grads_clipped, self.trainable_variables))
        gradnorm = tf.linalg.global_norm(grads)

        return loss_struct, gradnorm

        # todo: add all these for RNN.
        # 'neural_input'  : self.neural_input.astype(np.float32),
        #                 'desired_output': self.desired_output.astype(np.float32),
        #                 'mask'          : self.mask,
        #                 'input_tuning'  : self.input_tuning,
        #                 'stimulus_ori'  : self.stimulus_ori,
        #                 'stimulus_kap'  : self.stimulus_kap

    def save_model(self, t, savestruct):
        filename = self.hp['model_output'] + os.path.sep +'iter' + str(t)
        # self.save_weights(filename)

        tf.saved_model.save(self, filename)
        self.plot_summary(savestruct, filename=filename)

    def plot_summary(self, savestruct, filename = None):
        test_lossStruct = savestruct['test_lossStruct']
        test_outputs    = savestruct['test_outputs']
        loss_test       = savestruct['loss_test']
        est_summary     = savestruct['est_summary']
        dec_summary     = savestruct['dec_summary']
        stim_test       = savestruct['stim_test']
        test_data       = savestruct['test_data']

        if filename is None:
            pass
        else:
            # estimation summary)
            behavior_figure(est_summary, filename=os.path.join(filename, 'BehaviorSummary'))

            # plot rnn decision effects on estimation
            df_trials, df_sum = estimation_decision(test_data, test_outputs, stim_test)
            plot_decision_effects(df_trials, df_sum, filename=os.path.join(filename, 'DecisionEffectsOnEstim'))
            plot_rnn_output(test_data, test_outputs, stim_test, savename=os.path.join(filename,'sampleTrial'))

        return df_trials, df_sum
