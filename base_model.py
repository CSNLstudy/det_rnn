import os, sys, yaml

import numpy as np
from scipy.interpolate import interp1d
import tensorflow as tf

from tensorboard.plugins.hparams import api as tf_hpapi

sys.path.append('../')
import det_rnn.train as utils_train
import det_rnn.analysis as utils_analysis
from utils.general import get_logger, Progbar, export_plot

class BaseModel(tf.Module):
    def __init__(self, hp, par_train,
                 hypsearch_dict=None,
                 dtype=tf.float32, logger=None):
        """
        hp: network hyperparameters
        par_train: stimulus parameters used to train the network (since efficient coding is dependent on stimulus distribution)

        Creates
         - output directory
         - logger
         - tensorboard writer; consider initializing in the train section.
         - hyperparameter yaml file
        """

        super(BaseModel, self).__init__()
        self.hp = hp

        self.dtype = dtype
        self._initialize_variable(hp, par_train)
        self.build()

        # tensorboard
        if not os.path.exists(self.hp['output_path']):
            os.makedirs(self.hp['output_path'])
        self.writer = tf.summary.create_file_writer(self.hp['output_path'])

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

        # dump all the hyperparameters into a readable YAML file
        with open(self.hp['output_path'] + os.path.sep + 'data.yml', 'w') as outfile:
            yaml.dump(hp, outfile, default_flow_style=False)

    # all the networks to train goes here.
    def build(self):
        raise NotImplementedError

    ''' Train operations'''
    # todo: implement train; add schedulers?

    def evaluate(self, trial_info):
        # inputs and outputs from the trial
        neural_input    = trial_info['input_tuning']
        output_tuning   = trial_info['output_tuning']
        labels          = tf.math.argmax(output_tuning,1)

        raise NotImplementedError

        # should return
        # (1) a loss struct; should contain the final aggregated loss (unregularized); loss_struct['loss']
            # regularization loss for submodules are calculated via
        # (2) output struct (logits)
        return 0

    def calc_loss(self, labels, logits):
        raise NotImplementedError

    ''' UTILS '''
    def _initialize_variable(self, hp, par_train):
        raise NotImplementedError

    def train(self, stim_train, stim_test):
        """
        """

        t = 0  # time control of nb of steps
        loss_train = []
        loss_test = []

        prog = Progbar(target=self.hp['nsteps_train'])

        # interact with environment
        while t < self.hp['nsteps_train']:
            t += 1

            # perform a training step
            train_data = utils_train.tensorize_trial(stim_train.generate_trial(), dtype = self.dtype)
            loss_struct, grad = self.update_step(train_data, t)
            loss_train += [loss_struct['loss']]

            # evaluate loss on test set
            test_data                       = utils_train.tensorize_trial(stim_test.generate_trial(), dtype = self.dtype)
            test_lossStruct, test_logits    = self.evaluate(test_data)
            loss_test                       += [test_lossStruct['loss']]

            ground_truth, estim_mean, raw_error, beh_perf = utils_analysis.behavior_summary(test_data, test_logits,
                                                                                            stim_test)
            errors_circular = np.arctan2(np.sum(np.sin(2 * raw_error)), np.sum(np.cos(2 * raw_error))) / 2
            errors          = np.arctan2(np.sin(2 * raw_error), np.cos(2 * raw_error)) / 2
            # note that this error is not circular mean.

            # add losses to tensorboard
            with self.writer.as_default():
                tf.summary.scalar('Test average error', np.mean(np.abs(errors)), step=t)
                for (name, val) in loss_struct.items():
                    tf.summary.scalar('train ' + name, val, step=t)
                for (name, val) in test_lossStruct.items():
                    tf.summary.scalar('test ' + name, val, step=t)


            # logging stuff
            prog.update(t, exact=[("Train Loss", loss_struct['loss']),
                                  ("Test Loss", test_lossStruct['loss']),
                                  ("Grads", grad)])

            if (t % self.hp['saving_freq']) == 1:
                template = '\n Epoch {}, Train Loss: {}, Test Loss: {}'
                self.logger.info(template.format(t + 1,
                                                loss_struct['loss'],
                                                test_lossStruct['loss']))

                self.save_model(t, stim_test) #Save Model

        # last words
        self.logger.info("- Training done.")
        self.save_model('final', stim_test) # save and output behavior
        export_plot(loss_train, loss_test, 'loss' , self.hp['output_path'] + os.path.sep + 'losses')

        # output graph on tensorflow using trace (need to enable tf.function
        tf.summary.trace_on(graph=True, profiler=False)
        test_data       = utils_train.tensorize_trial(stim_test.generate_trial(), dtype = self.dtype)
        test_input      = test_data['input_tuning']
        test_Y          = self.__call__(test_input)
        with self.writer.as_default():
            tf.summary.trace_export(name="loss",step=t)

    def update_step(self, trial_info):
        """
        trial_info  : from the stimulus struct
        t           : time
        """
        optimizer = tf.optimizers.Adam(self.hp['learning_rate']) #todo: flexible learning rate

        # calculate gradients
        with tf.GradientTape() as tape:
            loss_struct, logits = self.evaluate(trial_info)

            # add regularization loss; todo: check that this wasn't added in the evalute (or elsewhere)
            reg_loss = []
            for smod in self.submodules:
                reg_loss += smod.losses # losses store the regularizer losses; todo: dataformat when we add activation or bias losses?
            alllosses = loss_struct['loss'] + tf.reduce_sum(tf.squeeze(reg_loss))
        grads = tape.gradient(alllosses, self.trainable_variables)  # vardict is in trainable_variables if they are variables

        # clip gradient
        grads_clipped = []  # gradient capping and clipping
        for grad in grads:
            if tf.math.reduce_any(tf.math.is_nan(grad)):
                print('wtf grads are NaNs')
                assert not tf.math.reduce_any(tf.math.is_nan(grads))
                # tf.softmax(trial_info['output_tuning'])
            grads_clipped.append(tf.clip_by_norm(grad, self.hp['clip_max_grad_val']))
        optimizer.apply_gradients(zip(grads_clipped, self.trainable_variables))
        gradnorm = tf.linalg.global_norm(grads)

        return loss_struct, gradnorm

        # todo: add all these for RNN.
        # 'neural_input'  : self.neural_input.astype(np.float32),
        #                 'desired_output': self.desired_output.astype(np.float32),
        #                 'mask'          : self.mask,
        #                 'input_tuning'  : self.input_tuning,
        #                 'stimulus_ori'  : self.stimulus_ori,
        #                 'stimulus_kap'  : self.stimulus_kap

    def save_model(self, t, stim_test):
        filename = self.hp['model_output'] + os.path.sep +'iter' + str(t)
        tf.saved_model.save(self, filename)

        # todo: output behavior plot
        test_data               = utils_train.tensorize_trial(stim_test.generate_trial(), dtype = self.dtype)
        test_input              = test_data['input_tuning']
        lossStruct, test_Y      = self.evaluate(test_input)

        ground_truth, estim_mean, raw_error, beh_perf   = utils_analysis.behavior_summary(test_data, test_Y, stim_test)
        utils_analysis.behavior_figure(ground_truth, estim_mean, raw_error, beh_perf,
                                       filename = filename + os.path.sep + 'behavior')
        utils_analysis.biasvar_figure(ground_truth, estim_mean, raw_error, stim_test,
                                      filename = filename + os.path.sep + 'Bias Variance')