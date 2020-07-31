import os, sys, yaml

import numpy as np
from scipy.interpolate import interp1d
import tensorflow as tf

from tensorboard.plugins.hparams import api as tf_hpapi

sys.path.append('../')
from .vnet_hyper import hp
import det_rnn.train as utils_train
import det_rnn.analysis as utils_analysis
from utils.general import get_logger, Progbar, export_plot

EPSILON = 1e-7

__all__ = ['VNET']

class VNET(tf.Module):
    def __init__(self, hp, par_train,
                 hypsearch_dict=None,
                 dtype=tf.float32, logger=None):
        """
        hp: network hyperparameters
        par_train: stimulus parameters used to train the network (since efficient coding is dependent on stimulus distribution)
        """
        super(VNET, self).__init__()
        self.hp = hp
        self.dtype = dtype
        self._initialize_variable(hp, par_train)
        self.build()

        ### should I initialize these in the train?
        # tensorboard
        if not os.path.exists(self.hp['output_path']):
            os.makedirs(self.hp['output_path'])
        self.writer = tf.summary.create_file_writer(self.hp['output_path'])

        if hypsearch_dict is not None:
            with self.writer.as_default():
                tf_hpapi.hparams(hypsearch_dict)  # record the values used in this trial in tensorboard

        # for saving model; called by self.saveModel
        if not os.path.exists(self.hp['model_output']):
            os.makedirs(self.hp['model_output'])

        # logger
        self.logger = logger
        if logger is None:
            self.logger = get_logger(self.hp['log_path'])

        # dump all the hyperparameters into a readable YAML file
        with open(self.hp['output_path'] + os.path.sep + 'data.yml', 'w') as outfile:
            yaml.dump(hp, outfile, default_flow_style=False)

    # @tf.Module.with_name_scope
    #@tf.function
    def __call__(self, neural_input):
        [sens_m, sens_p, sens_act, sens_g] = self.sensory_layer(neural_input)
        logits = self.output_layer(sens_m)

        #return logits
        return logits

    # all the networks to train goes here.
    def build(self):
        """
        self.stim_tuning = tf.keras.layers.Dense(1, input_shape = (self.hp['n_tuned_input'],),
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.hp['loss_L1'], l2=self.hp['loss_L2']),
                        name='sensory_mu')

        self.sens_logk   = tf.keras.layers.Dense(1, input_shape = (self.hp['n_tuned_input'],),
                       kernel_regularizer = tf.keras.regularizers.l1_l2(l1=self.hp['loss_L1'], l2=self.hp['loss_L2']),
                       name='sensory_k') # log(kappa) is a real number => positive kapp; no bias

        if 'sensory_gain' is True:
            self.sens_g = tf.keras.layers.Dense(1, input_shape = (self.hp['n_tuned_input'],),
                             activation=tf.nn.relu,
                             kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.hp['loss_L1'], l2=self.hp['loss_L2']),
                             name='sensory_gain')  # positive gain

        if self.hp['sensory_noise_type'] == 'Normal_learn':
            self.sensory_noiseStd = tf.keras.layers.Dense(self.hp['n_sensory'], input_shape = (self.hp['n_tuned_input'],),
                             activation=tf.nn.relu,
                             kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.hp['loss_L1'], l2=self.hp['loss_L2']),
                             name='sensory_noiseStd')

        self.output_layer = tf.keras.layers.Dense(self.hp['n_tuned_output'], input_dim = (self.hp['n_sensory'],),
                        kernel_regularizer = tf.keras.regularizers.l1_l2(l1=self.hp['loss_L1'], l2=self.hp['loss_L2']),
                        name='output_logits')  # logits; can be negative
        """
        self.stim_tuning = tf.keras.Sequential(name='sensory_mu')
        self.stim_tuning.add(tf.keras.layers.Dense(self.hp['n_tuned_input'], input_shape = (self.hp['n_tuned_input'],),
                                                   kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.hp['loss_L1'], l2=self.hp['loss_L2'])))
        self.stim_tuning.add(tf.keras.layers.Dense(self.hp['n_tuned_input'],
                                                   kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.hp['loss_L1'],
                                                                                                  l2=self.hp['loss_L2'])))
        self.stim_tuning.add(tf.keras.layers.Dense(1,
                                                   kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.hp['loss_L1'],
                                                                                                  l2=self.hp[
                                                                                                      'loss_L2'])))


        self.sens_logk   = tf.keras.Sequential(name='sensory_k')
        self.sens_logk.add(tf.keras.layers.Dense(self.hp['n_tuned_input'], input_shape = (self.hp['n_tuned_input'],),
                                                 kernel_regularizer = tf.keras.regularizers.l1_l2(l1=self.hp['loss_L1'], l2=self.hp['loss_L2'])))
        self.sens_logk.add(tf.keras.layers.Dense(self.hp['n_tuned_input'],
                                                 kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.hp['loss_L1'],
                                                                                                l2=self.hp['loss_L2'])))
        self.sens_logk.add(tf.keras.layers.Dense(1,
                                                 kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.hp['loss_L1'],
                                                                                                l2=self.hp['loss_L2'])))

        if 'sensory_gain' is True:
            self.sens_g = tf.keras.layers.Dense(1, input_shape = (self.hp['n_tuned_input'],),
                             activation=tf.nn.relu,
                             kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.hp['loss_L1'], l2=self.hp['loss_L2']),
                             name='sensory_gain')  # positive gain

        if self.hp['sensory_noise_type'] == 'Normal_learn':
            self.sensory_noiseStd = tf.keras.layers.Dense(self.hp['n_sensory'], input_shape = (self.hp['n_tuned_input'],),
                             activation=tf.nn.relu,
                             kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.hp['loss_L1'], l2=self.hp['loss_L2']),
                             name='sensory_noiseStd')

        self.output_layer  = tf.keras.Sequential(name='output_logits') #errors out here for somereason if i change the input_dim...
        self.output_layer.add(tf.keras.layers.Dense(self.hp['n_tuned_output'], input_dim = self.hp['n_sensory'],
                                kernel_regularizer = tf.keras.regularizers.l1_l2(l1=self.hp['loss_L1'], l2=self.hp['loss_L2'])))
        self.output_layer.add(tf.keras.layers.Dense(self.hp['n_tuned_output'],
                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.hp['loss_L1'], l2=self.hp['loss_L2'])))
        self.output_layer.add(tf.keras.layers.Dense(self.hp['n_tuned_output'],
                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.hp['loss_L1'], l2=self.hp['loss_L2'])))

    ''' Train operations'''
    # todo: implement train; add schedulers?
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

                self.saveModel(t, stim_test) #Save Model

        # last words
        self.logger.info("- Training done.")
        self.saveModel('final', stim_test) # save and output behavior
        export_plot(loss_train, loss_test, 'loss' , self.hp['output_path'] + os.path.sep + 'losses')

        # output graph on tensorflow using trace (need to enable tf.function
        tf.summary.trace_on(graph=True, profiler=False)
        test_data       = utils_train.tensorize_trial(stim_test.generate_trial(), dtype = self.dtype)
        test_input      = test_data['input_tuning']
        test_Y          = self.__call__(test_input)
        with self.writer.as_default():
            tf.summary.trace_export(name="loss",step=t)

    def evaluate(self, trial_info):
        # inputs and outputs from the trial
        neural_input    = trial_info['input_tuning']
        output_tuning   = trial_info['output_tuning']
        labels          = tf.math.argmax(output_tuning,1) # todo: mode this to stimulus?

        # run model
        [sens_m, sens_p, sens_act, sens_g]          = self.sensory_layer(neural_input)
        logits                                      = self.output_layer(sens_m)
        [loss, loss_ce, loss_MI, loss_smoothPost, loss_pe]   = self.calc_loss(labels, logits)

        return {'loss': loss, 'loss_ce': loss_ce, 'loss_MI': loss_MI, 'loss_smoothPost': loss_smoothPost, 'loss_pe': loss_pe}, \
               logits

    def update_step(self, trial_info, t):
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

    ''' layers '''
    #@tf.function # todo: change back to tf.function
    def sensory_layer(self, neural_input):
        # population tuning (1 distribution for vonMises likelihood in the sensory space) neural network to define sensory inputs and outputs
        # sens_mu = tf.math.acos(tf.math.cos(self.stim_tuning(neural_input))/(1+EPSILON)) # convert range to 0 to pi
        # note derivaitve of acos(x) is nan if x = 1; so I scale by epsilon here...
        # note: the derivative here is very weird. tuning = 0 deg has almost infinite gradient
        # deriviatve of asin(sin(x)) = tan(X), and d of acos(cos(x)) = cot(x) [ infinite at different locations]
        # sens_mu = tf.math.floormod(self.stim_tuning(neural_input),tf.constant(np.pi))
        # note: if you use mod, the operation is not circular => so the derivative might go a wrong way
        sens_mu_real = self.stim_tuning(neural_input)
        sens_mu = tf.math.atan2(tf.math.sin(sens_mu_real), tf.math.cos(sens_mu_real))

        sens_logk = self.sens_logk(neural_input)

        # Von Mises population distribution; todo: check gradient?
        # shape = B x n_sensory
        sens_p = tf.math.exp(tf.exp(sens_logk) * tf.math.cos(2*(self.sensory_tunings- sens_mu),
                                                            name='sensory_vm_cos'),
                             name='sensory_vm_exp')/\
                     (tf.constant(2.*np.pi)*tf.math.bessel_i0e(tf.exp(sens_logk), name='sensory_vm_bessel'))

        if 'sensory_gain' is True:
            sens_g = self.sens_g(neural_input)
            sens_act = sens_g * sens_p  # todo (?): multiply a constant gain to all input units
        else:
            sens_g = None
            sens_act = sens_p

        # Sensory measurement. add noise todo: how do we properly scale noise? learn weights for noise, proportional to gain?
        if self.hp['sensory_noise_type'] == 'Normal_fixed':
            # scale neural noise by time constants of the neuron.... todo: check if how the noise variance is calculated
            noise = tf.random.normal(sens_act.shape, 0,
                                     tf.sqrt(2 * self.hp['alpha_neuron'] * hp['noise_sd']),
                                     dtype=self.dtype)
            sens_m = sens_act + noise
        elif self.hp['sensory_noise_type'] == 'Normal_learn':  # assume average of poisson neurons; scale by time constants
            noiseStd = self.sensory_noiseStd(neural_input)
            noise = tf.random.normal(sens_act.shape, 0,
                                     tf.sqrt(2 * self.hp['alpha_neuron']),
                                     dtype=self.dtype)
            sens_m = sens_act + noiseStd * noise
        elif self.hp['sensory_noise_type'] == 'Normal_poisson':  # assume average of poisson neurons; scale by time constants
            noise = tf.random.normal(sens_act.shape, 0,
                                     tf.sqrt(2 * self.hp['alpha_neuron']),
                                     dtype=self.dtype)
            sens_m = sens_act + tf.math.sqrt(sens_act) * noise
        elif self.hp['sensory_noise_type'] == 'Poisson': # return measurement, with poisson noise
            # todo: gradient for poisson samples doesnt work; is there a reparametrization trick?
            #sens_m = tf.squeeze(tf.random.poisson((1,), sens_act, dtype=self.dtype, name = 'sensory_noise')) # support: natural numbers
            raise NotImplementedError
        else:
            sens_m = sens_act

        return sens_m, sens_p, sens_act, sens_g

    #@tf.function
    def calc_loss(self, labels, logits):
        # main loss
        #loss_ce = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_probs, logits,
        #                                                                      axis=-1, name='loss_CE'))
        loss_ce = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.one_hot(labels, depth = self.hp['n_tuned_output']),
                                                                              logits, axis=-1, name='loss_CE'))

        # a = tf.nn.softmax_cross_entropy_with_logits(tf.one_hot(labels, depth = self.hp['n_tuned_output']),logits, axis=-1, name='loss_CE')
        # tf.math.argmax(logits,1); tf.math.argmax(tf.nn.softmax(logits),1)
        # problem: logits is not diverse..

        # todo: implement "prediction error" signal
        post_prob       = tf.nn.softmax(logits, axis=1)
        post_support    = tf.expand_dims(tf.constant(np.arange(0, np.pi, np.pi / self.hp['n_tuned_output']),dtype=self.dtype),axis=1)  # 0 to pi
        post_mean       = tf.math.atan2(tf.matmul(post_prob, tf.sin(2 * post_support)),
                                        tf.matmul(post_prob, tf.cos(2 *  post_support))) / 2  # -pi/2 to pi/2
        ground_truth    = tf.matmul(tf.one_hot(labels, depth = self.hp['n_tuned_output']),post_support)
        raw_error       = post_mean - ground_truth
        errors          = tf.math.atan2(tf.math.sin(2 * raw_error), tf.math.cos(2 * raw_error)) / 2
        loss_pe         = tf.reduce_mean(tf.square(errors))


        # info loss
        loss_MI = 0. # todo: is there a tractable way to implement this?

        # smoothness loss
        loss_smoothPost = tf.square(tf.norm(tf.linalg.matmul(logits, self.laplacian),ord='euclidean'))

        # total loss
        loss = loss_ce \
               + self.hp['loss_MI'] * loss_MI \
               + self.hp['loss_p_smooth'] * loss_smoothPost\
               + self.hp['loss_pe'] * loss_pe


        return loss, loss_ce, loss_MI, loss_smoothPost, loss_pe

    ''' UTILS '''
    def _initialize_variable(self, hp, par_train):
        # todo: need to change formats for all hp?
        # todo: also save other variables?

        # bias for sensory tunings
        if self.hp['sensory_repr'] is 'Efficient': # i.e. Wei and Stocker 2015; make prior distribution uniform in sensory space
            stimulus_tunings = np.float32(np.arange(0, 180, 180 / par_train['n_ori']))  # do not include 180
            stim_cdf = np.cumsum(par_train['stim_p'])

            finterp = interp1d(stimulus_tunings, stim_cdf, kind = 'cubic')
            unif_prior_space = np.float32(np.linspace(0, 1, hp['n_sensory']))  # include 0 and 1
            sensory_tunings = finterp(unif_prior_space)

            self.sensory_tunings = tf.convert_to_tensor(sensory_tunings,
                                                        name='sensory_tunings', dtype=self.dtype)
        elif self.hp['sensory_repr'] is 'Uniform': # uniform in stimulus space, do not include both pi
            self.sensory_tunings = tf.constant(np.arange(0, np.pi, np.pi/hp['n_sensory']),
                                               name='sensory_tunings', dtype=self.dtype)
        elif self.hp['sensory_repr'] is 'Learn': # set as free parameters,
            self.sensory_tunings    = tf.Variable(np.arange(0, np.pi, np.pi/hp['n_sensory']),
                                                  name='sensory_tunings', dtype=self.dtype)


        # generate laplacian kernel for smoothness loss
        laplacmat = np.zeros((hp['n_tuned_output'],hp['n_tuned_output']))
        laplacmat[0,0] = -0.25
        laplacmat[1,0] = 0.5
        laplacmat[2,0] = -0.25

        for shift in range(hp['n_tuned_output']):
            laplacmat[:,shift] = np.roll(laplacmat[:,0], shift=shift, axis=0)

        self.laplacian = tf.constant(laplacmat,dtype = self.dtype, name='laplacian')

    def saveModel(self, t, stim_test):
        filename = self.hp['model_output'] + os.path.sep +'iter' + str(t)
        tf.saved_model.save(self, filename)

        # todo: output behavior plot
        test_data       = utils_train.tensorize_trial(stim_test.generate_trial(), dtype = self.dtype)
        test_input      = test_data['input_tuning']
        test_Y          = self.__call__(test_input)

        ground_truth, estim_mean, raw_error, beh_perf   = utils_analysis.behavior_summary(test_data, test_Y, stim_test)
        utils_analysis.behavior_figure(ground_truth, estim_mean, raw_error, beh_perf,
                                       filename = filename + os.path.sep + 'behavior')
        utils_analysis.biasvar_figure(ground_truth, estim_mean, raw_error, stim_test,
                                      filename = filename + os.path.sep + 'Bias Variance')