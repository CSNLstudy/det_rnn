import os, sys, yaml

import numpy as np
from scipy.interpolate import interp1d
import tensorflow as tf

from tensorboard.plugins.hparams import api as tf_hpapi

sys.path.append('../')
from base_model import BaseModel
import det_rnn.train as utils_train
import det_rnn.analysis as utils_analysis
from utils.general import get_logger, Progbar, export_plot

EPSILON = 1e-7

__all__ = ['RVNET']

"""Variational network with ring attractors (i.e. Ben-Yishai)"""

class RVNET(BaseModel):
    def __init__(self, hp, par_train,
                 hypsearch_dict=None, logger=None):
        """
        hp: network hyperparameters
        par_train: stimulus parameters used to train the network (since efficient coding is dependent on stimulus distribution)
        """
        super(RVNET, self).__init__(hp,
                                    par_train,
                                    hypsearch_dict=hypsearch_dict,
                                    dtype=hp['dtype'],
                                    logger=logger)

        self._initialize_variable(hp, par_train)
        #self.build()

    def evaluate(self, trial_info):
        sensory_input   = self.sensory_layer(trial_info['input_tuning'])
        output, states  = self.rnn(sensory_input)
        lossStruct      = self.calc_loss(trial_info,output)

        # should return
        # (1) a loss struct; should contain the final aggregated loss (unregularized); loss_struct['loss']
            # regularization loss for submodules are calculated via
        # (2) output struct (logits)
        return lossStruct, \
               {'output': output, 'states': states}

    def calc_loss(self, trial_info, rnn_output):
        # inputs and outputs from the trial
        neural_input        = trial_info['input_tuning']
        neural_output       = trial_info['desired_output']
        output_tuning       = trial_info['output_tuning']
        labels              = tf.math.argmax(output_tuning,1)

        [B,T,N]             = rnn_output.shape()

        # todo: how should I interpret the neural outputs? logits? unnormalized probability?
        post_prob           = rnn_output/tf.math.sum(rnn_output,-1)

        # MSE loss
        loss_mse = tf.reduce_mean(tf.math.abs(neural_output - rnn_output))

        # sampled posterior (monte-carlo method), we want to maximize the posterior probability of the true orientation.
        loss_mcpost     = -1 * tf.math.reduce_mean(tf.gather(post_prob,labels,axis=-1))

        # absolute prediction error signal
        post_support    = tf.expand_dims(tf.constant(np.arange(0, np.pi, np.pi / self.hp['n_tuned_output']),dtype=self.dtype),axis=1)  # 0 to pi
        post_mean       = tf.math.atan2(tf.matmul(post_prob, tf.sin(2 * post_support)),
                                        tf.matmul(post_prob, tf.cos(2 *  post_support))) / 2  # -pi/2 to pi/2
        ground_truth    = tf.matmul(tf.one_hot(labels, depth = self.hp['n_tuned_output']),post_support)
        raw_error       = post_mean - ground_truth
        errors          = tf.math.atan2(tf.math.sin(2 * raw_error), tf.math.cos(2 * raw_error)) / 2
        loss_pe         = tf.reduce_mean(tf.square(errors))

        # info loss; todo: is there a tractable way to implement this?
        loss_MI = 0.

        # cross-entropy loss (one-hot)
        # todo: factor out before adding logs; check precision...
        loss_ce = -1 * tf.math.reduce_mean(tf.one_hot(labels, depth=self.hp['n_tuned_output'])
                                           * tf.math.log(post_prob + self.dtype.min))

        # average smoothness loss, per time and batch
        loss_smoothPost = tf.square(tf.norm(tf.linalg.matmul(post_prob, self.laplacian),ord='euclidean'))/(B*T)

        # total loss
        loss = self.hp['loss_mse'] * loss_mse \
               + self.hp['loss_mcpost'] * loss_mcpost \
               + self.hp['loss_ce'] * loss_ce \
               + self.hp['loss_MI'] * loss_MI \
               + self.hp['loss_p_smooth'] * loss_smoothPost\
               + self.hp['loss_pe'] * loss_pe

        return {'loss'              : loss,
                'loss_MSE'          : loss_mse,
                'loss_mcpost'       : loss_mcpost,
                'loss_ce'           : loss_ce,
                'loss_MI'           : loss_MI,
                'loss_smoothPost'   : loss_smoothPost,
                'loss_pe'           : loss_pe}

    ''' UTILS '''
    def _initialize_variable(self, hp, par_train):
        # todo: need to change formats for all hp?
        # todo: also save other variables?
        # todo: same as build()??

        # build layers
        self.sensory_layer  = SensoryLayer(hp,par_train)
        self.rnncell        = RNNCell(hp, par_train)
        self.rnn            = tf.keras.layers.RNN(self.rnncell)

        ## generate laplacian kernel for smoothness loss
        laplacmat = np.zeros((hp['n_tuned_output'], hp['n_tuned_output']))
        laplacmat[0, 0] = -0.25
        laplacmat[1, 0] = 0.5
        laplacmat[2, 0] = -0.25
        for shift in range(hp['n_tuned_output']):
            laplacmat[:, shift] = np.roll(laplacmat[:, 0], shift=shift, axis=0)
        self.laplacian = tf.constant(laplacmat, dtype=self.dtype, name='laplacian')

############################ Submodules ############################

class SensoryLayer(tf.Module):
    def __init__(self, hp, par_train, **kwargs):
        self.dtype      = hp['dtype']
        self.n_ori      = par_train['n_ori']
        self.stim_size  = par_train['n_tuned_input'] # number of neurons encoding the input (uniformly distributed in (0,pi))
        self.layer_size = hp['n_sensory']
        self.ani        = tf.constant(hp['sensory_input_ani'],  dtype=self.dtype)
        self.gain       = tf.constant(hp['sensory_input_gain'], dtype=self.dtype)

        super(SensoryLayer, self).__init__(**kwargs)
        self.stimulus_tunings       = np.float32(np.arange(0, 180, 180 / par_train['n_ori']))
        self.stim_cdf               = np.cumsum(par_train['stim_p'])

        if hp['sensory_repr'] is 'Efficient':  # i.e. Wei and Stocker 2015; make prior distribution uniform in sensory space
            self.finterp            = interp1d(self.stimulus_tunings, self.stim_cdf, kind='cubic')
            self.unif_prior_space   = np.float32(np.linspace(0, 1, hp['n_sensory']))  # include 0 and 1
            self.sensory_tunings = tf.convert_to_tensor(self.finterp(self.unif_prior_space),
                                                        name='sensory_tunings', dtype=self.dtype)
        elif hp['sensory_repr'] is 'Uniform':  # uniform in stimulus space, do not include both pi
            self.sensory_tunings = tf.constant(np.arange(0, np.pi, np.pi / hp['n_sensory']),
                                               name='sensory_tunings', dtype=self.dtype)
        elif hp['sensory_repr'] is 'Learn':  # set as free parameters,
            self.sensory_tunings = tf.Variable(np.arange(0, np.pi, np.pi / hp['n_sensory']),
                                               name='sensory_tunings', dtype=self.dtype)

        self.input_tuning          = tf.convert_to_tensor(np.arange(0, 180, 180 / self.stim_size),
                                                          dtype=self.dtype)


    def build(self, input_shape):
        # no need to build other units. (we konw the input size, and nothing is dependent on the input size)
        self.built = True

    def __call__(self, inputs):
        '''
        input shape: [B, t, n_input]
        '''
        # cell activations
        [B,T,N]         = inputs.shape
        pref_dirs_in    = tf.tile(tf.reshape(self.input_tuning,[1,1,-1,1]),
                                  [B, T, 1, self.layer_size]) # B x n_tuned_input x sensory_n

        pref_dirs_out   = tf.tile(tf.reshape(self.sensory_tunings,[1,1,1,-1]),
                                  [B, T, self.input_size, 1])

        out_tuning      = self.gain * (tf.ones([B,T,N],dtype=self.dtype) -
                                       self.ani + self.ani* tf.math.cos(2*(pref_dirs_in-pref_dirs_out)))

        # weigh the tunings by the inputs and then take the sum.
        out = tf.reduce_sum(tf.tile(inputs,[1,1,1,self.layer_size] * out_tuning),axis=3)

        return out # shape = B x T x n_layer_size

class RNNCell(tf.keras.layers.Layer):
    def __init__(self, hp, **kwargs):
        self.dtype          = hp['dtype']
        self.output_size    = hp['n_sensory'] # recurrent: same as inputsize
        self.state_size     = None # indicate after STSP

        self.tau            = hp['neuron_tau']
        self.dt             = hp['dt']
        self.alpha          = hp['dt']/hp['neuron_tau']
        self.stsp           = hp['neuron_stsp']
        self.activation     = hp['rnn_activation']

        self.noisetype      = hp['rnn_noise_type']

        self.add_noise      = Noise(hp,
                                    noisetype = self.noisetype,
                                    n_neurons = self.output_size)

        # todo: implement stsp with Dale's law
        if self.stsp:
            '''
            state_size[0] := hidden states
            state_size[1] := x; available neurotransmitter (Masse, 2019) 
            state_size[2] := u; neurotransmitter utilization  (Masse, 2019)
            '''
            self.state_size = [hp['n_sensory'],hp['n_sensory'],hp['n_sensory']]
        else:
            self.state_size = hp['n_sensory']  # todo: add STSP states to the cell

        super(RNNCell, self).__init__(**kwargs)

        self.build_rnnmat(hp)
        self.build()

    def build_rnnmat(self,hp):
        if hp['rnn_weights'] is 'sym_sensory': # symmetric in the sensory space
            rnnmat          = np.zeros((hp['n_sensory'], hp['n_sensory'])) # recurrent matrix
            tunings         = np.arange(0, np.pi, np.pi / hp['n_sensory'])
            for shift in range(hp['n_sensory']):
                rnnmat[:, shift] = hp['rnn_weights_scale'] + hp['rnn_weights_shift'] * \
                                   np.cos(2*tunings[shift] - tunings)

            self.rnnmat = tf.constant(rnnmat, name='rnnmat', dtype=self.dtype)
        elif hp['rnn_weights'] is 'sym_stimulus':
            rnnmat          = np.zeros((hp['n_sensory'], hp['n_sensory'])) # recurrent matrix
            for shift in range(hp['n_sensory']):
                rnnmat[:, shift] = hp['rnn_weights_scale'] + hp['rnn_weights_shift'] * \
                                   np.cos(2*(self.sensory_tunings[shift] - self.sensory_tunings))

            self.rnnmat = tf.constant(rnnmat, name='rnnmat', dtype=self.dtype)
        elif hp['rnn_weights'] is 'learn': # learn the best sensory neuron tunings
            rnnmat = np.random((hp['n_sensory'], hp['n_sensory']))
            self.rnnmat = tf.Variable(rnnmat, name='rnnmat', dtype=self.dtype)
            #self.rnnmat = self.add_weight(shape=(hp['n_sensory'], hp['n_sensory']),
            #                              initializer="uniform",
            #                              name='rnnmat') #todo: try add_weight?

    def build(self, input_shape):
        # no need to build other units. (we konw the input size, and nothing is dependent on the input size)
        self.built = True

    def call(self, inputs, states):
        """
        inputs: [B, output_size]
        states: list of states
        output: [output, states]
        """

        prev_h = states[0]
        # decay and the new change.
        h = (1-self.alpha)*prev_h + self.alpha * self.activation(tf.dot(prev_h, self.rnnmat) + inputs)

        if self.noisetype is not None:
            h = self.add_noise(h,inputs)

        return h, [h]

class Noise(tf.Module):
    def __init__(self, hp,
                 noisetype = 'Normal_poisson',
                 n_neurons = 1):
        self.noisetype      = noisetype
        self.n_neurons      = n_neurons # need if the noise is to be learned
        self.tau            = hp['neuron_tau']
        self.dt             = hp['dt']
        self.alpha          = hp['dt']/hp['neuron_tau']
        self.hp             = hp # also need loss

        self.noise_sd       = None # defined below

        if self.noisetype == 'Normal_fixed':
            # scale neural noise by time constants of the neuron.... todo: check if how the noise variance is calculated
            self.noise_sd = hp['noise_sd']

    def build(self, input_shape):
        # todo: when does this get built? when it is first called after init?
        # todo: how should I implement regularization
        if self.noisetype == 'Normal_learn':  # assume average of poisson neurons; scale by time constants
            self.noisenet = tf.keras.layers.Dense(
                self.n_neurons, input_shape=(input_shape[-1],),
                activation=tf.nn.relu,
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.hp['loss_L1'], l2=self.hp['loss_L2']),
                name='sensory_noiseStd')
        self.built = True

    def __call__(self,sens_act, neural_input = None):
        gauss_noise = tf.random.normal(sens_act.shape, 0,
                                       tf.sqrt(2 * self.hp['alpha_neuron']),
                                       dtype=self.dtype)

        if self.noisetype == 'Normal_fixed':
            self.noise_sd = self.noise_sd # fixed (defined at init)
        elif self.noisetype == 'Normal_learn':  # assume average of poisson neurons; scale by time constants
            self.noise_sd = self.noisenet(neural_input)
        elif self.noisetype == 'Normal_poisson':  # assume average of poisson neurons; scale by time constants
            self.noise_sd = tf.math.sqrt(sens_act)
        else:
            self.noise_sd = 1

        sens_m = sens_act + self.noise_sd * gauss_noise

        return sens_m

