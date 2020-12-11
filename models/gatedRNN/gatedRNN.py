import os, sys

import numpy as np
from scipy.interpolate import interp1d
import tensorflow as tf

sys.path.append('../')
from models.base.base_model import BaseModel
from models.base.noise import Noise
import det_rnn.train as utils_train
import det_rnn.analysis as utils_analysis
from det_rnn.base.functions import random_normal_abs, alternating, modular_mask, w_rnn_mask


EPSILON = 1e-7

__all__ = ['gRNN']

"""Gated recurrent neural network performing decision-making task and estimation task."""

class gRNN(BaseModel):
    def __init__(self, hp, par_train,
                 hypsearch_dict=None, logger=None):
        """
        hp: network hyperparameters
        par_train: stimulus parameters used to train the network (since efficient coding is dependent on stimulus distribution)
        """
        if hp['dtype'] == 'tf.float32':
            targetdtype= tf.float32

        super(gRNN, self).__init__(hp,
                                   par_train,
                                   hypsearch_dict=hypsearch_dict,
                                   dtype=targetdtype,
                                   logger=logger)

        self._initialize_variable(hp, par_train)
        #self.build()

        ''' UTILS '''

    def _initialize_variable(self, hp, par_train):
        # todo: need to change formats for all hp?
        # todo: also save other variables?
        # todo: same as build()??

        self.loss_spike     = hp['loss_spike']
        self.loss_L1        = hp['loss_L1']
        self.loss_L2        = hp['loss_L2']

        # build layers
        self.rnncell = RNNCell(hp=hp)
        self.rnn    = tf.keras.layers.RNN(self.rnncell, return_state=True, return_sequences=True, name='RNN')
        self.dec    = tf.keras.layers.Dense(par_train['n_output_dm'] - par_train['n_rule_output_dm'],
                                            kernel_regularizer = tf.keras.regularizers.l1_l2(l1=hp['loss_L1'],
                                                                                             l2=hp['loss_L2']),
                                            name = 'decision'
                                            ) # should be 2

        # units
        self.est    = tf.keras.layers.Dense(par_train['n_tuned_output'],
                                            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=hp['loss_L1'],
                                                                                           l2=hp['loss_L2']),
                                            name = 'estimation')

        self.out_gate = hp['out_gate']

        if self.out_gate:
            self.dec_mdm    = tf.keras.layers.Dense(par_train['n_output_dm'] - par_train['n_rule_output_dm'],
                                                kernel_regularizer = tf.keras.regularizers.l1_l2(l1=hp['loss_L1'],
                                                                                                 l2=hp['loss_L2']),
                                                name = 'dec_mdm'
                                                ) # should be 2

            self.dec_mda    = tf.keras.layers.Dense(par_train['n_output_dm'] - par_train['n_rule_output_dm'],
                                                kernel_regularizer = tf.keras.regularizers.l1_l2(l1=hp['loss_L1'],
                                                                                                 l2=hp['loss_L2']),
                                                name = 'dec_mda'
                                                ) # should be 2

            self.est_mdm    = tf.keras.layers.Dense(par_train['n_tuned_output'],
                                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=hp['loss_L1'],
                                                                                               l2=hp['loss_L2']),
                                                name = 'est_mdm')

            self.est_mda    = tf.keras.layers.Dense(par_train['n_tuned_output'],
                                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=hp['loss_L1'],
                                                                                               l2=hp['loss_L2']),
                                                name = 'est_mda')


        # save indices for convenience
        self.n_rule_input       = par_train['n_rule_input']
        self.n_rule_output_dm   = par_train['n_rule_output_dm']
        self.n_rule_output_em   = par_train['n_rule_output_em']
        self.input_rule_rg      = par_train['input_rule_rg']

        # todo: make a training mask
        # self.mask =

        ## generate laplacian kernel for smoothness loss
        laplacmat = np.zeros((hp['n_tuned_output'], hp['n_tuned_output']))
        laplacmat[0, 0] = -0.25
        laplacmat[1, 0] = 0.5
        laplacmat[2, 0] = -0.25
        for shift in range(hp['n_tuned_output']):
            laplacmat[:, shift] = np.roll(laplacmat[:, 0], shift=shift, axis=0)

        self.laplacian = tf.constant(laplacmat, dtype=self.dtype, name='laplacian')

    # tf.function
    def evaluate(self, trial_info):
        # sensory_input   = self.sensory_layer(trial_info['neural_input'][:,:,2:]) # todo: take out rule more robustly

        neural_input = tf.transpose(trial_info['neural_input'][:, :, :],perm=[1, 0, 2])
        if self.hp['neuron_stsp']:
            #self.rnn.reset_states() # checked that the states are reset every time evaluate is called
            outputs, states1, states2, states3 = self.rnn(neural_input)
            states = [states1, states2, states3]
        else:
            outputs, states     = self.rnn(neural_input)

        # [B, T, N, 1] = rnn_output.shape (has a null dimension for matrix multiplication in the rnncell)
        out = tf.squeeze(outputs,axis=-1)

        # todo: should I add nonlinearity to the dm and em modules?
        if self.out_gate:
            out_dm = (1+self.dec_mdm(neural_input[:,:,:self.n_rule_input])) * self.dec(out) + \
                     self.dec_mda(neural_input[:,:,:self.n_rule_input])

            out_em = (1+self.est_mdm(neural_input[:,:,:self.n_rule_input])) * self.est(out) + \
                     self.est_mda(neural_input[:,:,:self.n_rule_input])
        else:
            out_dm = self.dec(out)
            out_em = self.est(out)

        out_dm = tf.nn.relu(out_dm)
        out_em = tf.nn.relu(out_em)

        out         = tf.transpose(out,perm=[1, 0, 2]) # todo: check dimensions
        out_dm      = tf.transpose(out_dm,perm=[1, 0, 2])
        out_em      = tf.transpose(out_em,perm=[1, 0, 2])

        lossStruct = self.calc_loss(trial_info, {'out_rnn': out, 'out_dm': out_dm,'out_em': out_em})

        # should return
        # (1) a loss struct; should contain the final aggregated loss (unregularized); loss_struct['loss']
        # (2) output struct (logits)
        return lossStruct, {'rnn_output': out, 'rnn_states': states,
                            'dec_output': out_dm, 'est_output': out_em}

    # tf.function
    def calc_loss(self, trial_info, output):
        # inputs and outputs from the trial
        neural_input        = trial_info['input_tuning']
        dec_rule            = trial_info['neural_input'][:, :, 1]
        est_rule            = trial_info['neural_input'][:,:,2]
        est_desired_out     = trial_info['desired_estim'][:,:,self.n_rule_output_em:]
        dec_desired_out     = trial_info['desired_decision'][:, :, self.n_rule_output_dm:]
        output_tuning       = trial_info['output_tuning']

        rnn_output  = output['out_rnn'] # todo: check dimensions
        out_dm      = output['out_dm']
        out_em      = output['out_em']

        ## Estimation loss
        labels              = tf.math.argmax(output_tuning,1)
        [T,B,N]             = rnn_output.shape

        # todo: how should I interpret the neural outputs? logits? unnormalized probability?
        # out_em/tf.math.reduce_sum(out_em,axis=-1, keepdims=True)

        # MSE loss
        est_loss_mse = tf.reduce_mean(tf.math.square(est_desired_out - out_em))
        dec_loss_mse = tf.reduce_mean(tf.math.square(dec_desired_out - out_dm))

        # interpret output probabilities as logits
        est_prob           = tf.nn.softmax(out_em)
        dec_prob           = tf.nn.softmax(out_em)

        # interpret desired outputs as unnormalized probabilities
        # nan when there is no output (0) => normalization
        est_target         = est_desired_out / tf.reduce_sum(est_desired_out + EPSILON, axis=2, keepdims=True)
        dec_target         = dec_desired_out / tf.reduce_sum(dec_desired_out + EPSILON, axis=2, keepdims=True)

        # todo: put on masks?
        # average over time then batch
        est_loss_ce = tf.reduce_mean(
            tf.reduce_mean(
            tf.reduce_sum(
            tf.gather(-est_target, self.input_rule_rg['estimation'], axis=0)*
            tf.gather(tf.nn.log_softmax(out_em), self.input_rule_rg['estimation'], axis=0),
            axis=-1)
                ,axis=0))
        dec_loss_ce = tf.reduce_mean(
            tf.reduce_mean(
            tf.reduce_sum(
            tf.gather(-dec_target,self.input_rule_rg['decision'], axis=0) *
            tf.gather(tf.nn.log_softmax(out_dm), self.input_rule_rg['decision'], axis=0),axis=-1)
                , axis=0))

        # regularization loss (implemented as keras module add_loss
        spike_loss = tf.reduce_mean(tf.math.square(rnn_output))

        # total loss
        loss = self.hp['loss_mse_dec'] * dec_loss_mse \
               + self.hp['loss_mse_est'] * est_loss_mse \
               + self.hp['loss_ce_dec'] * dec_loss_ce \
               + self.hp['loss_ce_est'] * est_loss_ce \
               + self.loss_spike * spike_loss

        return {'loss'          : loss,
                'loss_mse_dec'  : dec_loss_mse,
                'loss_mse_est'  : est_loss_mse,
                'loss_ce_dec'   : dec_loss_ce,
                'loss_ce_est'   : est_loss_ce,
                'spike_loss'    : spike_loss}

############################ Submodules ############################
class RNNCell(tf.keras.layers.Layer):
    def __init__(self, hp=None, **kwargs):
        if hp is not None:
            self.build(hp)
            if hp['dtype'] == 'tf.float32':
                todtype = tf.float32
            super(RNNCell, self).__init__(dtype = todtype, name='RNNCell')
            # no need to build other units. (we konw the input size, and nothing is dependent on the input size)
            self.build_weights()
        else:
            # todo: loading from saved_model (fix this)
            # does not take attributes
            super(RNNCell, self).__init__(dtype=tf.float32, name='RNNCell') # todo change dtype
            for key, value in kwargs.items():
                setattr(self, key, value)

        self.build_rnn()
        self.built = True

    def build_rnn(self):
        # regularization losses
        self.add_loss(lambda: self.loss_L1 * tf.reduce_mean(tf.abs(self.rnnmat)))
        self.add_loss(lambda: self.loss_L1 * tf.reduce_mean(tf.abs(self.Win)))
        self.add_loss(lambda: self.loss_L1 * tf.reduce_mean(tf.abs(self.Wmd_x)))
        self.add_loss(lambda: self.loss_L1 * tf.reduce_mean(tf.abs(self.Wmd_a)))

        self.add_loss(lambda: self.loss_L2 * tf.reduce_mean(tf.math.square(self.rnnmat)))
        self.add_loss(lambda: self.loss_L2 * tf.reduce_mean(tf.math.square(self.Win)))
        self.add_loss(lambda: self.loss_L2 * tf.reduce_mean(tf.math.square(self.Wmd_x)))
        self.add_loss(lambda: self.loss_L2 * tf.reduce_mean(tf.math.square(self.Wmd_a)))

        # stsp parameters
        self.alpha_std_mat = tf.cast(alternating(self.alpha_std, self.n_hidden), self.dtype)
        self.alpha_stf_mat = tf.cast(alternating(self.alpha_stf, self.n_hidden), self.dtype)
        self.U_mat         = tf.cast(alternating(self.U, self.n_hidden),self.dtype)

    def build_weights(self):
        rnnmat  = np.random.random((self.n_hidden, self.n_hidden)) # positive weights
        Win     = np.random.randn(self.n_hidden,self.n_tuned_input)
        Wmd_x   = np.random.randn(self.n_hidden,self.n_rule_input)
        Wmd_a   = np.random.randn(self.n_hidden,self.n_rule_input)
        EI_mask = w_rnn_mask(self.n_hidden, self.exc_inh_prop)

        # network weights
        self.rnnmat     = tf.Variable(rnnmat, name='rnnmat', dtype=self.dtype)
        self.Win        = tf.Variable(Win, name='Win', dtype=self.dtype)
        self.Wmd_x      = tf.Variable(Wmd_x, name='Wmd_x', dtype=self.dtype)
        self.Wmd_a      = tf.Variable(Wmd_a, name='Wmd_a', dtype=self.dtype)
        self.EI_mask    = tf.cast(EI_mask.transpose(), self.dtype) # build EI mask for the recurrent weights

    def build(self, hp, input_shape = None):
        self.tau = hp['neuron_tau']
        self.dt = hp['dt']
        self.alpha = hp['dt'] / hp['neuron_tau']
        # self.activation = hp['rnn_activation']

        self.stsp = hp['neuron_stsp']
        self.exc_inh_prop = hp['exc_inh_prop']

        # reinitialized later:
        self.alpha_std = hp['alpha_std']
        self.alpha_stf = hp['alpha_stf']
        self.U = hp['U']

        # network parameters
        self.rnn_weights = hp['rnn_weights']
        self.n_hidden = hp['n_hidden']
        self.n_tuned_input = hp['n_tuned_input']
        self.n_rule_input = hp['n_rule_input']

        # noise type and method for adding noise.
        self.noisetype = hp['rnn_noise_type']
        self.noise_sd = hp['noise_sd']
        # self.add_noise      = Noise(hp,
        #                            noisetype = self.noisetype,
        #                            n_neurons = self.n_hidden)

        # stsp with Dale's law
        # state_size and output_size is used by keras
        if self.stsp:
            '''
            state_size[0] := hidden states
            state_size[1] := x; available neurotransmitter (Masse, 2019) 
            state_size[2] := u; neurotransmitter utilization  (Masse, 2019)
            '''
            self.state_size = [self.n_hidden, self.n_hidden, self.n_hidden]
        else:
            self.state_size = self.n_hidden  # todo: add STSP states to the cell
        self.output_size = self.n_hidden  # state_size and output_size is used by keras

        self.loss_spike = hp['loss_spike']
        self.loss_L1 = hp['loss_L1']
        self.loss_L2 = hp['loss_L2']

    #@tf.function
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        states = [tf.zeros((batch_size,self.n_hidden,1),dtype=self.dtype)]
        if self.stsp:
            states += [tf.ones((batch_size, self.n_hidden,1))]
            states += [tf.tile(tf.expand_dims(tf.expand_dims(self.U_mat, axis=0),axis=-1),
                               (batch_size,1,1))]

        return states

    #@tf.function
    def call(self, inputs, states):
        """
        inputs: [B, n_sensory + n_MD]
        states: list of states
        output: [output, states]
        """
        prev_h = states[0]
        [B,n,null] = states[0].shape
        # decay and the new change.
        # tf.math.reduce_any(tf.math.is_nan(h))

        if self.stsp:
            syn_x = states[1]
            syn_u = states[2]

            # broadcast these (to batch size)
            alpha_std_mat = tf.expand_dims(tf.expand_dims(self.alpha_std_mat, axis=0),axis=-1)
            U_mat = tf.expand_dims(tf.expand_dims(self.U_mat, axis=0), axis=-1)
            alpha_stf_mat = tf.expand_dims(tf.expand_dims(self.alpha_stf_mat, axis=0),axis=-1)

            syn_x += alpha_std_mat * (1. - syn_x) - self.dt / 1000 * syn_u * syn_x * prev_h
            syn_u += alpha_stf_mat * (U_mat - syn_u) + self.dt / 1000 * U_mat * (1. -syn_u) * prev_h

            syn_x = tf.minimum(tf.constant(1.), tf.nn.relu(syn_x))
            syn_u = tf.minimum(tf.constant(1.), tf.nn.relu(syn_x))
            h_post = syn_u * syn_x * prev_h
        else:
            h_post = prev_h

        # with gating.
        # h > 0; todo: check if h>0 for stsp
        wrnn = tf.nn.relu(self.rnnmat) * self.EI_mask

        # inputs: B x n
        inputmat = tf.expand_dims(inputs, axis=-1)
        dh_pre = (1+tf.matmul(self.Wmd_x, inputmat[:,:self.n_rule_input,:]))*tf.matmul(wrnn,h_post) + \
                 tf.matmul(self.Win, inputmat[:,self.n_rule_input:,:]) + \
                 tf.matmul(self.Wmd_a, inputmat[:, :self.n_rule_input,:])

        if self.noisetype is not None and dh_pre.shape[0] is not None:
            dh_pre = dh_pre + \
                     self.noise_sd * tf.random.normal(tf.shape(dh_pre), 0, tf.sqrt(2 * self.alpha),dtype=self.dtype)
        h = (1-self.alpha)*prev_h + self.alpha * tf.nn.relu(tf.nn.tanh(dh_pre))
        # using just relu => h blows up over time
        # h > 0  if prev_h > 0

        # todo: check Masse implementation.
        #        h_post = tf.nn.relu(tf.cast(_h, tf.float32) * (1. - hp['alpha_neuron'])
        #                            + hp['alpha_neuron'] * (tf.cast(rnn_input, tf.float32) @ tf.nn.relu(self.var_dict['w_in'])
        #                                                    + tf.cast(_h_post, tf.float32) @ _w_rnn + self.var_dict['b_rnn'])
        #                            + tf.random.normal(_h.shape, 0, tf.sqrt(2 * hp['alpha_neuron']) * hp['noise_rnn_sd'],
        #                                               dtype=tf.float32))
        # tf.math.reduce_any(tf.math.is_nan(h))
        if self.stsp:
            return h, [h, syn_x, syn_u]
        else:
            return h, [h]

    # for savedmodels
    def get_config(self):
        config = {}
        # config = super(RNNCell,self).get_config()
        # config['rnnmat'] = self.rnnmat
        # config['Win'] = self.Win
        # config['Wmd_x'] = self.Wmd_x
        # config['Wmd_a'] = self.Wmd_a
        # config['EI_mask'] = self.EI_mask
        # config['alpha_std'] = self.alpha_std
        # config['alpha_stf'] = self.alpha_stf
        # config['U'] = self.U
        #
        config['tau'] = self.tau
        config['dt'] = self.dt
        config['alpha'] = self.alpha

        config['stsp'] = self.stsp
        config['exc_inh_prop'] = self.exc_inh_prop

        config['rnn_weights'] = self.rnn_weights
        config['n_hidden'] = self.n_hidden
        config['n_tuned_input'] = self.n_tuned_input
        config['n_rule_input'] = self.n_rule_input

        config['noisetype'] = self.noisetype
        config['noise_sd'] = self.noise_sd

        config['alpha_std'] = self.alpha_std
        config['alpha_stf'] = self.alpha_stf
        config['U'] = self.U

        config['state_size'] = self.state_size
        config['output_size'] = self.output_size

        config['loss_spike'] = self.loss_spike
        config['loss_L1'] = self.loss_L1
        config['loss_L2'] = self.loss_L2

        return config
        # return self.__dict__

#    def from_config(cls, config):
#        return cls(config['hp'])
