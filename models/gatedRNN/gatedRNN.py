import copy, os, sys

import numpy as np
from scipy.interpolate import interp1d
import tensorflow as tf
from matplotlib import pyplot as plt

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
    def __init__(self, hp,
                 hypsearch_dict=None, logger=None):
        """
        hp: network hyperparameters
        """
        if hp['dtype'] == 'tf.float32':
            targetdtype= tf.float32

        super(gRNN, self).__init__(hp,
                                   hypsearch_dict=hypsearch_dict,
                                   dtype=targetdtype,
                                   logger=logger)

        self._initialize_variable()
        print('(gRNN) Network initialization complete')

    ''' UTILS '''
    def _initialize_variable(self):
        print('(gRNN) Building rnn...')

        # build layers
        self.rnncell = RNNCell(hp=self.hp)

        # build layers
        self.rnn    = tf.keras.layers.RNN(self.rnncell, return_state=True, return_sequences=True, name='RNN')
        self.dec    = tf.keras.layers.Dense(self.hp['n_output_dm'] - self.hp['n_rule_output_dm'],
                                            kernel_regularizer = tf.keras.regularizers.l1_l2(l1=self.hp['loss_L1'],
                                                                                             l2=self.hp['loss_L2']),
                                            name = 'decision'
                                            ) # should be 2

        # units
        self.est    = tf.keras.layers.Dense(self.hp['n_tuned_output'],
                                            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.hp['loss_L1'],
                                                                                           l2=self.hp['loss_L2']),
                                            name = 'estimation')

        if self.hp['gate_out']:
            self.dec_mdm    = tf.keras.layers.Dense(self.hp['n_output_dm'] - self.hp['n_rule_output_dm'],
                                                kernel_regularizer = tf.keras.regularizers.l1_l2(l1=self.hp['loss_L1'],
                                                                                                 l2=self.hp['loss_L2']),
                                                name = 'dec_mdm'
                                                ) # should be 2

            self.dec_mda    = tf.keras.layers.Dense(self.hp['n_output_dm'] - self.hp['n_rule_output_dm'],
                                                kernel_regularizer = tf.keras.regularizers.l1_l2(l1=self.hp['loss_L1'],
                                                                                                 l2=self.hp['loss_L2']),
                                                name = 'dec_mda'
                                                ) # should be 2

            self.est_mdm    = tf.keras.layers.Dense(self.hp['n_tuned_output'],
                                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.hp['loss_L1'],
                                                                                               l2=self.hp['loss_L2']),
                                                name = 'est_mdm')

            self.est_mda    = tf.keras.layers.Dense(self.hp['n_tuned_output'],
                                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.hp['loss_L1'],
                                                                                               l2=self.hp['loss_L2']),
                                                name = 'est_mda')

        ## generate laplacian kernel for smoothness loss
        laplacmat = np.zeros((self.hp['n_tuned_output'], self.hp['n_tuned_output']))
        laplacmat[0, 0] = -0.25
        laplacmat[1, 0] = 0.5
        laplacmat[2, 0] = -0.25
        for shift in range(self.hp['n_tuned_output']):
            laplacmat[:, shift] = np.roll(laplacmat[:, 0], shift=shift, axis=0)

        self.laplacian = tf.constant(laplacmat, dtype=self.dtype, name='laplacian')

        # scheduler for learning
        if self.hp['scheduler'] == 'scheduler_estimFirst':
            self.scheduler = scheduler_estimFirst(self.hp)
        else:
            self.scheduler = None

    @tf.function
    def evaluate(self, trial_info):
        # sensory_input   = self.sensory_layer(trial_info['neural_input'][:,:,2:]) # todo: take out rule more robustly

        neural_input = tf.transpose(trial_info['neural_input'][:, :, :],perm=[1, 0, 2])
        if self.hp['stsp']:
            #self.rnn.reset_states() # checked that the states are reset every time evaluate is called
            outputs, states1, states2, states3 = self.rnn(neural_input)
            states = [states1, states2, states3]
        else:
            outputs, states     = self.rnn(neural_input)

        # [B, T, N, 1] = rnn_output.shape (has a null dimension for matrix multiplication in the rnncell)
        out = tf.squeeze(outputs,axis=-1)

        if self.hp['dropout'] > 0:
            out_dm = tf.nn.dropout(out,self.hp['dropout'])
            out_em = tf.nn.dropout(out,self.hp['dropout'])
        else:
            out_dm = out
            out_em = out
            
        # todo: should I add nonlinearity to the dm and em modules?
        if self.hp['gate_out']:
            out_dm = (0+self.dec_mdm(neural_input[:,:,:self.hp['n_rule_input']])) * self.dec(out_dm) + \
                     self.dec_mda(neural_input[:,:,:self.hp['n_rule_input']])

            out_em = (0+self.est_mdm(neural_input[:,:,:self.hp['n_rule_input']])) * self.est(out_em) + \
                     self.est_mda(neural_input[:,:,:self.hp['n_rule_input']])
        else:
            out_dm = self.dec(out_dm)
            out_em = self.est(out_em)

        out_dm = tf.nn.sigmoid(out_dm)
        out_em = tf.nn.sigmoid(out_em)

        out         = tf.transpose(out,perm=[1, 0, 2]) # todo: check dimensions
        out_dm      = tf.transpose(out_dm,perm=[1, 0, 2])
        out_em      = tf.transpose(out_em,perm=[1, 0, 2])

        lossStruct = self.calc_loss(trial_info,
                                    {'out_rnn': out, 'out_dm': out_dm,'out_em': out_em})

        # should return
        # (1) a loss struct; should contain the final aggregated loss (unregularized); loss_struct['loss']
        # (2) output struct (logits)
        return lossStruct, {'rnn_output': out, 'rnn_states': states,
                            'dec_output': out_dm, 'est_output': out_em}

    @tf.function
    def calc_loss(self, trial_info, output):
        if self.scheduler is not None:
            train_params = self.scheduler.get_params()
        else:
            train_params = self.hp

        # inputs and outputs from the trial
        neural_input        = trial_info['input_tuning']
        dec_rule            = trial_info['neural_input'][:, :, 1]
        est_rule            = trial_info['neural_input'][:,:,2]
        est_desired_out     = trial_info['desired_estim'][:,:,self.hp['n_rule_output_em']:]
        dec_desired_out     = trial_info['desired_decision'][:, :, self.hp['n_rule_output_dm']:]
        output_tuning       = trial_info['output_tuning']

        # no training mask. Just calculate losses over the task period.
        #dec_mask = trial_info['mask_decision'][:, :, :self.hp['n_rule_output_dm']]
        #est_mask = trial_info['mask_estim'][:, : , :self.hp['n_rule_output_em']]

        rnn_output  = output['out_rnn'] # todo: check dimensions
        out_dm      = output['out_dm']
        out_em      = output['out_em']

        ## Estimation loss
        labels              = tf.math.argmax(output_tuning,1)
        [T,B,N]             = rnn_output.shape

        # todo: how should I interpret the neural outputs? logits? unnormalized probability?
        # out_em/tf.math.reduce_sum(out_em,axis=-1, keepdims=True)

        # MSE loss
        # average over time and neurons
        est_loss_mse = tf.reduce_mean(
            tf.gather(tf.math.square(est_desired_out - out_em), self.hp['input_rule_rg']['estimation'], axis=0))
        dec_loss_mse = tf.reduce_mean(
            tf.gather(tf.math.square(dec_desired_out - out_dm), self.hp['input_rule_rg']['decision'][50:], axis=0))

        # interpret desired outputs as unnormalized probabilities
        # nan when there is no output (0) => normalization
        est_target         = est_desired_out / tf.reduce_sum(est_desired_out + EPSILON, axis=2, keepdims=True)
        dec_target         = dec_desired_out / tf.reduce_sum(dec_desired_out + EPSILON, axis=2, keepdims=True)

        ## CE loss with probs
        ## interpret output probabilities as probabilities... (not logits for softmax)
        est_prob         = out_em / tf.reduce_sum(out_em + EPSILON, axis=2, keepdims=True)
        dec_prob         = out_dm / tf.reduce_sum(out_dm + EPSILON, axis=2, keepdims=True)

            # average over time then batch
            # log: cross entropy
        est_loss_ce = tf.reduce_mean(
            tf.reduce_mean(
            tf.reduce_sum(
            tf.gather(-est_target, self.hp['input_rule_rg']['estimation'], axis=0)*
            tf.gather(tf.math.log(est_prob + EPSILON), self.hp['input_rule_rg']['estimation'], axis=0),
            axis=-1)
                ,axis=0))
        dec_loss_ce = tf.reduce_mean(
            tf.reduce_mean(
            tf.reduce_sum(
            tf.gather(-dec_target,self.hp['input_rule_rg']['decision'][50:], axis=0) *
            tf.gather(tf.math.log(dec_prob + EPSILON), self.hp['input_rule_rg']['decision'][50:], axis=0),
                axis=-1)
                , axis=0))
        # gather 50 timepoints after = 500 ms

        # # loss with softmax
        # est_loss_ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        #     labels=tf.gather(-est_target, self.hp['input_rule_rg']['estimation'], axis=0),
        #     logits=tf.gather(out_em, self.hp['input_rule_rg']['estimation'], axis=0),
        #     axis=-1))
        # dec_loss_ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        #     labels=tf.gather(-dec_target,self.hp['input_rule_rg']['decision'], axis=0) ,
        #     logits=tf.gather(out_dm, self.hp['input_rule_rg']['decision'], axis=0),
        #     axis=-1))

        # regularization loss (implemented as keras module add_loss
        spike_loss = tf.reduce_mean(tf.math.square(rnn_output))

        l1_loss = tf.reduce_mean(tf.abs(self.rnncell.rnnmat)) + tf.reduce_mean(tf.abs(self.rnncell.Win))
        l2_loss = tf.reduce_mean(tf.math.square(self.rnncell.rnnmat)) + tf.reduce_mean(tf.math.square(self.rnncell.Win))
        l1_loss += tf.reduce_mean(tf.abs(self.rnncell.Wmd_rnn_a))
        l2_loss += tf.reduce_mean(tf.math.square(self.rnncell.Wmd_rnn_a))

        l1_loss += tf.reduce_sum([tf.reduce_mean(tf.abs(v)) for v in self.dec.trainable_variables
                                 if v.name.find('bias') <0])
        l1_loss += tf.reduce_sum([tf.reduce_mean(tf.abs(v)) for v in self.est.trainable_variables
                                 if v.name.find('bias') <0])
        l2_loss += tf.reduce_sum([tf.reduce_mean(tf.square(v)) for v in self.dec.trainable_variables
                                 if v.name.find('bias') <0])
        l2_loss += tf.reduce_sum([tf.reduce_mean(tf.square(v)) for v in self.est.trainable_variables
                                 if v.name.find('bias') <0])

        if self.hp['gate_rnn']:
            l1_loss += tf.reduce_mean(tf.abs(self.rnncell.Wmd_rnn_x))
            l2_loss += tf.reduce_mean(tf.math.square(self.rnncell.Wmd_rnn_x))

        if self.hp['gate_out']:
            l1_loss += tf.reduce_sum([tf.reduce_mean(tf.abs(v)) for v in self.dec_mdm.trainable_variables
                                      if v.name.find('bias') <0])
            l1_loss += tf.reduce_sum([tf.reduce_mean(tf.abs(v)) for v in self.est_mdm.trainable_variables
                                      if v.name.find('bias') <0])
            l2_loss += tf.reduce_sum([tf.reduce_mean(tf.square(v)) for v in self.dec_mdm.trainable_variables
                                      if v.name.find('bias') <0])
            l2_loss += tf.reduce_sum([tf.reduce_mean(tf.square(v)) for v in self.est_mdm.trainable_variables
                                      if v.name.find('bias') <0])
            l1_loss += tf.reduce_sum([tf.reduce_mean(tf.abs(v)) for v in self.dec_mda.trainable_variables
                                      if v.name.find('bias') <0])
            l1_loss += tf.reduce_sum([tf.reduce_mean(tf.abs(v)) for v in self.est_mda.trainable_variables
                                      if v.name.find('bias') <0])
            l2_loss += tf.reduce_sum([tf.reduce_mean(tf.square(v)) for v in self.dec_mda.trainable_variables
                                      if v.name.find('bias') <0])
            l2_loss += tf.reduce_sum([tf.reduce_mean(tf.square(v)) for v in self.est_mda.trainable_variables
                                      if v.name.find('bias') <0])

        # total loss
        # l1 and l2 loss should be accounted for automatically by keras.
        loss = train_params['loss_mse_dec'] * dec_loss_mse \
               + train_params['loss_mse_est'] * est_loss_mse \
               + train_params['loss_ce_dec'] * dec_loss_ce \
               + train_params['loss_ce_est'] * est_loss_ce \
               + train_params['loss_spike'] * spike_loss \
               + train_params['loss_L1'] * l1_loss \
               + train_params['loss_L2'] * l2_loss

        return {'loss'          : loss,
                'loss_mse_dec'  : dec_loss_mse,
                'loss_mse_est'  : est_loss_mse,
                'loss_ce_dec'   : dec_loss_ce,
                'loss_ce_est'   : est_loss_ce,
                'spike_loss'    : spike_loss,
                'loss_l1'       : l1_loss,
                'loss_l2'       : l2_loss}

    def get_hp(self):
        return self.hp

    def visualize_weights(self, rnnidx = None):
        fig = plt.figure(constrained_layout = True, figsize=(15, 15))
        ax1 = plt.subplot2grid((16, 16), (0, 0), rowspan=10, colspan=10);
        ax1.set_title("w_rnn")
        ax1.set_ylabel("H")
        ax2 = plt.subplot2grid((16, 16), (0, 10), rowspan=10, colspan=1);
        ax2.set_title("wmd_rnn_a")
        ax2.set_xlabel("Input rule")
        ax3 = plt.subplot2grid((16, 16), (0, 11), rowspan=10, colspan=5);
        ax3.set_title("w_input")
        ax3.set_xlabel("Input")
        ax4 = plt.subplot2grid((16, 16), (10, 0), rowspan=1, colspan=10);
        ax4.set_title("w_out_dec")
        ax4.set_ylabel("Dec")
        ax5 = plt.subplot2grid((16, 16), (11, 0), rowspan=5, colspan=10);
        ax5.set_title("w_out_est")
        ax5.set_ylabel("Est")
        ax5.set_xlabel("H")

        allmax = tf.reduce_max([tf.reduce_max(self.rnncell.rnnmat),
            tf.reduce_max(self.rnncell.Win),
            tf.reduce_max(self.rnncell.Wmd_rnn_a),
            tf.reduce_max(self.dec.trainable_variables[0]),
            tf.reduce_max(self.est.trainable_variables[0])])

        allmin = tf.reduce_min([tf.reduce_min(self.rnncell.rnnmat),
            tf.reduce_min(self.rnncell.Win),
            tf.reduce_min(self.rnncell.Wmd_rnn_a),
            tf.reduce_min(self.dec.trainable_variables[0]),
            tf.reduce_min(self.est.trainable_variables[0])])

        im1 = ax1.imshow(self.rnncell.rnnmat.numpy(),
                         interpolation='none',
                         aspect='auto')
        im2 = ax2.imshow(self.rnncell.Wmd_rnn_a.numpy(),
                         interpolation='none',
                         aspect='auto')
        im3 = ax3.imshow(self.rnncell.Win.numpy(),
                         interpolation='none',
                         aspect='auto')
        im4 = ax4.imshow(self.dec.trainable_variables[0].numpy().T,
                         interpolation='none',
                         aspect='auto')
        im5 = ax5.imshow(self.est.trainable_variables[0].numpy().T,
                         interpolation='none',
                         aspect='auto')

        #cbarax1 = fig.add_axes([1.10, 6 / 8, 0.01, 2 / 8])
        #cbarax2 = fig.add_axes([1.15, 6 / 8, 0.01, 2 / 8])
        #cbarax3 = fig.add_axes([1.10, 3 / 8, 0.01, 2 / 8])
        #cbarax4 = fig.add_axes([1.15, 3 / 8, 0.01, 2 / 8])
        #cbarax5 = fig.add_axes([1.20, 3 / 8, 0.01, 2 / 8])
        #cbarax5 = fig.add_axes([1.20, 6 / 8, 0.01, 7 / 8])

        #fig.colorbar(im1, cax=cbarax1)
        #fig.colorbar(im2, cax=cbarax2)
        #fig.colorbar(im3, cax=cbarax3)
        #fig.colorbar(im4, cax=cbarax4)
        #fig.colorbar(im5, cax=cbarax5)

        # plt.tight_layout()
        plt.show()

class scheduler_none():
    def __init__(self,hp):
        self.hp     = copy.deepcopy(hp)

    def get_parms(self):
        train_params = {
            'loss_mse_dec'  : self.hp['loss_mse_dec'],
            'loss_mse_est'  : self.hp['loss_mse_est'],
            'loss_ce_dec'   : self.hp['loss_ce_dec'],
            'loss_ce_est'   : self.hp['loss_ce_est'],
            'loss_spike'    : self.hp['loss_spike'],
            'loss_L1'       : self.hp['loss_L1'],
            'loss_L2'       : self.hp['loss_L2'],
            'dropout'       : self.hp['dropout'],
            'learning_rate' : self.hp['learning_rate'],
            'clip_max_grad_val' : self.hp['clip_max_grad_val']
        }

        return train_params

    def update(self,t,est_perf,dec_perf):
        # do nothing
        a = t+1


class scheduler_estimFirst():
    def __init__(self,hp):
        self.hp     = copy.deepcopy(hp)
        self.switch = False # switch to est and dec.
        self.decay = 0.99
        self.nbad = 0
        self.baseLR = hp['learning_rate']

    def get_params(self):
        train_params = {
            'loss_mse_dec'  : self.hp['loss_mse_dec'],
            'loss_mse_est'  : self.hp['loss_mse_est'],
            'loss_ce_dec'   : self.hp['loss_ce_dec'],
            'loss_ce_est'   : self.hp['loss_ce_est'],
            'loss_spike'    : self.hp['loss_spike'],
            'loss_L1'       : self.hp['loss_L1'],
            'loss_L2'       : self.hp['loss_L2'],
            'dropout'       : self.hp['dropout'],
            'learning_rate' : self.hp['learning_rate'],
            'clip_max_grad_val' : self.hp['clip_max_grad_val']
        }

        if not self.switch: # set decision weights to 0
            train_params['loss_mse_dec'] = 0
            train_params['loss_ce_dec'] = 0
            train_params['dropout'] = 0

        return train_params

    def update(self,t,est_perf,dec_perf):
        # learn estimation first, and makes sure est is good
        if self.switch == False:
            if est_perf > 0.95 : # estimation and decision mode
                self.switch = True
                self.nbad = 0
            if est_perf < 0.7:
                self.nbad += 1

        if self.switch == True:
            if est_perf < 0.8: # If estimation drops too low, switch back to estimation only
                self.switch = False
                self.nbad += 1

            if dec_perf < 0.6: # bad performance
                self.nbad += 1

        if est_perf > 0.75 and dec_perf > 0.75:  # try fine tuning, lower learning rate
            self.nbad = 0
            self.hp['learning_rate'] = self.hp['learning_rate'] * self.decay

        if self.nbad > 10: # try increasing learning rate
            # may be stuck at local minima.
            # note that ADAM optimizer by default normalizes learning by gradient norm.
            self.hp['learning_rate'] = self.hp['learning_rate'] / self.decay

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
        # zero argument needed to reference a tf variable in the loss.
        # not part of the network topology (can't be serialized
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer
        self.add_loss(lambda: self.hp['loss_L1']* tf.reduce_mean(tf.abs(self.rnnmat)))
        self.add_loss(lambda: self.hp['loss_L1'] * tf.reduce_mean(tf.abs(self.Win)))
        self.add_loss(lambda: self.hp['loss_L2'] * tf.reduce_mean(tf.math.square(self.rnnmat)))
        self.add_loss(lambda: self.hp['loss_L2'] * tf.reduce_mean(tf.math.square(self.Win)))

        self.add_loss(lambda: self.hp['loss_L1'] * tf.reduce_mean(tf.abs(self.Wmd_rnn_a)))
        self.add_loss(lambda: self.hp['loss_L2'] * tf.reduce_mean(tf.math.square(self.Wmd_rnn_a)))

        if self.hp['gate_rnn']:
            self.add_loss(lambda: self.hp['loss_L1'] * tf.reduce_mean(tf.abs(self.Wmd_rnn_x)))
            self.add_loss(lambda: self.hp['loss_L2'] * tf.reduce_mean(tf.math.square(self.Wmd_rnn_x)))

        # stsp parameters
        if self.hp['stsp'] is True:
            self.alpha_std_mat = tf.cast(alternating(self.hp['alpha_std'], self.hp['n_hidden']), self.dtype)
            self.alpha_stf_mat = tf.cast(alternating(self.hp['alpha_stf'], self.hp['n_hidden']), self.dtype)
            self.U_mat         = tf.cast(alternating(self.hp['U'], self.hp['n_hidden']),self.dtype)

    def build_weights(self):
        # network weights
        self.rnnmat     = tf.Variable(tf.random.normal([self.hp['n_hidden'], self.hp['n_hidden']]), name='rnnmat',
                                      dtype=self.dtype)
        self.Win        = tf.Variable(tf.random.normal([self.hp['n_hidden'], self.hp['n_tuned_input']]), name='Win',
                                      dtype=self.dtype)

        self.Wmd_rnn_a = tf.Variable(tf.random.normal([self.hp['n_hidden'], self.hp['n_rule_input']]),
                                     name='Wmd_rnn_a', dtype=self.dtype)

        # gating weights
        if self.hp['gate_rnn']:
            self.Wmd_rnn_x = tf.Variable(tf.random.normal([self.hp['n_hidden'], self.hp['n_rule_input']]),
                                         name='Wmd_rnn_x',dtype=self.dtype)

        self.EI_mask    = tf.cast(w_rnn_mask(self.hp['n_hidden'], self.hp['exc_inh_prop']).transpose(), self.dtype) #
        # build EI mask for the recurrent weights
        if self.hp['tau_train']:
            self.tau_unnorm = tf.Variable(tf.random.normal([self.hp['n_hidden']]), name='tau_unnorm', dtype=self.dtype)
        else:
            self.tau_unnorm = tf.random.normal([self.hp['n_hidden']])

    @tf.function
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        states = [tf.zeros((batch_size,self.hp['n_hidden'],1),dtype=self.dtype)]
        if self.hp['stsp']:
            states += [tf.ones((batch_size, self.hp['n_hidden'],1))]
            states += [tf.tile(tf.expand_dims(tf.expand_dims(self.U_mat, axis=0),axis=-1),
                               (batch_size,1,1))]

        return states

    @tf.function
    def __call__(self, inputs, states):
        """
        inputs: [B, n_sensory + n_MD]
        states: list of states
        output: [output, states]
        """
        prev_h = states[0]
        [B,N,null] = states[0].shape
        # decay and the new change.
        # tf.math.reduce_any(tf.math.is_nan(h))

        if self.hp['stsp']:
            syn_x = states[1]
            syn_u = states[2]

            # broadcast these (to batch size)
            alpha_std_mat = tf.expand_dims(tf.expand_dims(self.alpha_std_mat, axis=0),axis=-1)
            U_mat = tf.expand_dims(tf.expand_dims(self.U_mat, axis=0), axis=-1)
            alpha_stf_mat = tf.expand_dims(tf.expand_dims(self.alpha_stf_mat, axis=0),axis=-1)

            syn_x += alpha_std_mat * (1. - syn_x) - self.hp['dt'] / 1000 * syn_u * syn_x * prev_h
            syn_u += alpha_stf_mat * (U_mat - syn_u) + self.hp['dt'] / 1000 * U_mat * (1. -syn_u) * prev_h

            syn_x = tf.minimum(tf.constant(1.), tf.nn.relu(syn_x))
            syn_u = tf.minimum(tf.constant(1.), tf.nn.relu(syn_x))
            h_post = syn_u * syn_x * prev_h
        else:
            h_post = prev_h

        # with gating.
        # weight matrix parametrization
        if self.hp['dale'] is True:
            wrnn = tf.nn.relu(self.rnnmat) * self.EI_mask
        else:
           wrnn =  self.rnnmat

        # inputs: B x n
        inputmat = tf.expand_dims(inputs, axis=-1)

        # transfer function
        if self.hp['act'] == 'sigmoid':
            h_post = tf.sigmoid(h_post)
        elif self.hp['act'] == 'relu':
            h_post = tf.nn.relu(h_post)

        # decay
        tau = tf.sigmoid(self.tau_unnorm) * (self.hp['tau_max'] - self.hp['tau_min']) + self.hp['tau_min'] # normalize tau.
        alpha = self.hp['dt'] / tau # Nhidden x 1

        # update step
        dh_pre = wrnn @ h_post + self.Win @ inputmat[:,self.hp['n_rule_input']:,:]
        dh_pre += self.Wmd_rnn_a @ inputmat[:, :self.hp['n_rule_input'], :]

        if self.hp['gate_rnn']:
            gate = (0 + self.Wmd_rnn_x @ inputmat[:,:self.hp['n_rule_input'],:]) # gated by rules
        else:
            gate = 1

        h = (1-tf.expand_dims(alpha,axis=-1))*prev_h + tf.expand_dims(alpha,axis=-1) * (gate * dh_pre)

        # inject noise
        if self.hp['rnn_noise_type'] is not None and dh_pre.shape[0] is not None:
            h = h + self.hp['noise_sd'] * tf.random.normal(tf.shape(dh_pre),
                                                           mean=0,
                                                           stddev = tf.expand_dims(tf.sqrt(2 * alpha),axis=-1),
                                                           dtype = self.dtype)

        # tf.math.reduce_any(tf.math.is_nan(h))
        if self.hp['stsp']:
            return h, [h, syn_x, syn_u]
        else:
            return h, [h]

    # build for rnncell
    def build(self, hp, input_shape = None):
        print('(gRNN) Building rnncell...')
        self.hp = hp
        # self.add_noise      = Noise(hp,
        #                            noisetype = self.hp['rnn_noise_type'],
        #                            n_neurons = self.n_hidden)

        # stsp with Dale's law
        # state_size and output_size is used by keras
        if self.hp['stsp']:
            '''
            state_size[0] := hidden states
            state_size[1] := x; available neurotransmitter (Masse, 2019) 
            state_size[2] := u; neurotransmitter utilization  (Masse, 2019)
            '''
            self.state_size = [self.hp['n_hidden'], self.hp['n_hidden'], self.hp['n_hidden']]
        else:
            self.state_size = self.hp['n_hidden']  # todo: add STSP states to the cell
        self.output_size = self.hp['n_hidden']  # state_size and output_size is used by keras

    # for savedmodels
    # dictionary of important parameters
    def get_config(self):
        config = {}
        # config = super(RNNCell,self).get_config()
        # config['rnnmat'] = self.rnnmat
        # config['Win'] = self.Win
        # config['Wmd_rnn_x'] = self.Wmd_rnn_x
        # config['Wmd_rnn_a'] = self.Wmd_rnn_a
        # config['EI_mask'] = self.EI_mask
        # config['alpha_std'] = self.alpha_std
        # config['alpha_stf'] = self.alpha_stf
        # config['U'] = self.U
        #

        config['tau_min'] = self.hp['tau_min']
        config['tau_max'] = self.hp['tau_max']
        config['dt'] = self.hp['dt']

        config['dale'] = self.hp['dale']
        config['stsp'] = self.hp['stsp']
        config['exc_inh_prop'] = self.hp['exc_inh_prop']
        config['alpha_std'] = self.hp['alpha_std']
        config['alpha_stf'] = self.hp['alpha_stf']
        config['U'] = self.hp['U']

        config['activation'] = self.hp['act']

#        config['rnn_weights'] = self.rnn_weights
        config['n_hidden'] = self.hp['n_hidden']
        config['n_tuned_input'] = self.hp['n_tuned_input']
        config['n_rule_input'] = self.hp['n_rule_input']

        config['gate_rnn'] = self.hp['gate_rnn']
        config['gate_rnn'] = self.hp['gate_out']

        config['rnn_noise_type'] = self.hp['rnn_noise_type']
        config['noise_sd'] = self.hp['noise_sd']

        config['state_size'] = self.state_size
        config['output_size'] = self.output_size

        config['loss_spike'] = self.hp['loss_spike']
        config['loss_L1'] = self.hp['loss_L1']
        config['loss_L2'] = self.hp['loss_L2']

        return config

#    def from_config(cls, config):
#        return cls(config['hp'])
