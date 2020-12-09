import os, sys

import numpy as np
import tensorflow as tf

EPSILON = 1e-7

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
                                  [B, T, N, 1])

        out_tuning      = self.gain * (tf.ones([B,T,N,1],dtype=self.dtype) -
                                       self.ani + self.ani* tf.math.cos(2*(pref_dirs_in-pref_dirs_out)))

        # weigh the tunings by the inputs and then take the sum.
        out = tf.reduce_sum(tf.tile(tf.reshape(inputs,[B,T,N,1]),
                                    [1,1,1,self.layer_size]) * out_tuning,axis=3)
        #tf.reshape(inputs, [B, T, N, 1])

        return out # shape = B x T x n_layer_size