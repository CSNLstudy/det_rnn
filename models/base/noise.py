import tensorflow as tf

class Noise(tf.Module):
    def __init__(self, hp,
                 noisetype = 'Normal_fixed',
                 n_neurons = 1):
        self.dtype          = hp['dtype']

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
                                       tf.sqrt(2 * self.alpha),
                                       dtype=self.dtype)

        if self.noisetype == 'Normal_fixed':
            self.noise_sd = self.noise_sd # fixed (defined at init)
        elif self.noisetype == 'Normal_learn':  # assume average of poisson neurons; scale by time constants
            self.noise_sd = self.noisenet(neural_input)
        elif self.noisetype == 'Normal_poisson':  # assume average of poisson neurons; scale by time constants
            self.noise_sd = tf.math.sqrt(tf.math.abs(sens_act)) # todo: make sure the activations are above 0.
        else:
            self.noise_sd = 1

        sens_m = sens_act + self.noise_sd * gauss_noise

        return sens_m
