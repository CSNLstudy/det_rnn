import tensorflow as tf
from .hyper import *
import numpy as np

EPSILON = 1e-7

__all__ = ['Model']

class Model(tf.Module):
    def __init__(self, hp=hp):
        super(Model, self).__init__()
        self.hp = hp
        self._initialize_variable(hp)
        self.n_rule_output = hp['n_rule_output']  #
        self.mse_weight = hp['mse_weight']
        self.optimizer = tf.optimizers.Adam(hp['learning_rate'])  # josh: do we want optimizer inside the model?

    def __call__(self, trial_info, hp):
        y, loss = self._train_oneiter(trial_info['neural_input'],
                                      trial_info['desired_output'],
                                      trial_info['mask'], hp)
        return y, loss

    def build(self):
        # build graph
        self.add_placeholders()
        self.m = self.sensory_layer(self.neural_input)
        self.logits = self.output_layer(self.m)

        # add update operator for target network
        self.add_update_target_op("q", "target_q")

        # add square loss
        self.add_loss_op(self.q, self.target_q)

        # add optmizer for the main networks
        self.add_optimizer_op("q")

    def add_placeholders(self):
        # input, rules, etc...
        self.neural_input   = tf.placeholder(tf.float32, shape=[None] + [self.hp['n_tuned_input']]) # batch size and input size
        self.neural_output  = tf.placeholder(tf.float32, shape=[None]) # batchsize, indicate which neuron is the output.

    def sensory_layer(self, neural_input):
        # population tuning (1 distribution for vonMises likelihood in the sensory space) neural network to define sensory inputs and outputs
        sens_u = tf.math.asin(tf.math.sin(tf.layers.dense(neural_input, 1, name='sensory_mu',
                                                          kernel_regularizer = tf.keras.regularizers.l1_l2(l1=self.hp['loss_L1'], l2=self.hp['loss_L2']))
                                          )) # convert range to -pi to pi
        sens_k = tf.layers.dense(neural_input, 1, activation=tf.nn.relu, name='sensory_k',
                                 kernel_regularizer = tf.keras.regularizers.l1_l2(l1=self.hp['loss_L1'], l2=self.hp['loss_L2'])
                                 ) # positive kappa

        # Von Mises; todo: check gradient?
        sens_p = tf.math.exp(sens_k * tf.math.cos(self.var_dict['sensory_tunings']- sens_u,
                                                  name='sensory_vm_cos'),
                             name='sensory_vm_exp')/\
                     (tf.constant(2.*np.pi)*tf.math.bessel_i0e(sens_k, name='sensory_vm_bessel'))

        if 'sensory_gain' is True:
            sens_g = tf.layers.dense(neural_input, 1, activation=tf.nn.relu, name='sensory_gain',
                                     kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.hp['loss_L1'],
                                                                                    l2=self.hp['loss_L2'])
                                     )  # positive gain
            sens_act = sens_g * sens_p  # todo (?): multiply a constant gain to all input units
        else:
            sens_act = sens_p

        # Sensory measurement. add noise todo: how do we properly scale noise? learn weights for noise, proportional to gain?
        if self.hp['sensory_noise_type'] == 'Normal': # return measurement, with normal noise
            sens_m = tf.random.normal(sens_act.shape, sens_act, tf.sqrt(2 * hp['alpha_neuron']) * hp['noise_rnn_sd'], dtype=tf.float32)
        elif self.hp['sensory_noise_type'] == 'Poisson': # return measurement, with poisson noise
            sens_m = tf.random.poisson(sens_act.shape, sens_act, dtype=tf.float32, name = 'sensory_noise')
            #todo: how does the graph work here; does the input change?

        return sens_m, sens_p, sens_act, sens_g

    def output_layer(self,sens_m):
        logits = tf.nn.relu(tf.layers.dense(sens_m, self.hp['n_tuned_output'],
                                            name='sensory_k',
                                            kernel_regularizer = tf.keras.regularizers.l1_l2(l1=self.hp['loss_L1'], l2=self.hp['loss_L2'])))
        return logits

    def loss(self, labels, logits):
        # main loss
        loss_ce = tf.nn.softmax_cross_entropy_with_logits(labels, logits, axis=-1, name='loss_CE')

        # info loss
        loss_MI = 0. # todo: is there a tractable way to implement this?

        # smoothness loss
        loss_smoothPost = tf.square(tf.norm(self.laplacian.matmul(logits),ord='euclidean'))

        loss = loss_ce + self.hp['loss_MI'] * loss_MI + self.hp['loss_p_smooth'] * loss_smoothPost

        return loss


    def optimize_op(self):
        optimizer = tf.train.AdamOptimizer(self.lr)
        qnet_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        grads = optimizer.compute_gradients(self.loss, var_list=qnet_w)
        grad_list = []
        for i, (grad, var) in enumerate(grads):
            if grad is not None:
                if self.config.grad_clip is True:
                    grads[i] = (tf.clip_by_norm(grad, self.config.clip_val), var)
                grad_list.append(grad)

        apply_grads = optimizer.apply_gradients(grads)
        self.train_op = apply_grads  # is this supposed to be a list sess.run(self.train_op)
        self.grad_norm = tf.global_norm(grad_list)

    def _initialize_variable(self, hp):
        # todo: change formats for all hp?
        _var_dict = {}
        for k, v in hp.items():
            if k[-1] == '0':  # make this more robust. how to add weights in tf?
                name = k[:-1]
                _var_dict[name] = tf.Variable(hp[k], name=name, dtype='float32')
        # todo: also save other variables?

        # bias for sensory tunings todo: initialize to something else (i.e. Wei and Stocker transformation)
        if self.hp['sensory_efficient']:

            cdf = #todo:
            _var_dict['sensory_tunings'] = tf.convert_to_tensor(np.linspace(-np.pi, np.pi,hp['n_tuned_input']),
                                                                name='sensory_tunings', dtype='float32')
        else:
            _var_dict['sensory_tunings'] = tf.Variable(np.linspace(-np.pi, np.pi,hp['n_tuned_input']),
                                                       name='sensory_tunings', dtype='float32')
        self.var_dict = _var_dict

        # generate laplacian kernel for loss
        laplac = np.zeros(self.hp['n_tuned_output'])
        laplac[0] = -0.25
        laplac[1] = 0.5
        laplac[2] = -0.25
        spectrum = tf.signal.fft(tf.cast(laplac, tf.complex64))
        self.laplacian = tf.linalg.LinearOperatorCirculant(spectrum, input_output_dtype=tf.float32)

