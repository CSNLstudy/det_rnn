import tensorflow as tf
import tensorflow_probability as tfp
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
        self.neural_output  = tf.placeholder(tf.float32, shape=[None] + [self.hp['n_tuned_output']])

    def sensory_layer(self, neural_input):
        # population tuning (1 distribution for vonMises likelihood in the sensory space) neural network to define sensory inputs and outputs
        sens_u = tf.math.asin(tf.math.sin(tf.layers.dense(neural_input, 1, name='sensory_mu'))) # convert range to -pi to pi
        sens_k = tf.layers.dense(neural_input, 1, activation=tf.nn.relu, name='sensory_k') # positive kappa
        sens_g = tf.layers.dense(neural_input, 1, activation=tf.nn.relu, name='sensory_gain')  # positive kappa

        # create a von Mises distribution
        sens_p = tf.math.exp(sens_k * tf.math.cos(self.var_dict['sensory_tunings']- sens_u,
                                                  name='sensory_vm_cos'),
                             name='sensory_vm_exp')/\
                     (tf.constant(2.*np.pi)*tf.math.bessel_i0e(sens_k, name='sensory_vm_bessel'))
        sens_act = sens_p
        # sens_act = sens_g * sens_p # todo (?): multiply a constant gain to all input units

        if self.hp['sensory_noise_type'] == 'Normal': # return measurement, with normal noise
            sens_m = tf.random.normal(sens_act.shape, sens_act, tf.sqrt(2 * hp['alpha_neuron']) * hp['noise_rnn_sd'], dtype=tf.float32)
        elif self.hp['sensory_noise_type'] == 'Poisson': # return measurement, with poisson noise
            sens_m = tf.random.poisson(sens_act.shape, sens_act, dtype=tf.float32, name = 'sensory_noise') # todo: how does the graph work here; does the input change?

        return sens_m

    def output_layer(self,sens_m):
        logits = tf.nn.relu(tf.layers.dense(sens_m, self.hp['n_tuned_output'], name='sensory_k'))
        return logits

    def loss(self,labels, logits):
        tf.nn.softmax_cross_entropy_with_logits(labels, logits, axis=-1, name='loss_CE')

    def optimize_op(self):
        optimizer = tf.train.AdamOptimizer(self.lr)
        qnet_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)im

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


    def model(self, input_data, hp):

            return y_stack, h_stack, syn_x_stack, syn_u_stack

    def return_losses(self, input_data, target_data, mask, hp):

        return _Y, {'loss': loss, 'perf_loss': perf_loss, 'spike_loss': spike_loss}


    def _train_oneiter(self, input_data, target_data, mask, hp):
        with tf.GradientTape() as t:

        self.optimizer.apply_gradients(grads_and_vars=capped_gvs)
        return _Y, {'loss': loss, 'perf_loss': perf_loss, 'spike_loss': spike_loss}


    def _initialize_variable(self, hp):
        _var_dict = {}
        for k, v in hp.items():
            if k[-1] == '0':  # make this more robust. how to add weights in tf?
                name = k[:-1]
                _var_dict[name] = tf.Variable(hp[k], name=name, dtype='float32')
        # todo: also save other variables?

        # bias for sensory tunings
        _var_dict['sensory_tunings'] = tf.Variable(np.linspace(-np.pi, np.pi,hp['n_tuned_input']), name='sensory_tunings', dtype='float32')

        self.var_dict = _var_dict