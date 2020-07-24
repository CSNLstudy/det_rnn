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

    def add_placeholders(self):
        # input, rules, etc...
        self.neural_input = tf.placeholder(tf.float32, shape=[None] + self.hp['n_input'])

    def sensory_layer(self, input_data, hp):
        # neural network to define sensory inputs and outputs
        sens_u = tf.layers.dense(self.neural_input, 1, name='sensory_mu')
        sens_k = tf.layers.dense(self.neural_input, 1, name='sensory_k')

        var_dict['sensory_tunings']

        return

    def optimize_op(self):
        optimizer = tf.train.AdamOptimizer(self.lr)
        qnet_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

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
        _var_dict['sensory_tunings'] = tf.Variable(np.ones(hp['n_tuned_input']), name='sensory_tunings', dtype='float32')
        _var_dict['output'] = tf.Variable(np.ones(hp['n_tuned_input']), name='output', dtype='float32')

        self.var_dict = _var_dict



    def _calc_loss(self, y, target, mask, hp):

        return loss