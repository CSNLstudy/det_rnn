#%%
from Stimulus_JS import Stimulus
from hyper import make_par
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#%%

# nS = 8
# n_batch = 40
# n_hidden = 30

# Stim = Stimulus(nS = nS, nBatch = n_batch)
# input1 = Stim.generate_stimulus()
# output1 = Stim.generate_output()

# n_input = input1.shape[-1]
# n_output = output1.shape[-1]
# hp = make_par(n_input, n_hidden, n_output, n_batch, exc_inh_prop=0.8)


#%%
class Model(tf.Module):

    def __init__(self, hp):
        super(Model, self).__init__()
        self._initialize_variable(hp)
        self.optimizer = tf.optimizers.Adam(hp['learning_rate'])
        self.loss_mask = tf.ones(1, dtype = tf.float32)
    
    @tf.function
    def __call__(self, input_data, output_data,  hp):
        y, loss = self.train_onestep(input_data, output_data, hp)
        return y, loss

    def _initialize_variable(self, hp):
        _var_dict = {}
        for k, v in hp.items():
            if k[-1] == '0':
                name = k[:-1]
                print('{} initialized'.format(k[:-1]))
                x = tf.abs(tf.random.normal(shape =hp[k].shape, stddev=tf.cast(hp['initialize_std'], tf.float32)))
                if k[:-1]+'_mask' in list(hp.keys()):
                    x *= hp[k[:-1]+'_mask']
                    # print(k[:-1]+'_mask')
                
                _var_dict[name] = tf.Variable(x, name=name, dtype='float32')

        self.var_dict = _var_dict

    @tf.function
    def _rnn_cell(self, _h, rnn_input, _syn_x, _syn_u, hp):

        _w_rnn = tf.nn.relu(self.var_dict['w_rnn']) * tf.cast(hp['EI_mask'], tf.float32)
        if hp['masse']:
            _syn_x += hp['alpha_std'] * (1. - _syn_x) - hp['dt']/1000 * _syn_u * _syn_x * _h
            _syn_u += hp['alpha_stf'] * (hp['U'] - _syn_u) + hp['dt']/1000 * hp['U'] * (1. - _syn_u) * _h
            _syn_x = tf.minimum(tf.constant(1.), tf.nn.relu(_syn_x))
            _syn_u = tf.minimum(tf.constant(1.), tf.nn.relu(_syn_u))
            _h_post = _syn_u * _syn_x * _h
        else:
            _h_post = _h

        _h = _h * (1. - hp['alpha_neuron']) + \
                        tf.nn.relu(hp['alpha_neuron'] * (tf.cast(rnn_input, tf.float32) @ tf.nn.relu(self.var_dict['w_in']) + \
                            tf.cast(_h_post, tf.float32) @ _w_rnn + self.var_dict['b_rnn']) + \
                            tf.random.normal(_h.shape, 0, tf.sqrt(2*hp['alpha_neuron'])*hp['noise_rnn_sd'], dtype=tf.float32))

        _h = tf.minimum(_h, tf.constant(10000, dtype = tf.float32)) # maximum rate constraint

        return _h, _syn_x, _syn_u

    @tf.function
    def rnn_model(self, input_data, hp):
        _syn_x = hp['syn_x_init']
        _syn_u = hp['syn_u_init']
        _h = tf.cast(tf.tile(self.var_dict['h'], (_syn_x.shape[0], 1)), tf.float32)

        sz = input_data.shape[0]

        h_stack = tf.TensorArray(tf.float32, size=sz)
        syn_x_stack = tf.TensorArray(tf.float32, size=sz)
        syn_u_stack = tf.TensorArray(tf.float32, size=sz)
        i = 0
        for i in tf.range(sz):
            _h, _syn_x, _syn_u = self._rnn_cell(_h, input_data[i], _syn_x, _syn_u, hp)
            h_stack = h_stack.write(i, _h)
            syn_x_stack = syn_x_stack.write(i, _syn_x)
            syn_u_stack = syn_u_stack.write(i, _syn_u)

        h_stack = h_stack.stack()
        syn_x_stack = syn_x_stack.stack()
        syn_u_stack = syn_u_stack.stack()
        y_stack = (h_stack @ self.var_dict['w_out']) + self.var_dict['b_out']

        return y_stack, h_stack

    @tf.function
    def train_onestep(self, input_data, output_data, hp):
        # print('tracing')
        # tf.print('calculating')
        with tf.GradientTape() as t:
            y, h_stack = self.rnn_model(input_data, hp)
            sft = tf.nn.softmax_cross_entropy_with_logits(labels = output_data, logits = y, axis = -1)
            perf_loss = tf.reduce_mean(sft*self.loss_mask)
            spike_loss = tf.reduce_mean(h_stack**2)*hp['spike_cost']
            loss = perf_loss + spike_loss
        
        vars_and_grads = t.gradient(loss, self.var_dict)
        capped_gvs = [] # gradient capping and clipping
        for var, grad in vars_and_grads.items():
            if 'w_rnn' in var:
                grad *= hp['w_rnn_mask']
            elif 'w_out' in var:
                grad *= hp['w_out']
            elif 'w_in' in var:
                grad *= hp['w_in_mask']

            if grad is None:
                capped_gvs.append((grad, self.var_dict[var]))
            else:
                capped_gvs.append((tf.clip_by_norm(grad, hp['clip_max_grad_val']), self.var_dict[var]))
        self.optimizer.apply_gradients(grads_and_vars=capped_gvs)

        return y , {'perf_loss' : perf_loss, 'spike_loss' : spike_loss, 'loss': loss}
        

# #%%
# self  = Model(hp)
# plt.imshow(self.var_dict['w_out'].numpy())
# #%%

# Stim = Stimulus(nS = nS, nBatch = n_batch)
# Stim.steps_input2 = 0

# input1 = Stim.generate_stimulus()
# input_data = tf.constant(input1, dtype = tf.float32)
# output1 = Stim.generate_output()
# output_data = tf.constant(output1, dtype = tf.float32)

# plt.imshow(input1[:,1,:].transpose())

#%%
