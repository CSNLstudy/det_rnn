import numpy as np
import tensorflow as tf
from det_rnn.base import par
from det_rnn.base.functions import random_normal_abs, alternating, modular_mask, w_rnn_mask

__all__ = ['hp', 'hp_spec']

# Model hyperparameters(modifiable)
hp  = {
	'masse' : True,
	'loss_fun'	: 1, # 0:'mse', 1:'centropy'
	'learning_rate' : 2e-2,	  # adam optimizer learning rate
	'dt'            : 10.,
	'clip_max_grad_val'  : 0.1,
	'alpha_neuron'  : 0.1,
	'spike_cost'  : 2e-4, # mse: 2e-5, centropy: 2e-3
	'weight_cost' : 0.,
	'dm_cost' : 0.1,
	'noise_rnn_sd': 0.5,

	'h0': random_normal_abs((1, par['n_hidden'])),
	'w_in0': random_normal_abs((par['n_input'], par['n_hidden'])),
	'w_rnn0': random_normal_abs((par['n_hidden'], par['n_hidden'])),
	'b_rnn0': np.zeros(par['n_hidden'], dtype=np.float32),
	'w_out0': random_normal_abs((par['n_hidden'],par['n_output'])) * par['w_out_mask'],
	'b_out0': np.zeros(par['n_output'], dtype=np.float32),

	'syn_x_init': np.ones((par['batch_size'], par['n_hidden']), dtype=np.float32),
	'syn_u_init': np.tile(alternating((0.15, 0.45), par['n_hidden']), (par['batch_size'], 1)),
	'alpha_std': alternating((0.05, 0.00667), par['n_hidden']),
	'alpha_stf': alternating((0.00667, 0.05), par['n_hidden']),
	'dynamic_synapse': np.ones(par['n_hidden'], dtype=np.float32),
	'U': alternating((0.15, 0.45), par['n_hidden']),

	'w_in_mask': np.ones((par['n_input'], par['n_hidden']), dtype=np.float32),
	'w_rnn_mask': np.ones((par['n_hidden'], par['n_hidden']), dtype=np.float32) - np.eye(par['n_hidden'],dtype=np.float32),
	'w_out_mask': np.concatenate((np.ones((par['n_exc'], par['n_output']),dtype=np.float32),
									  np.zeros((par['n_hidden']-par['n_exc'], par['n_output']), dtype=np.float32)),axis=0), # Todo(HL): no input from inhibitory neurons

}

if par['modular']:
	hp['EI_mask'] = modular_mask(par['connect_prob'], par['n_hidden'], par['exc_inh_prop'])
else:
	hp['EI_mask'] = w_rnn_mask(par['n_hidden'], par['exc_inh_prop'])

# Tensorize hp
for k, v in hp.items():
    hp[k] = tf.constant(v, name=k)

# hp_spec
hp_spec = {}
for k, v in hp.items():
	hp_spec[k] = tf.TensorSpec(v.numpy().shape, tf.dtypes.as_dtype(v.numpy().dtype), name=k)
