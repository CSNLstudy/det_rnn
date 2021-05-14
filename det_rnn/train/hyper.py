import numpy as np
import tensorflow as tf
from det_rnn.base import par
from det_rnn.base.functions import initialize, alternating, modular_mask, w_rnn_mask

__all__ = ['hp', 'hp_spec', 'update_hp']

# Model hyperparameters(modifiable)
hp  = {
	'masse'       : False,
	'dale'        : False,
	'w_out_dm_fix': False,
	'DtoE_off'    : False,
	'EtoD_off'    : False,

	'gain'      : 1.,
	'loss_fun'	: 1, # 0:'mse', 1:'centropy'
	'task_type' : 0, # 0:'decision', 1:'estimation'
	'learning_rate' : 2e-2,	  # adam optimizer learning rate
	'dt'            : 20.,
	'clip_max_grad_val'  : 0.1,
	'spike_cost'  : 2e-7, # mse: 2e-5, centropy: 2e-3
	'weight_cost' : 0.,
	'noise_rnn_sd': 0.1,

	'lam_decision': 1.,
	'lam_estim'   : 1.,
    'tau_neuron'  : 100,
    'tau_std' : alternating((200, 1500), par['n_hidden1']+par['n_hidden2']),
    'tau_stf' : alternating((200, 1500), par['n_hidden1']+par['n_hidden2']),
	'w_out_dm': np.kron(np.eye(2), np.ones(int(par['n_hidden1']/2))).T.astype(np.float32),

	'syn_x_init': np.ones((par['batch_size'], par['n_hidden1']), dtype=np.float32),
	'syn_u_init': np.tile(alternating((0.15, 0.45), par['n_hidden1']), (par['batch_size'], 1)),
	'U': alternating((0.15, 0.45), par['n_hidden1'])
}

if par['modular']:
	hp['EI_mask'] = modular_mask(par['connect_prob'], par['n_hidden1'], par['exc_inh_prop'])
else:
	hp['EI_mask'] = w_rnn_mask(par['n_hidden1'], par['exc_inh_prop'])

def update_hp(hp):
	hp.update({
		'w_in10'    : initialize((par['n_input'],   par['n_hidden1']),   gain=hp['gain']),
		'w_in20'    : initialize((par['n_input'],   par['n_hidden2']),   gain=hp['gain']),
		'w_rnn110'  : initialize((par['n_hidden1'], par['n_hidden1']),   gain=hp['gain']),
		'w_rnn120'  : initialize((par['n_hidden1'], par['n_hidden2']),   gain=hp['gain']),
		'w_rnn210'  : initialize((par['n_hidden2'], par['n_hidden1']),   gain=hp['gain']),
		'w_rnn220'  : initialize((par['n_hidden2'], par['n_hidden2']),   gain=hp['gain']),
		'w_out_dm0' : initialize((par['n_hidden1'], par['n_output_dm']), gain=hp['gain']),
		'w_out_em0' : initialize((par['n_hidden2'], par['n_output_em']), gain=hp['gain'])
	})

	hp.update({
		'alpha_neuron1': np.float32(hp['dt']/hp['tau_neuron']),
		'alpha_neuron2': np.float32(hp['dt']/hp['tau_neuron']),
		'alpha_std': np.float32(hp['dt']/hp['tau_std']), #
		'alpha_stf': np.float32(hp['dt']/hp['tau_stf']) #
	})

	return hp

hp = update_hp(hp)

# Tensorize hp
for k, v in hp.items():
    hp[k] = tf.constant(v, name=k)	

# hp_spec: NEED TO BE CHANGED
hp_spec = {}
for k, v in hp.items():
	hp_spec[k] = tf.TensorSpec(v.numpy().shape, tf.dtypes.as_dtype(v.numpy().dtype), name=k)