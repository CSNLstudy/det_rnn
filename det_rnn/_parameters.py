import numpy as np
from ._functions import *

__all__ = ['par', 'update_parameters']

# All the relevant parameters ========================================================================
par = {
	# Experiment design: unit: second(s)
	'design':  {'iti'	: (0, 1.5),
			   	'stim'	: (1.5, 3.0),
			   	'delay'	: (3.0, 4.5),
			   	'estim'	: (4.5, 6.0)},
	'output_range' : ((4.5,6.0)), # None fills the whole period

	# Mask specs
	'dead': ((0,0.1),(4.4,4.5)),
	'mask': {'iti'	: 1., 'stim' : 1., 'delay'	: 1., 'estim' : 5.,
			 'rule_iti' : 2., 'rule_stim' : 2., 'rule_delay' : 2., 'rule_estim' : 10.},  # strength

	# Rule specs
	'input_rule'   : {'fixation' : (0,6.0),
					  'response' : (4.5,6.0)},
	'output_rule'  : {'fixation' : (0,4.5)},
	'input_rule_strength'	: 1,  # TODO(HG): check this
	'output_rule_strength' 	: 1,  # josh: why do we need rule strengths?

	# stimulus type
	'type'					: 'orientation',  # size, orientation

	# multiple trials
	'trial_per_subblock'	: 1,

	# stimulus encoding/response decoding type
	'stim_encoding'			: 'single',  # 'single', 'double'
	'resp_decoding'			: 'disc', 	# 'conti', 'disc'

	# Network configuration
	'exc_inh_prop'          : 0.8,    # excitatory/inhibitory ratio
	'modular'				: False,
	'connect_prob'			: 0.1,    # modular connectivity

	# Timings and rates
	'dt'                    : 10,     # unit: ms
	'learning_rate'         : 2e-2,	  # adam optimizer learning rate
	'membrane_time_constant': 100,    # tau

	# Input and noise
	'input_mean'            : 0.0,
	'noise_in_sd'           : 0.1,
	'noise_rnn_sd'          : 0.05,    # 0.5

	# Tuning function data
	'strength_input'        : 0.8,      # magnitutde scaling factor for von Mises
	'strength_output'       : 0.8,      # magnitutde scaling factor for von Mises
	'kappa'                 : 2,        # concentration scaling factor for von Mises

	# Loss parameters
	'spike_regularization'  : 'L2',      # 'L1' or 'L2'
	'spike_cost'            : 2e-5,
	'weight_cost'           : 0.,
	'clip_max_grad_val'     : 0.1,

	# Synaptic plasticity specs
	'masse'					: False,
	'tau_fast'              : 200,
	'tau_slow'              : 1500,
	'U_stf'                 : 0.15,
	'U_std'                 : 0.45,

	# Training specs
	'n_iterations'        : 300,
	'iters_between_outputs' : 100,

	# TODO(HG): clarify here
	# Neuronal settings
	'n_receptive_fields': 1,
	'n_tuned_input'	 : 24,  # number of possible orientation-tuned neurons (input)
	'n_tuned_output' : 24,  # number of possible orientation-tuned neurons (input)
	'n_ori'	 	 : 24 , # number of possible orientaitons (output)
	'noise_mean' : 0,
	'noise_sd'   : 0.005,     # 0.05
	'n_recall_tuned' : 24,   # precision at the moment of recall
	'n_hidden' 	 : 160,		 # number of rnn units

	# Experimental settings
	'batch_size' 	: 1024,      # if image, 128 recommended
	'alpha_neuron'  : 0.2,    # changed from tf.constant

	# Optimizer
	'optimizer' : 'Adam', # TODO(HG):  other optim. options?
	'loss_fun'	: 'mse', # 'cosine', 'mse', 'mse_normalize', 'centropy'
}


def update_parameters(par):
	# ranges and masks
	par.update({'design_rg': _convert_to_rg(par['design'], par['dt']),
				'output_rg': _convert_to_rg(par['output_range'], par['dt']),
				'dead_rg'  : _convert_to_rg(par['dead'], par['dt']),
				'input_rule_rg'	:  _convert_to_rg(par['input_rule'], par['dt']),
				'output_rule_rg': _convert_to_rg(par['output_rule'], par['dt']),
				'n_rule_input'	: len(par['input_rule']),
				'n_rule_output'	: len(par['output_rule'])})

	# TODO(HG) implement as a function(NOT ELEGANT)?
	## set n_input
	if par['stim_encoding'] == 'single':
		par['n_input'] = par['n_rule_input'] + par['n_tuned_input']

	elif par['stim_encoding'] == 'double':
		par['n_input'] = par['n_rule_input'] + par['n_tuned_input'] * 2

	## set n_estim_output
	if par['resp_decoding'] == 'conti':
		par['n_output'] = par['n_rule_output'] + 1

	elif par['resp_decoding'] == 'disc':
		par['n_output'] = par['n_rule_output'] + par['n_tuned_output']


	#
	par.update({
		'n_timesteps' 		: sum([len(v) for _,v in par['design_rg'].items()]),
		'n_exc'        		: int(par['n_hidden']*par['exc_inh_prop']),
	})

	#
	if par['modular']:
		par['EImodular_mask'] = _modular_mask(par['connect_prob'], par['n_hidden'], par['exc_inh_prop'])
	else:
		par['EImodular_mask'] = _w_rnn_mask(par['n_hidden'], par['exc_inh_prop'])

	par.update({
		'rg_exc': range(par['n_exc']),
		'rg_inh': range(par['n_exc'], par['n_hidden']),
		'w_in_mask': np.ones((par['n_input'], par['n_hidden']), dtype=np.float32),
		'w_rnn_mask': np.ones((par['n_hidden'], par['n_hidden']), dtype=np.float32) - np.eye(par['n_hidden'],dtype=np.float32),
		'w_out_mask': np.ones((par['n_hidden'], par['n_output']), dtype=np.float32)
	})

	# parameters
	par.update({
		'h0': _random_normal_abs((1, par['n_hidden'])),
		'w_in0': _random_normal_abs((par['n_input'], par['n_hidden'])),
		'w_rnn0': _random_normal_abs((par['n_hidden'], par['n_hidden'])),
		'b_rnn0': np.zeros(par['n_hidden'], dtype=np.float32),
		'w_out0': _random_normal_abs((par['n_hidden'],par['n_output'])) * par['w_out_mask'],
		'b_out0': np.zeros(par['n_output'], dtype=np.float32),

		'syn_x_init': np.ones((par['batch_size'], par['n_hidden']), dtype=np.float32),
		'syn_u_init': np.tile(_alternating((0.15, 0.45), par['n_hidden']), (par['batch_size'], 1)),
		'alpha_std': _alternating((0.05, 0.00667), par['n_hidden']),
		'alpha_stf': _alternating((0.00667, 0.05), par['n_hidden']),
		'dynamic_synapse': np.ones(par['n_hidden'], dtype=np.float32),
		'U': _alternating((0.15, 0.45), par['n_hidden']),
	})

	return par

par = update_parameters(par)


