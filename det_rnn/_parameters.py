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
	'output_range' : 'design', # estim period

	# Mask specs
	'dead': 'design', # ((0,0.1),(estim_start, estim_start+0.1))
	'mask': {'iti'	: 1., 'stim' : 1., 'delay'	: 1., 'estim' : 5.,
			 'rule_iti' : 2., 'rule_stim' : 2., 'rule_delay' : 2., 'rule_estim' : 10.},  # strength

	# Rule specs
	'input_rule' :  'design', # {'fixation': whole period, 'response':estim}
	'output_rule'  : 'design', # {'fixation' : (0,before estim)}
	'input_rule_strength'	: 0.8,  # TODO(HG): check this
	'output_rule_strength' 	: 0.8,

	# stimulus type
	'type'					: 'orientation',  # size, orientation

	# multiple trials
	'trial_per_subblock'	: 1,

	# stimulus encoding/response decoding type
	'stim_encoding'			: 'single',  # 'single', 'double'
	'resp_decoding'			: 'disc', 	# 'conti', 'disc'

	# Network configuration
	'exc_inh_prop'          				: 0.8,    	# excitatory/inhibitory ratio
	'modular'								: False,
	'connect_prob_within_module'			: 0.8,
	'connect_prob_adjacent_module_forward'	: 0.4,
	'connect_prob_distant_module_forward'	: 0.2,
	'connect_prob_adjacent_module_back'		: 0.4,
	'connect_prob_distant_module_back'		: 0.2,
	'scale_gamma' 							: 0.1, 		# w_rnn2in should have smaller scale than the other weights (=1)
	'recurrent_inhiddenout' 				: True,   	# in-hidden-out neurons are recurrently connected

	# Timings and rates
	'dt'                    : 10,     # unit: ms
	'learning_rate'         : 2e-2,	  # adam optimizer learning rate
	'membrane_time_constant': 100,    # tau

	# Input and noise
	'input_mean'            : 0.0,
	'noise_rnn_sd'          : 0.5,    # TODO(HL): rnn_sd changed from 0.05 to 0.5 (Masse)

	# Tuning function data
	'strength_input'        : 0.8,      # magnitutde scaling factor for von Mises
	'strength_output'       : 0.8,      # magnitutde scaling factor for von Mises
	'kappa'                 : 2,        # concentration scaling factor for von Mises

	# Loss parameters
	'spike_regularization'  : 'L2',      # 'L1' or 'L2'
	'spike_cost'            : 2e-5,
	'weight_cost'           : 0.,
	'clip_max_grad_val'     : 0.1,
	'orientation_cost' 		: 1, # TODO(HL): cost for target-output

	# Synaptic plasticity specs
	'masse'					: False,
	'tau_fast'              : 200,
	'tau_slow'              : 1500,
	'U_stf'                 : 0.15,
	'U_std'                 : 0.45,

	# Training specs
	'n_iterations'        : 300,
	'iters_between_outputs' : 100,

	# Neuronal settings
	'n_receptive_fields': 1,
	'n_tuned_input'	 	: 24,  # number of possible orientation-tuned neurons (input)
	'n_tuned_output' 	: 24,  # number of possible orientation-tuned neurons (input)
	'n_ori'	 	 		: 24 , # number of possible orientaitons (output)
	'noise_mean' 		: 0,
	'noise_sd'   		: 0.005,     # 0.05
	'n_recall_tuned' 	: 24,   # precision at the moment of recall
	'n_hidden' 	 		: 100,		 # number of rnn units TODO(HL): h_hidden to 100

	# Experimental settings
	'batch_size' 	: 1024, # if image, 128 recommended
	'alpha_input' 	: 0.7, 	# Chaudhuri et al., Neuron, 2015
	'alpha_hidden' 	: 0.2,
	'alpha_output' 	: 0.7,  # Chaudhuri et al., Neuron, 2015; Motor (F1) cortex has similar decay profile with sensory cortex
	'alpha_neuron'  : 0.2,

	# Optimizer
	'optimizer' : 'Adam', # TODO(HG):  other optim. options?
	'loss_fun'	: 'mse', # 'cosine', 'mse', 'mse_normalize', 'centropy'

}

def update_parameters(par):
	# ranges and masks
	par.update({'design_rg': _convert_to_rg(par['design'], par['dt'])})

	#
	par.update({
		'n_timesteps' 		: sum([len(v) for _,v in par['design_rg'].items()]),
		'n_exc'        		: int(par['n_hidden']*par['exc_inh_prop']),
	})

	# default settings
	if par['output_range'] is 'design':
		par['output_rg'] = _convert_to_rg(par['design']['estim'], par['dt'])
	else:
		par['output_rg'] = _convert_to_rg(par['output_range'], par['dt'])

	# TODO(HG): this may not work if design['estim'] is 2-dimensional
	if par['dead'] is 'design':
		par['dead_rg'] = _convert_to_rg(((0,0.1),
										 (par['design']['estim'][0],par['design']['estim'][0]+0.1)),par['dt'])
	else:
		par['dead_rg'] = _convert_to_rg(par['dead'], par['dt'])

	if par['input_rule'] is 'design':
		par['input_rule_rg'] = _convert_to_rg({'response': par['design']['estim']},par['dt'])
		par['n_rule_input']  = 1
	else:
		par['input_rule_rg'] = _convert_to_rg(par['input_rule'], par['dt'])
		par['n_rule_input']  = len(par['input_rule'])

	if par['output_rule'] is 'design':
		par['output_rule_rg'] = _convert_to_rg({'fixation':(0,par['design']['delay'][1])}, par['dt'])
		par['n_rule_output'] = 1
	else:
		par['output_rule_rg'] = _convert_to_rg(par['output_rule'], par['dt'])
		par['n_rule_output']  = len(par['output_rule'])

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
	par['EI_mask'] = _EI_mask(par['n_hidden'], par['exc_inh_prop'])
	par['modular_sparse_mask'] = _modular_sparse_mask(par['n_input'], par['n_hidden'], par['n_output'],
											   par['connect_prob_within_module'],
											   par['connect_prob_adjacent_module_forward'],
											   par['connect_prob_distant_module_forward'],
											   par['connect_prob_adjacent_module_back'],
											   par['connect_prob_distant_module_back'])

	par['input_silencing_mask'], par['hidden_silencing_mask'], par['out_silencing_mask'] \
		= _silencing_mask(par['n_input'], par['n_hidden'], par['n_output'])

	par['alpha_mask'] = _alpha_mask(par['n_input'], par['n_hidden'], par['n_output'],
									par['alpha_input'], par['alpha_hidden'], par['alpha_output'],
									par['batch_size'])
	par.update({
		'rg_exc': range(par['n_exc']),
		'rg_inh': range(par['n_exc'], par['n_hidden']),
		'w_in_mask': np.ones((par['n_input'], par['n_hidden']), dtype=np.float32),
		'w_rnn_mask': np.ones((par['n_hidden'], par['n_hidden']), dtype=np.float32) - np.eye(par['n_hidden'],dtype=np.float32),
		'w_out_mask': np.concatenate((np.ones((par['n_exc'], par['n_output']),dtype=np.float32),
									  np.zeros((par['n_hidden']-par['n_exc'], par['n_output']), dtype=np.float32)),
									 axis=0) # Todo(HL): no input from inhibitory neurons

	})

	# parameters
	par.update({

		'h0': _initialize((1, par['n_hidden']), par['scale_gamma']),
		'w_in0': _initialize((par['n_input'], par['n_hidden']), par['scale_gamma']),
		'w_rnn0': _initialize((par['n_hidden'], par['n_hidden']), par['scale_gamma']),
		'w_out0': _initialize((par['n_hidden'], par['n_output']), par['scale_gamma']),
		'b_rnn0': np.zeros(par['n_hidden'], dtype=np.float32),
		'b_out0': np.zeros(par['n_output'], dtype=np.float32),

		'syn_x_init': np.ones((par['batch_size'], par['n_hidden']), dtype=np.float32),
		'syn_u_init': np.tile(_alternating((0.15, 0.45), par['n_hidden']), (par['batch_size'], 1)),
		'alpha_std': _alternating((0.05, 0.00667), par['n_hidden']),
		'alpha_stf': _alternating((0.00667, 0.05), par['n_hidden']),
		'dynamic_synapse': np.ones(par['n_hidden'], dtype=np.float32),
		'U': _alternating((0.15, 0.45), par['n_hidden']),	})

	return par

par = update_parameters(par)
