import numpy as np
from .functions import convert_to_rg

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
	'input_rule_strength'	: 0.8, 
	'output_rule_strength' 	: 0.8,

	# stimulus type
	'type'					: 'orientation',  # size, orientation

	# stimulus distribution
	'stim_dist'				: 'uniform', # or a specific input

	# multiple trials
	'trial_per_subblock'	: 1, 	  # should divide the batch_size

	# stimulus encoding/response decoding type
	'stim_encoding'			: 'single',  # 'single', 'double'
	'resp_decoding'			: 'disc', 	 # 'conti', 'disc'

	# Network configuration
	'exc_inh_prop'          : 0.8,    # excitatory/inhibitory ratio
	'modular'				: False,
	'connect_prob'			: 0.1,    # modular connectivity

	# Timings and rates
	'dt'                    : 10.,     # unit: ms
	'membrane_time_constant': 100,     # tau

	# Input and noise
	'input_mean'            : 0.0,
	'noise_in_sd'           : 0.1,
	'noise_rnn_sd'          : 0.5,    # TODO(HL): rnn_sd changed from 0.05 to 0.5 (Masse)

	# Tuning function data
	'strength_input'        : 0.8,      # magnitutde scaling factor for von Mises
	'strength_output'       : 0.8,      # magnitutde scaling factor for von Mises
	'kappa'                 : 2,        # concentration scaling factor for von Mises

	# Loss parameters
	'orientation_cost' 		: 1, # TODO(HL): cost for target-output

	# Training specs
	'n_iterations'        : 300,
	'iters_between_outputs' : 100,

	# Neuronal settings
	'n_receptive_fields': 1,
	'n_tuned_input'	 : 24,  # number of possible orientation-tuned neurons (input)
	'n_tuned_output' : 24,  # number of possible orientation-tuned neurons (input)
	'n_ori'	 	 : 24 , # number of possible orientaitons (output)
	'noise_mean' : 0,
	'noise_sd'   : 0.005,     # 0.05
	'n_recall_tuned' : 24,   # precision at the moment of recall
	'n_hidden' 	 : 100,		 # number of rnn units TODO(HL): h_hidden to 100

	# Experimental settings
	'batch_size' 	: 128,    
	'alpha_neuron'  : 0.1,    # changed from tf.constant TODO(HL): alpha changed from 0.2 to 0.1 (Masse)

	# Optimizer
	'optimizer' : 'Adam', # TODO(HG):  other optim. options?
	'loss_fun'	: 'mse_sigmoid', # 'cosine', 'mse', 'mse_normalize', 'centropy'
}


def update_parameters(par):
	# ranges and masks
	par.update({'design_rg': convert_to_rg(par['design'], par['dt'])})

	#
	par.update({
		'n_timesteps' 		: sum([len(v) for _,v in par['design_rg'].items()]),
		'n_exc'        		: int(par['n_hidden']*par['exc_inh_prop']),
	})

	# 
	par.update({
		'n_subblock' : int(par['batch_size']/par['trial_per_subblock'])
	})
	

	# default settings
	if par['output_range'] is 'design':
		par['output_rg'] = convert_to_rg(par['design']['estim'], par['dt'])
	else:
		par['output_rg'] = convert_to_rg(par['output_range'], par['dt'])

	# TODO(HG): this may not work if design['estim'] is 2-dimensional
	if par['dead'] is 'design':
		par['dead_rg'] = convert_to_rg(((0,0.1),
										 (par['design']['estim'][0],par['design']['estim'][0]+0.1)),par['dt'])
	else:
		par['dead_rg'] = convert_to_rg(par['dead'], par['dt'])

	if par['input_rule'] is 'design':
		par['input_rule_rg'] = convert_to_rg({'fixation': (0,par['design']['estim'][1]),
											   'response': par['design']['estim']},par['dt'])
		par['n_rule_input']  = 2
	else:
		par['input_rule_rg'] = convert_to_rg(par['input_rule'], par['dt'])
		par['n_rule_input']  = len(par['input_rule'])

	if par['output_rule'] is 'design':
		par['output_rule_rg'] = convert_to_rg({'fixation':(0,par['design']['delay'][1])}, par['dt'])
		par['n_rule_output'] = 1
	else:
		par['output_rule_rg'] = convert_to_rg(par['output_rule'], par['dt'])
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

	## stimulus distribution TODO(HG): this is erroroneous
	if par['stim_dist'] == 'uniform':
		par['stim_p'] = np.ones(par['n_ori'])
	else:
		par['stim_p'] = par['stim_dist']

	## stimulus normalization
	par['stim_p'] = par['stim_p']/np.sum(par['stim_p'])

	par.update({
		'rg_exc': range(par['n_exc']),
		'rg_inh': range(par['n_exc'], par['n_hidden']),
		'w_in_mask': np.ones((par['n_input'], par['n_hidden']), dtype=np.float32),
		'w_rnn_mask': np.ones((par['n_hidden'], par['n_hidden']), dtype=np.float32) - np.eye(par['n_hidden'],dtype=np.float32),
		'w_out_mask': np.concatenate((np.ones((par['n_exc'], par['n_output']),dtype=np.float32),
									  np.zeros((par['n_hidden']-par['n_exc'], par['n_output']), dtype=np.float32)),
									 axis=0) # Todo(HL): no input from inhibitory neurons
	})

	return par

par = update_parameters(par)


