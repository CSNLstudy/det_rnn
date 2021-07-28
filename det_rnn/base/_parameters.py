import numpy as np
from .functions import convert_to_rg

__all__ = ['par', 'update_parameters']

# All the relevant parameters ========================================================================
par = {
	# Experiment design: unit: second(s)
	'design': {'iti' : (0, 1.5),
			   'stim': (1.5, 3.0),
			   'delay'   : ((3.0, 4.0),(5.5,6.0)),
			   'decision': (4.0, 5.5),
			   'estim'   : (6.0, 7.5)},
	'dm_output_range': 'design',  # decision period
	'em_output_range': 'design',  # estim period

	# Mask specs
	'dead': 'design',  # ((0,0.1),(estim_start, estim_start+0.1))
	'mask_dm': {'iti': 0., 'stim': 0., 'decision': 1., 'delay': 1., 'estim': 1.,
				'rule_iti': 0., 'rule_stim': 0., 'rule_decision': 2.,  'rule_delay': 1., 'rule_estim': 1.},  # strengthe
	# 'mask_em': {'iti': 1., 'stim': 1., 'decision': 1., 'delay': 1., 'estim': 200.,
	# 			'rule_iti': 2., 'rule_stim': 2., 'rule_decision': 2., 'rule_delay': 2., 'rule_estim': 400.},  # strength
	'mask_em': {'iti': 0., 'stim': 1., 'decision': 1., 'delay': 1., 'estim': 1.,
				'rule_iti': 0., 'rule_stim': 2., 'rule_decision': 2., 'rule_delay': 2., 'rule_estim': 1.},  # strength
	# 'mask_em': {'iti': 1., 'stim': 1., 'decision': 1., 'delay': 1., 'estim': 10.,
	# 			'rule_iti': 2., 'rule_stim': 2., 'rule_decision': 2., 'rule_delay': 2., 'rule_estim': 20.},  # strength


	## for onehot output, 'estim' : 20., 'rule_estim' : 20. worked

	# Rule specs
	'input_rule': 'design',  # {'fixation': whole period, 'response':estim}
	'output_dm_rule': 'design',  # {'fixation' : (0,before estim)}
	'output_em_rule': 'design',  # {'fixation' : (0,before estim)}
	'input_rule_strength'  : 0.8,
	'output_dm_rule_strength' : 0.8,
	'output_em_rule_strength' : 0.8,

	# Decision specs
	# 'reference': [-3, -2, -1, 1, 2, 3],
	'reference': [-4, -3, -2, -1, 1, 2, 3, 4],
	# 'reference': [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5],
	# 'reference': [-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10,11],
	'strength_ref': 1.,
	'strength_decision': 0.8,

	# Discrimination spec
	'n_discrim': 3,

	# stimulus type
	'type'					: 'orientation',  # size, orientation

	# stimulus distribution
	'stim_dist'				: 'uniform', # or a specific input
	'ref_dist'				: 'uniform', # or a specific input

	# multiple trials
	'trial_per_subblock'	: 1, 	  # should divide the batch_size

	# stimulus encoding/response decoding type
	'stim_encoding'			: 'single',  # 'single', 'double'
	'resp_decoding'			: 'disc',  # 'conti', 'disc', 'onehot'

	# Network configuration
	'exc_inh_prop'          : 0.8,    # excitatory/inhibitory ratio
	'modular'				: False,
	'connect_prob'			: 0.1,    # modular connectivity

	# Timings and rates
	'dt'                    : 20.,     # unit: ms
	'tau'   			    : 100,     # tau

	# Input and noise
	'input_mean'            : 0.0,
	'noise_in_sd'           : 0.1,
	'noise_rnn_sd'          : 0.5,    # HL: rnn_sd changed from 0.05 to 0.5 (Masse)

	# Tuning function data
	'strength_input'        : 0.8,      # magnitutde scaling factor for von Mises
	'strength_output'       : 0.8,      # magnitutde scaling factor for von Mises
	'kappa'                 : 2,        # concentration scaling factor for von Mises

	# Loss parameters
	'orientation_cost' 		: 1, # TODO(HL): cost for target-output

	# Training specs
	# 'n_iterations'        : 300,
	# 'iters_between_outputs' : 100,

	# Neuronal settings
	'n_receptive_fields': 1,
	'n_tuned_input'	 : 24,  # number of possible orientation-tuned neurons (input)
	'n_tuned_output' : 24,  # number of possible orientation-tuned neurons (input)
	'n_ori'	 	     : 24 , # number of possible orientaitons (output)
	'noise_mean'     : 0,
	'noise_sd'       : 0.005, # 0.05
	'n_recall_tuned' : 24,  # precision at the moment of recall
	'n_hidden1' 	 : 48,  
	'n_hidden2' 	 : 24,  
	'n_hiddenr' 	 : 24,  

	'noise_in_str1'  : 'design',  # population1
	'noise_in_str2'  : 'design',  # population2

	# Experimental settings
	'batch_size' 	: 128,
	# 'alpha_neuron'  : 0.1,    # changed from tf.constant TODO(HL): alpha changed from 0.2 to 0.1 (Masse)

	# Optimizer
	'optimizer' : 'Adam', # TODO(HG):  other optim. options?
}


def update_parameters(par):
	# ranges and masks
	par.update({'design_rg': convert_to_rg(par['design'], par['dt'])})

	#
	par.update({
		'n_timesteps' : sum([len(v) for _ ,v in par['design_rg'].items()]),
		'n_exc'       : int(par['n_hidden1' ] *par['exc_inh_prop']),
	})

	#
	par.update({
		'n_ref'       : len(par['reference']),
		'n_subblock'  : int(par['batch_size' ] /par['trial_per_subblock'])
	})

	# default settings
	if par['dm_output_range'] == 'design':
		par['dm_output_rg'] = convert_to_rg(par['design']['decision'], par['dt'])
		# A = par['design']['decision']
		# B = par['design']['delay']
		# par['dm_output_rg'] = convert_to_rg(np.concatenate((A,B[-1])), par['dt'])
	else:
		par['dm_output_rg'] = convert_to_rg(par['em_output_range'], par['dt'])

	if par['em_output_range'] == 'design':
		_stim     = convert_to_rg(par['design']['stim'], par['dt'])
		_decision = convert_to_rg(par['design']['decision'], par['dt'])
		_delay    = convert_to_rg(par['design']['delay'], par['dt'])
		_estim    = convert_to_rg(par['design']['estim'], par['dt'])
		em_output = np.concatenate((_stim,_decision,_delay,_estim))
		
		# par['em_output_rg'] = convert_to_rg(par['design']['estim'], par['dt'])
		par['em_output_rg'] = em_output
	else:
		par['em_output_rg'] = convert_to_rg(par['em_output_range'], par['dt'])

	# TODO(HG): this may not work if design['estim'] is 2-dimensional
	if par['dead'] == 'design':
		par['dead_rg'] = convert_to_rg(((0 ,0.1),
										(par['design']['estim'][0] ,par['design']['estim'][0 ] +0.1)) ,par['dt'])
	else:
		par['dead_rg'] = convert_to_rg(par['dead'], par['dt'])

	if par['input_rule'] == 'design':
		# par['input_rule_rg'] = convert_to_rg({'decision'  : par['design']['decision'],
		# 									  'estimation': par['design']['estim']}, par['dt'])
		par['input_rule_rg'] = convert_to_rg({'decision'  : par['design']['decision']}, par['dt'])
		par['n_rule_input']  = 0
	else:
		par['input_rule_rg']  = convert_to_rg(par['input_rule'], par['dt'])
		par['n_rule_input']   = len(par['input_rule'])

	## set n_input
	if par['stim_encoding'] == 'single':
		par['n_input'] = par['n_rule_input'] + par['n_tuned_input']

	elif par['stim_encoding'] == 'double':
		par['n_input'] = par['n_rule_input'] + par['n_tuned_input'] * 2

	## Decision-phase range
	if par['output_dm_rule'] == 'design':
		par['output_dm_rule_rg'] = convert_to_rg({'fixation'  : ((0, par['design']['decision'][0]),
																 (par['design']['decision'][1], par['design']['estim'][1]))}, par['dt'])
		par['n_rule_output_dm']  = 0
	else:
		par['output_dm_rule_rg'] = convert_to_rg(par['output_dm_rule'], par['dt'])
		par['n_rule_output_dm']  = len(par['output_dm_rule'])

	## Estimation-phase range
	if par['output_em_rule'] == 'design':
		par['output_em_rule_rg'] = convert_to_rg({'fixation'  : (0, par['design']['estim'][0])}, par['dt'])
		par['n_rule_output_em']  = 0
	else:
		par['output_em_rule_rg'] = convert_to_rg(par['output_em_rule'], par['dt'])
		par['n_rule_output_em']  = len(par['output_em_rule'])

	## set n_estim_output
	par['n_output_dm'] = par['n_rule_output_dm'] + 2
	if par['resp_decoding'] == 'conti':
		par['n_output_em'] = par['n_rule_output_em'] + 1
	elif par['resp_decoding'] in ['disc', 'onehot']:
		par['n_output_em'] = par['n_rule_output_em'] + par['n_tuned_output']

	## stimulus distribution TODO(HG): this is erroroneous
	if par['stim_dist'] == 'uniform':
		par['stim_p'] = np.ones(par['n_ori'])
	else:
		par['stim_p'] = par['stim_dist']
	par['stim_p'] = par['stim_p' ] /np.sum(par['stim_p'])

	if par['ref_dist'] == 'uniform':
		par['ref_p'] = np.ones(par['n_ref'])
	else:
		par['ref_p'] = par['ref_dist']
	par['ref_p'] = par['ref_p' ] /np.sum(par['ref_p'])

	## Additional noise vector dependent on task structure
	# TODO(HG): Here, it is assumed that the estimation terminates a trial
	if par['noise_in_str1'] == 'design':		
		par['noise_in_vec1'] = np.zeros(par['n_timesteps'])
	else:
		par['noise_in_vec1'] = par['noise_in_str1']

	if par['noise_in_str2'] == 'design':		
		par['noise_in_vec2'] = np.zeros(par['n_timesteps'])
	else:
		par['noise_in_vec2'] = par['noise_in_str2']


	##
	par.update({
		'rg_exc': range(par['n_exc']),
		'rg_inh': range(par['n_exc'], par['n_hidden1']),
		# 'w_in_mask': np.ones((par['n_input'], par['n_hidden']), dtype=np.float32),
		# 'w_rnn_mask': np.ones((par['n_hidden'], par['n_hidden']), dtype=np.float32) - np.eye(par['n_hidden']
		# 																					 ,dtype=np.float32),
		# 'w_out_dm_mask': np.concatenate((np.ones((par['n_exc'], par['n_output_dm']),dtype=np.float32),
		# 								 np.zeros((par['n_hidden']-par['n_exc'],par['n_output_dm']), dtype=np.float32)),axis=0),
		# 'w_out_em_mask': np.concatenate((np.ones((par['n_exc'], int(par['n_output_em'])), dtype=np.float32),
		# 								 np.zeros((par['n_hidden'] - par['n_exc'], par['n_output_em']),dtype=np.float32)), axis=0),
		# 'w_out_dm_mask': np.ones((par['n_hidden'], par['n_output_dm']), dtype=np.float32),
		# 'w_out_em_mask': np.ones((par['n_hidden'], par['n_output_em']), dtype=np.float32)
	})

	return par

par = update_parameters(par)


