#todo: get rid of hp not needed?

__all__ = ['vnet_hp']

def vnet_hp(par = None):
	if par is None:
		from det_rnn.base import par

	# Model hyperparameters(modifiable)
	vnet_hp  = {
		# saving parameters
		'output_path'	: '/experiments/efficient_coding/model1/',
		'model_output'	: '/experiments/efficient_coding/model1/model',
		'log_path' 		: '/experiments/efficient_coding/model1/logs',
		'saving_freq' 	: 100, # save every saving_freq iterations

		# losses and training
		'loss_spike' 	: 2e-3, #'spike_cost'  : 2e-3, # todo: not used yet
		'loss_L1' 		: 1e-2, # weight regularization
		'loss_L2' 		: 1e-2, # weight regularization
		'loss_MI' 		: 1, # maximize mutual information
		'loss_p_smooth' : 1e-2, # laplace filter on the posterior
		'loss_pe' 		: 1e-1, # absolute prediction error

		'nsteps_train'  	: 500,
		'learning_rate' 	: 2e-2,	  # adam optimizer learning rate
		'clip_max_grad_val' : 0.1,

		# neuron
		'sensory_gain'		: True,
		'sensory_noise_type': 'Normal_poisson',
			# 'Normal_fixed': fix to neural noise_sd below
			# 'Normal_learn': learn noise parameters
			# 'Normal_poisson': Assumes that responses is an average of poisson-activated neurons
		'sensory_repr'		: 'Efficient', # 'Uniform', 'Learn' or 'Efficient'
			# Efficient: make the prior flat in the sensory space;
			# Uniform: tuning of sensory neurons is uniform in the stimulus space
			# learn: learn the best sensory neuron tunings

		'dt'			: 10.,  # ms

		# neuron, decay and noise
		'alpha_neuron'  : 0.1,   # time constant = dt/alpha ~ i.e. 10/0.1 = 100ms
		'neuron_tau' 	: 100,  #'alpha_neuron'  : 0.1, # time constant = dt/alpha ~ i.e. 10/0.1 = 100ms
		'noise_sd'  	: 0.5, # should we learn this?

		# neuron, stsp
		'neuron_stsp' 	: False, # parameter

		'''
		#'syn_x_init'	: np.ones((par['batch_size'], par['n_hidden']), dtype=np.float32), # do we need to initialize this here?
		#'syn_u_init'	: np.tile(alternating((0.15, 0.45), par['n_hidden']), (par['batch_size'], 1)), # josh: why alternating?
	
		#'alpha_std'		: alternating((0.05, 0.00667), par['n_hidden']), # efficacy time constant #josh: what is alternating??
		#'alpha_stf'		: alternating((0.00667, 0.05), par['n_hidden']), # utilization time constant
	
		#'dynamic_synapse'	: np.ones(par['n_hidden'], dtype=np.float32),
		#'U'					: alternating((0.15, 0.45), par['n_hidden']),
		'''
	
		# network; inherited from par (stimulus structures)
		'n_input'			: par['n_input'],
		'n_tuned_input'		: par['n_tuned_input'],
		'n_sensory'			: 30, #number of neurons in the sensory layer
		'n_hidden'			: par['n_hidden'],
		'n_tuned_output'	: par['n_tuned_output'],

		'n_rule_input'		: par['n_rule_input'],
		'n_rule_output'		: par['n_rule_output']}

	return vnet_hp