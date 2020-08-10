import tensorflow as tf

__all__ = ['rvnet_hp']

def rvnet_hp(par = None):
    # Model hyperparameters(modifiable, as needed)

    if par is None:
        from det_rnn.base import par

    rvnet_hp = {
        # saving parameters
        'output_path'       : '/experiments/rvnet/model1/',
        'model_output'      : '/experiments/rvnet/model1/model',
        'log_path' 	        : '/experiments/rvnet/model1/logs',
        'saving_freq' 	    : 100, # save every saving_freq iterations
        'dtype'             : tf.float32,

        ##########################################################################################
        # losses and training
        'loss_mse'          : 1,
        'loss_mcpost'       : 0, # monte-carlo sample of the posterior probability
        'loss_p_smooh'      : 1e-4, # laplace filter on the posterior
        'loss_pe' 		    : 1e-1, # absolute prediction error
        'loss_MI' 		    : 0, # maximize mutual information
        'loss_ce' 	        : 0, # maximize cross-entropy

        # regularization loss
        'loss_spi' 	        : 2e-3,  # 'spike_cost'  : 2e-3, # todo: not used yet
        'loss_L1' 		    : 1e-2  , # weight regularization
        'loss_L2' 		    : 1e-2, # weight regularization

        'nsteps_train'      : 500,
        'learning_rate'     : 2e-2,	  # adam optimizer learning rate
        'clip_max_grad_val' : 0.1,

        ##########################################################################################
        # neuron
        'dt'			    : 10.,  # ms

        # neuron, decay and noise
        #'alpha_neuron'      : 0.1,  # time constant = dt/alpha ~ i.e. 10/0.1 = 100ms
        'neuron_tau' 	    : 100,  # time constant = dt/alpha ~ i.e. 10/0.1 = 100ms
        'noise_sd'  	    : 0.5,  # should we learn this?

        # neuron, stsp
        'neuron_stsp' 	    : False, # parameter todo: implement Dale's law

        ##########################################################################################
        # network; inherited from par (stimulus structures)
        'n_input'			: par['n_input'],
        'n_tuned_inp'		: par['n_tuned_input'],
        'n_sensory'			: 100,  # number of neurons in the sensory layer
        'n_hidden'			: par['n_hidden'],
        'n_tuned_output'	: par['n_tuned_output'],

        'n_rule_inp'		: par['n_rule_input'],
        'n_rule_outp'		: par['n_rule_output'],

        # sensory layer
        'sensory_input_ani'     : 0.01, # input anisotropy epsilon (Ben-Yishai, 1995)
        'sensory_input_gain'    : 2, # fix gain or learn c (Ben-Yishai, 1995)

        'sensory_noise_type'    : 'Normal_poisson',
        # 'Normal_fixed':       fix to neural noise_sd below
        # 'Normal_learn':       learn noise parameters
        # 'Normal_poisson':     Assumes that responses is an average of poisson-activated neurons
        'sensory_repr'	        : 'Efficient', # 'Uniform', 'Learn' or 'Efficient'
        # Efficient:            make the prior flat in the sensory space;
        # Uniform:              tuning of sensory neurons is uniform in the stimulus space
        # learn:                learn the best sensory neuron tunings

        # network; recurrent layer
        'rnn_weights'           : 'sym_sensory',
        # learn:                learn the best sensory neuron tunings
        # sym_sensory:          symmetric in the sensory space
        # sym_stimulus:         symmetric in the stimulus space
        'rnn_weights_shift'     : 86,  # 'learn' or a number, J0 (Ben-Yishai, 1995)
        'rnn_weights_scale'     : 112, # 'learn' or a number, J2 (Ben-Yishai, 1995)
        'rnn_activation'        : tf.nn.sigmoid,
        'rnn_noise_type'        : 'Normal_poisson'
    }




    return rvnet_hp