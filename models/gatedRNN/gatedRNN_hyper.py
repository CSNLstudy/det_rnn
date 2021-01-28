import tensorflow as tf

__all__ = ['grnn_hp']

def grnn_hp(par = None):
    # Model hyperparameters(modifiable, as needed)

    if par is None:
        from det_rnn.base import par

    grnn_hp = {
        # saving parameters
        'output_path'       : '/Users/JRyu/github/det_rnn/experiments/grnn/model1/',
        'model_output'      : '/Users/JRyu/github/det_rnn/experiments/grnn/model1/model',
        'log_path' 	        : '/Users/JRyu/github/det_rnn/experiments/grnn/model1/logs',
        'saving_freq' 	    : 100, # save every saving_freq iterations
        'dtype'             : 'tf.float32',

        ##########################################################################################
        # losses and training
        'loss_mse_dec'          : 1,
        'loss_mse_est'          : 1,
        'loss_ce_dec'           : 1,
        'loss_ce_est'           : 1,

        # regularization loss
        'loss_spike' 	        : 2e-3,  # 'spike_cost'  : 2e-3, # todo: not used yet
        'loss_L1' 		    : 1e-2  , # weight regularization
        'loss_L2' 		    : 1e-2, # weight regularization

        'nsteps_train'      : 5000,
        'learning_rate'     : 2e-2,	  # adam optimizer learning rate
        'clip_max_grad_val' : 0.1,

        ##########################################################################################
        # neuron
        'dt'			    : 10.,  # ms

        # neuron, decay and noise
        #'alpha_neuron'      : 0.1,  # time constant = dt/alpha ~ i.e. 10/0.1 = 100ms
        'neuron_tau' 	    : 100,  # time constant = dt/alpha ~ i.e. 10/0.1 = 100ms
        'noise_sd'  	    : 0.5,  # should we learn this?

        # neuron, stsp and Dale's law
        'neuron_stsp' 	    : False, # parameter todo: implement Dale's law
        # todo (josh): facilitation and depress... check implementation?? put alternating in the rnn cell rather than
        #  with hp
        'exc_inh_prop'      : 0.8, # Dale's law
        'alpha_std'         : (0.05, 0.00667),  # efficacy time constant,  facilitating and depressing
        'alpha_stf'         : (0.00667, 0.05),  # utilization time constant
        'U'                 : (0.15, 0.45),

        # stimulus structure (inherited from par)
        'n_input'		    : par['n_input'],
        'n_tuned_input'		: par['n_tuned_input'],
         #'n_sensory'		: 100,  # number of neurons in the sensory layer
        'n_hidden'			: par['n_hidden'],
        'n_tuned_output'	: par['n_tuned_output'],

        'n_rule_input'		: par['n_rule_input'],
        'n_rule_output_dm'	: par['n_rule_output_dm'],
        'n_rule_output_em'	: par['n_rule_output_em'],

        ##########################################################################################
        # network;

        'out_gate'              : True,  # false not implemented yet
        # network; recurrent layer
        'rnn_gate'              : True, # false not implemented yet
        'rnn_weights'           : 'normal',
        # learn:                learn the best sensory neuron tunings
        # sym_sensory:          symmetric in the sensory space
        # sym_stimulus:         symmetric in the stimulus space
        'rnn_weights_shift'     : 86,  # 'learn' or a number, J0 (Ben-Yishai, 1995)
        'rnn_weights_scale'     : 112, # 'learn' or a number, J2 (Ben-Yishai, 1995)
        # 'rnn_activation'        : tf.nn.sigmoid,
        'rnn_noise_type'        : 'Normal_fixed'
    }

    return grnn_hp