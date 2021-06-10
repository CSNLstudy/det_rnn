import tensorflow as tf

__all__ = ['mgrnn_hp']

def mgrnn_hp(par = None):
    # Model hyperparameters(modifiable, as needed)

    if par is None:
        from det_rnn.base import par

    grnn_hp = {
        # saving parameters
        'output_base'       : 'D:/ProjGit/det_rnn/experiments/mgrnn/',
        'model_number'      : 0,
        'saving_freq' 	    : 50, # save every saving_freq iterations
        'dtype'             : 'tf.float32',

        ##########################################################################################
        # losses and training
        'loss_mse_dec'      : 0, #1e-4,
        'loss_mse_est'      : 0, # 1e-4,
        'loss_ce_dec'       : 1e-1, # CE is needed for distribution learning
        'loss_ce_est'       : 1e-1,

        # regularization loss
        'loss_spike' 	    : 0, #1e-2, #2e-3, # 'spike_cost'  : 2e-3,
        'loss_L1' 		    : 0, #1e-4, #2e-3, # weight regularization
        'loss_L2' 		    : 0, #2e-3, # weight regularization
        # dropout when training the output
        'dropout'           : 0, # <=0, no drop out. otherwise 0 < dropout < 1 to indicate probability of dropout
        'learning_rate'     : 5e-3,	  # adam optimizer learning rate = lowest possible learning rate. the first
                                      # learning rate is about x10
        'clip_max_grad_val' : 10,

        'scheduler'         : 'scheduler_timeconstant', # 'scheduler_separate', 'scheduler_estimFirst', None

        ##########################################################################################
        # neuron
        'dt'			    : 10.,  # ms

        # neuron, decay and noise
        #'alpha_neuron'      : 0.1,  # time constant = dt/alpha ~ i.e. 10/0.1 = 100ms
        # 'neuron_tau' 	    : 100,  # time constant = dt/alpha ~ i.e. 10/0.1 = 100ms
        'tau_train'         : False,
        'tau_min'           : 100, #10
        'tau_max'           : 100, #15000
        'noise_sd'  	    : 0.1,  # should we learn this?

        # neuron, stsp and Dale's law
        'dale'              : False, # parameter todo: implement Dale's law
        'stsp' 	            : False,
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
        'n_hidden'		    : [par['n_hidden'], par['n_hidden']],

        'n_tuned_output'	: par['n_tuned_output'],
        'n_output_dm'       : par['n_output_dm'],

        'n_rule_input'		: par['n_rule_input'],
        'n_rule_output_dm'	: par['n_rule_output_dm'],
        'n_rule_output_em'	: par['n_rule_output_em'],
        'activation'        : 'sigmoid', # 'relu'

        ##########################################################################################
        # network;
        # 'gate_in'               : True, # => add into the gate rnn
        'modular'               : True,
        'gate_out'              : False,
        'gate_rnn'              : True,
        'update'                : 'Kim', # or 'Masse'
        'out_representation'    : 'logits', #'probs'

        ###########################################################################################
        # input rule structure at the end for yaml visibility
        'input_rule_rg': par['input_rule_rg']
    }

    return mgrnn_hp