#Copying HWG's code
#%%
import numpy as np
import tensorflow as tf

# Inherited from JsL
def random_normal_abs(dims): # Todo (HL): random.gamma
    y = np.random.gamma(0.1, 1.0, size=dims) # for masse
    # y = np.random.gamma(0.001, 0.01, size=dims) # for nomasse(but not trainable)
    return np.float32(y)

def alternating(x, size):
	tmp = np.tile(np.array(x), np.int(np.ceil(size / 2)))
	tmp2 = tmp[0:size]
	return tmp2.astype(np.float32)

def w_rnn_mask(n_hidden, exc_inh_prop):
	n_exc = int(n_hidden * exc_inh_prop)
	rg_inh = range(n_exc, n_hidden)
	Crec = np.ones((n_hidden, n_hidden)) - np.eye(n_hidden)
	Crec[rg_inh,:] = Crec[rg_inh,:]*(-1.)
	return np.float32(Crec)


#%% Model hyperparameters(modifiable)

n_input = 9
n_hidden = 30
n_output = 10
n_batch = 29
exc_inh_prop = 0.8

def make_par(n_input, n_hidden, n_output, n_batch, exc_inh_prop = 0.8):

    par = {
        'n_input' : n_input,
        'n_hidden': n_hidden,
        'n_output': n_output,
        'n_batch' : n_batch,
        'exc_inh_prop': exc_inh_prop
    }

    par['n_exc'] = int(np.round(par['n_hidden']*par['exc_inh_prop']))

    hp  = {
        'masse' : True,
        'loss_fun'	: 1, # 0:'mse', 1:'centropy'
        'task_type' : 0, # 0:'decision', 1:'estimation'
        'learning_rate' : 2e-2,	  # adam optimizer learning rate
        'dt'            : 10.,
        'clip_max_grad_val'  : 0.1,
        'alpha_neuron'  : 0.1,
        'spike_cost'  : 2e-4, # mse: 2e-5, centropy: 2e-3
        'weight_cost' : 0.,
        'noise_rnn_sd': 0.5,
        'initialize_std': 3,


        'syn_x_init': np.ones((par['n_batch'], par['n_hidden']), dtype=np.float32),
        'syn_u_init': np.tile(alternating((0.15, 0.45), par['n_hidden']), (par['n_batch'], 1)),
        'alpha_std': alternating((0.05, 0.00667), par['n_hidden']),
        'alpha_stf': alternating((0.00667, 0.05), par['n_hidden']),
        'dynamic_synapse': np.ones(par['n_hidden'], dtype=np.float32),
        'U': alternating((0.15, 0.45), par['n_hidden']),

        'w_in_mask': np.ones((par['n_input'], par['n_hidden']), dtype=np.float32),
        'w_rnn_mask': np.ones((par['n_hidden'], par['n_hidden']), dtype=np.float32) - np.eye(par['n_hidden'],dtype=np.float32),
        'w_out_mask': np.concatenate((np.ones((par['n_exc'], par['n_output']), dtype=np.float32),
                                        np.zeros((par['n_hidden'] - par['n_exc'], par['n_output']),dtype=np.float32)), axis=0),

    }

    hp.update(
        {
        'h0': random_normal_abs((1, par['n_hidden'])),
        'w_in0': random_normal_abs((par['n_input'], par['n_hidden'])),
        'w_rnn0': random_normal_abs((par['n_hidden'], par['n_hidden'])),
        'b_rnn0': np.zeros(par['n_hidden'], dtype=np.float32),
        'w_out0': random_normal_abs((par['n_hidden'],par['n_output'])) * hp['w_out_mask'],
        'b_out0': np.zeros((par['n_output'],), dtype=np.float32)
        }
    )

    hp['EI_mask'] = w_rnn_mask(par['n_hidden'], par['exc_inh_prop'])

    for k, v in hp.items():
        hp[k] = tf.constant(v, name=k)

    hp.update(par)

    return hp
# %%
