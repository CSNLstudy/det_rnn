import numpy as np
import tensorflow as tf

__all__ = ['_initialize', '_random_normal_abs', '_alternating',
		   '_EI_mask', '_modular_weight_cost_mask', '_convert_to_rg',
		   '_silencing_mask', '_alpha_mask']

# Inherited from Masse
def _initialize(dims, shape=0.1, scale=1.0):
	w = np.random.gamma(shape, scale, size=dims).astype(np.float32)
	return w

# Inherited from JsL
def _random_normal_abs(dims, std):
    # y = np.random.gamma(0.1, scale, size=dims)
    y = np.abs(np.random.normal(0, std, size=dims)).astype(np.float32)
    return np.float32(y)

def _alternating(x, size):
	tmp = np.tile(np.array(x), np.int(np.ceil(size / 2)))
	tmp2 = tmp[0:size]
	return tmp2.astype(np.float32)

# Nonmodular w_RNN mask
def _EI_mask(n_total, exc_inh_prop):
	n_exc = int(n_total * exc_inh_prop)
	n_inh = n_total - n_exc
	ind_inh = np.round(np.linspace(1, n_total-1, n_inh)).astype(int)

	EI_list = np.ones(n_total, dtype=np.float32)
	EI_list[ind_inh] = -1.
	EI_matrix = np.diag(EI_list)

	return np.float32(EI_matrix)

def _modular_weight_cost_mask(n_input, n_output, connect_cost_within, connect_cost_forward, connect_cost_back):

	in2in = np.ones((n_input, n_input))*connect_cost_within
	in2out = np.ones((n_input, n_output))*connect_cost_forward

	out2in = np.ones((n_output, n_input))*connect_cost_back
	out2out = np.ones((n_output, n_output))*connect_cost_within

	in2 = np.concatenate((in2in, in2out), axis=1)
	out2 = np.concatenate((out2in, out2out), axis=1)

	imask = np.concatenate((in2, out2), axis=0)

	return np.float32(imask)

def _silencing_mask(n_tuned, n_untuned, n_output):

	in2in = np.ones((n_tuned, n_tuned))
	in2h = np.ones((n_tuned, n_untuned))
	in2out = np.ones((n_tuned, n_output))

	h2in = np.ones((n_untuned, n_tuned))
	h2h = np.ones((n_untuned, n_untuned))
	h2out = np.ones((n_untuned, n_output))

	out2in = np.ones((n_output, n_tuned))
	out2h = np.ones((n_output, n_untuned))
	out2out = np.ones((n_output, n_output))

	in2 = np.concatenate((in2in, in2h * 0, in2out * 0), axis=1)
	h2 = np.concatenate((h2in * 0, h2h, h2out), axis=1)
	out2 = np.concatenate((out2in * 0, out2h, out2out), axis=1)
	mask_silencing_tuned = np.concatenate((in2, h2, out2), axis=0)

	in2 = np.concatenate((in2in, in2h * 0, in2out), axis=1)
	h2 = np.concatenate((h2in * 0, h2h, h2out * 0), axis=1)
	out2 = np.concatenate((out2in, out2h * 0, out2out), axis=1)
	mask_silencing_untuned = np.concatenate((in2, h2, out2), axis=0)

	in2 = np.concatenate((in2in, in2h, in2out * 0), axis=1)
	h2 = np.concatenate((h2in, h2h, h2out * 0), axis=1)
	out2 = np.concatenate((out2in * 0, out2h * 0, out2out), axis=1)
	mask_silencing_out = np.concatenate((in2, h2, out2), axis=0)

	return np.float32(mask_silencing_tuned), np.float32(mask_silencing_untuned), np.float32(mask_silencing_out)

def _alpha_mask(n_input, n_output, alpha_input, alpha_output, batch_size):

	alpha_mask = np.concatenate((alpha_input*np.ones((batch_size, n_input)),
								 alpha_output*np.ones((batch_size, n_output)),
								 ), axis=1)

	return np.float32(alpha_mask)

def estim_error(output, par, trial_info):

	MAP_output = output[par['output_rg'][par['output_rg'] > np.max(par['dead_rg'])].min():, :, :]
	MAP_output = tf.nn.log_softmax(MAP_output, axis=2).numpy()
	MAP_output = np.sum(MAP_output, axis=0)
	MAP_output = np.argmax(MAP_output, axis=1) - 1
	MAP_dirs = par['stim_dirs'][MAP_output]
	Target_dirs = par['stim_dirs'][trial_info['stimulus_ori']]
	error = np.arccos(np.cos(2 * (MAP_dirs - Target_dirs) / 180 * np.pi)) / np.pi * 180 / 2

	return np.float32(error)

# Modular w_RNN mask
def _modular_mask(connect_prob, n_hidden, exc_inh_prop):
	n_exc = int(n_hidden * exc_inh_prop)
	rg_exc = range(n_exc)
	rg_inh = range(n_exc, n_hidden)

	exc_module1 = rg_exc[:len(rg_exc) // 2]
	inh_module1 = rg_inh[:len(rg_inh) // 2]
	exc_module2 = rg_exc[len(rg_exc) // 2:]
	inh_module2 = rg_inh[len(rg_inh) // 2:]

	Crec = np.zeros((n_hidden, n_hidden))
	for i in exc_module1:
		Crec[i, exc_module1] = 1
		Crec[i, i] = 0
		Crec[i, exc_module2] = 1 * (np.random.uniform(size=len(exc_module2)) < connect_prob)
		Crec[i, inh_module1] = -np.sum(Crec[i, rg_exc]) / len(inh_module1)
	for i in exc_module2:
		Crec[i, exc_module2] = 1
		Crec[i, i] = 0
		Crec[i, exc_module1] = 1 * (np.random.uniform(size=len(exc_module1)) < connect_prob)
		Crec[i, inh_module2] = -np.sum(Crec[i, rg_exc]) / len(inh_module2)
	for i in inh_module1:
		Crec[i, exc_module1] = 1
		Crec[i, inh_module1] = -np.sum(Crec[i, rg_exc]) / (len(inh_module1) - 1)
		Crec[i, i] = 0
	for i in inh_module2:
		Crec[i, exc_module2] = 1
		Crec[i, inh_module2] = -np.sum(Crec[i, rg_exc]) / (len(inh_module2) - 1)
		Crec[i, i] = 0

	Crec[Crec > 0] = 1.
	Crec[Crec < 0] = -1.
	return np.float32(Crec)

# Convert range specs(dictionary) into time-step domain
def _convert_to_rg(design, dt):
	if type(design) == dict:
		rg_dict = {}
		for k,v in design.items():
			if len(np.shape(v)) == 1:
				start_step = round(v[0] / dt * 1000.)
				end_step = round(v[1] / dt * 1000.)
				rg_dict[k] = np.arange(start_step, end_step)
			else:
				rg_dict[k] = np.concatenate([np.arange(round(i[0] / dt * 1000.),round(i[1] / dt * 1000.)) for i in v])
		return rg_dict

	elif type(design) in (tuple, list):
		if len(np.shape(design)) == 1:
			start_step = round(design[0] / dt * 1000.)
			end_step = round(design[1] / dt * 1000.)
			rg = np.arange(start_step, end_step)
		else:
			rg = np.concatenate([np.arange(round(i[0] / dt * 1000.), round(i[1] / dt * 1000.)) for i in design])
		return rg
