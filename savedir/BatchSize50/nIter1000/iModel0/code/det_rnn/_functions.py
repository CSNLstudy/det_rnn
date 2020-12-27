import numpy as np

__all__ = ['_initialize', '_random_normal_abs', '_alternating',
		   '_w_rnn_mask', '_modular_mask', '_convert_to_rg', '_w_EI_input_mask']

# Inherited from Masse
def _initialize(dims, shape=0.1, scale=1.0):
	w = np.random.gamma(shape, scale, size=dims).astype(np.float32)
	return w

# Inherited from JsL
def _random_normal_abs(dims, ): # Todo (HL): random.gamma
    y = np.random.gamma(0.1, 1, size=dims)
    # y = np.abs(np.random.normal(0, 0.01, size=dims)).astype(np.float32)
    return np.float32(y)

def _alternating(x, size):
	tmp = np.tile(np.array(x), np.int(np.ceil(size / 2)))
	tmp2 = tmp[0:size]
	return tmp2.astype(np.float32)

# Nonmodular w_RNN mask
def _w_rnn_mask(n_hidden, exc_inh_prop):

	n_exc = int(n_hidden * exc_inh_prop)
	n_inh = n_hidden - n_exc
	EI_list = np.ones(n_hidden, dtype=np.float32)
	EI_list[-n_inh:] = -1.
	EI_matrix = np.diag(EI_list)

	return np.float32(EI_matrix)

# Input neurons are either excitatory or inhibitory neurons
def _w_EI_input_mask(n_input, n_hidden, exc_inh_prop, n_tuned_input):
	
	EI_matrix = np.ones((n_input, n_hidden))
	n_inh = np.round(n_tuned_input * (1-exc_inh_prop))
	space_inh = np.floor(n_tuned_input/n_inh)
	n_rule = n_input - n_tuned_input
	for i in range(int(n_inh)):
		iInd = n_rule + int(space_inh*i)
		EI_matrix[iInd, :] = -1
	EI_matrix = np.tile(EI_matrix.T, 2).T

	EI_in2in_matrix = np.diag(EI_matrix[:,0])

	return np.float32(EI_matrix), np.float32(EI_in2in_matrix)

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
