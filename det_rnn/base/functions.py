import numpy as np

__all__ = ['initialize', 'random_normal_abs', 'alternating',
		   'w_rnn_mask', 'w_lat_mask', 'modular_mask', 'convert_to_rg']

# Inherited from Masse
def initialize(dims, shape=0.1, scale=1.0):
	w = np.random.gamma(shape, scale, size=dims).astype(np.float32)
	return w

# Inherited from JsL
def random_normal_abs(dims): # Todo (HL): random.gamma
    y = np.random.gamma(0.1, 1.0, size=dims) # for masse
    # y = np.random.gamma(0.001, 0.01, size=dims) # for nomasse(but not trainable)
    return np.float32(y)

def alternating(x, size):
	tmp = np.tile(np.array(x), np.int(np.ceil(size / 2)))
	tmp2 = tmp[0:size]
	return tmp2.astype(np.float32)

# Nonmodular w_RNN mask
def w_rnn_mask(n_hidden, exc_inh_prop):
	n_exc = int(n_hidden * exc_inh_prop)
	rg_inh = range(n_exc, n_hidden)
	Crec = np.ones((n_hidden, n_hidden)) - np.eye(n_hidden)
	Crec[rg_inh,:] = Crec[rg_inh,:]*(-1.)
	return np.float32(Crec)

# w_lat mask
def w_lat_mask(n_visual):
	Crec = np.ones((n_visual, n_visual)) - np.eye(n_visual)
	return np.float32(Crec)

# Modular w_RNN mask
def modular_mask(n_hidden, exc_inh_prop):
    n_exc1 = int(n_hidden * exc_inh_prop / 2.)
    n_exc2 = int(n_hidden * exc_inh_prop / 2.)
    rg_inh1 = range(n_exc1,int(n_hidden/2))
    rg_inh2 = range(int(n_hidden/2)+n_exc2,n_hidden)
    Crec = np.ones((n_hidden, n_hidden))
    Crec[range(n_hidden),range(n_hidden)] = 0
    Crec[rg_inh1,:] = -Crec[rg_inh1,:]
    Crec[rg_inh2,:] = -Crec[rg_inh2,:]
    return Crec.astype(np.float32)

# Convert range specs(dictionary) into time-step domain
def convert_to_rg(design, dt):
	if type(design) == dict:
		rg_dict = {}
		for k,v in design.items():
			if len(np.shape(v)) == 1:
				start_step = round(v[0] / dt * 1000.)
				end_step = round(v[1] / dt * 1000.)
				rg_dict[k] = np.arange(start_step, end_step, dtype=np.int32)
			else:
				rg_dict[k] = np.concatenate([np.arange(round(i[0] / dt * 1000.),round(i[1] / dt * 1000.), dtype=np.int32) for i in v])
		return rg_dict

	elif type(design) in (tuple, list):
		if len(np.shape(design)) == 1:
			start_step = round(design[0] / dt * 1000.)
			end_step = round(design[1] / dt * 1000.)
			rg = np.arange(start_step, end_step, dtype=np.int32)
		else:
			rg = np.concatenate([np.arange(round(i[0] / dt * 1000.), round(i[1] / dt * 1000.), dtype=np.int32) for i in design])
		return rg

