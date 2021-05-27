import numpy as np

__all__ = ['initialize', 'alternating',
           'w_rnn_mask', 'w_lat_mask', 'modular_mask', 'convert_to_rg', 'w_design']

win1_vec = np.array([ 2.59857481,  2.35565115,  1.75575469,  1.13477177,  0.6605314 ,
        0.2829742 , -0.15237553, -0.82288476, -1.49037653, -1.83974699,
       -1.94694559, -1.95609656, -1.94834135, -1.95609656, -1.94694559,
       -1.83974699, -1.49037653, -0.82288476, -0.15237553,  0.2829742 ,
        0.6605314 ,  1.13477177,  1.75575469,  2.35565115])

w11_seg1  = np.array([ 0.15607648,  0.15533725,  0.1651392 ,  0.17681165,  0.17296231,
        0.14339285,  0.08108697, -0.02002888, -0.14474041, -0.24771592,
       -0.30056284, -0.31745927, -0.32116234, -0.31808887, -0.30056284,
       -0.24771592, -0.14474041, -0.02002888,  0.08108697,  0.14339285,
        0.17296231,  0.17681165,  0.1651392 ,  0.15533725])

w11_seg2  = np.array([-0.34803867, -0.3629441 , -0.39370855, -0.41751019, -0.42556038,
       -0.41813586, -0.39434245, -0.34793454, -0.28007369, -0.19999853,
       -0.11695192, -0.04929947, -0.02362019, -0.05061482, -0.11695192,
       -0.19999853, -0.28007369, -0.34793454, -0.39434245, -0.41813586,
       -0.42556038, -0.41751019, -0.39370855, -0.3629441 ])

w21_vec  = np.array([ 0.16301947,  0.15240762,  0.11970752,  0.06528813, -0.00718662,
       -0.09092732, -0.17961154, -0.26556932, -0.3439168 , -0.41061385,
       -0.46592546, -0.50275467, -0.51666478, -0.50396961, -0.46592546,
       -0.41061385, -0.3439168 , -0.26556932, -0.17961154, -0.09092732,
       -0.00718662,  0.06528813,  0.11970752,  0.15240762,  0.16301947,
        0.15240762,  0.11970752,  0.06528813, -0.00718662, -0.09092732,
       -0.17961154, -0.26556932, -0.3439168 , -0.41061385, -0.46592546,
       -0.50275467, -0.51666478, -0.50396961, -0.46592546, -0.41061385,
       -0.3439168 , -0.26556932, -0.17961154, -0.09092732, -0.00718662,
        0.06528813,  0.11970752,  0.15240762])

w22_vec  = np.array([ 0.01704395, -0.05144068, -0.10454784, -0.08310243, -0.09664226,
       -0.17340838, -0.2398315 , -0.29353789, -0.35558997, -0.40100754,
       -0.43415138, -0.47748037, -0.50987298, -0.4791221 , -0.43415138,
       -0.40100754, -0.35558997, -0.29353789, -0.2398315 , -0.17340838,
       -0.09664226, -0.08310243, -0.10454784, -0.05144068,  0.01704395,
       -0.05144068, -0.10454784, -0.08310243, -0.09664226, -0.17340838,
       -0.2398315 , -0.29353789, -0.35558997, -0.40100754, -0.43415138,
       -0.47748037, -0.50987298, -0.4791221 , -0.43415138, -0.40100754,
       -0.35558997, -0.29353789, -0.2398315 , -0.17340838, -0.09664226,
       -0.08310243, -0.10454784, -0.05144068])

# Inherited from Masse
# def initialize(dims, shape=0.1, scale=1.0):
# 	w = np.random.gamma(shape, scale, size=dims).astype(np.float32)
# 	return np.float32(w)

def initialize(dims, gain=1., shape=0.1, scale=1.0):
    w = gain*np.random.gamma(shape, scale, size=dims).astype(np.float32)
    return np.float32(w)

# Inherited from JsL
# def random_normal_abs(dims): # Todo (HL): random.gamma
#     y = np.random.gamma(0.1, 1.0, size=dims) # for masse
#     # y = np.random.gamma(0.001, 0.01, size=dims) # for nomasse(but not trainable)
#     return np.float32(y)

def w_design(w_name, par):
    if w_name == 'w_in1':
        n         = par['n_tuned_input']
        b         = 2.5
        # x_support = np.arange(2*np.pi, step=2*np.pi/n)
        # dexp      = np.exp(-np.abs(x_support-np.pi)*b)*b*2 - b
        # dexp      = np.roll(dexp, -np.argmax(dexp))
        dexp      = win1_vec
        quo, rem  = divmod(par['n_hidden1'], n)
        w, w_apdx = np.empty((0,n)), np.empty((0,n))

        if quo > 0: 
            w      = np.tile(np.stack([np.roll(dexp, s) for s in range(n)]),quo).T
        if rem > 0:
            w_apdx = np.stack([np.roll(dexp, s) for s in np.arange(n, step=int(n/rem))[:rem]])

        w = np.concatenate((w,w_apdx), axis=0).T

    elif ((w_name == 'w_in2') | (w_name == 'w_out_em')):
        if w_name == 'w_in2':
            n     = par['n_tuned_input']
        else:
            n     = par['n_tuned_output']

        # repeat quo times, and evenly space by rem
        cos_vec   = np.cos(np.arange(2*np.pi, step=2*np.pi/n))
        quo, rem  = divmod(par['n_hidden2'], n)
        w, w_apdx = np.empty((0,n)), np.empty((0,n))

        if quo > 0: 
            w      = np.tile(np.stack([np.roll(cos_vec, s) for s in range(n)]),quo).T
        if rem > 0:
            w_apdx = np.stack([np.roll(cos_vec, s) for s in np.arange(n, step=int(n/rem))[:rem]])

        if w_name == 'w_in2':
            w = np.concatenate((w,w_apdx), axis=0).T * 0.8 # do not know why this works but anyways
        else:
            w = np.concatenate((w,w_apdx), axis=0)   * 0.3 # do not know why this works but anyways
    
    elif w_name   == 'w_out_dm':
        w = np.kron(np.eye(2), np.ones(int(par['n_hidden1']/2))).T

    elif w_name   == 'w_rnn11':
        # TODO(HG): generalize here
        w = np.zeros((par['n_hidden1'],par['n_hidden1']))
        for i in range(24):
            w[:24, i]    = np.roll(w11_seg1, i)
            w[24:, i+24] = np.roll(w11_seg1, i)
            w[24:, i]    = np.roll(w11_seg2, i)
            w[:24, i+24] = np.roll(w11_seg2, i)

    elif w_name   == 'w_rnn21':
        # TODO(HG): generalize here
        w  = np.zeros((par['n_hidden2'],par['n_hidden1']))
        for i in range(24):
            w[:,i]    = np.roll(w21_vec,  6+i)
            w[:,i+24] = np.roll(w21_vec, -6+i)

    elif w_name   == 'w_rnn22':
        # TODO(HG): generalize here
        w = np.zeros((par['n_hidden2'],par['n_hidden2']))
        for i in range(48):
            w[:, i] = np.roll(w22_vec, i)

    return w.astype(np.float32)


def alternating(x, size):
    tmp = np.tile(np.array(x), np.int(np.ceil(size / 2)))
    tmp2 = tmp[0:size]
    return tmp2.astype(np.float32)

# Nonmodular w_RNN mask
def w_rnn_mask(n_hidden, exc_inh_prop):
    n_exc = int(n_hidden * exc_inh_prop)
    rg_inh = range(n_exc, n_hidden)
    Crec = np.ones((n_hidden, n_hidden)) # - np.eye(n_hidden)
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

