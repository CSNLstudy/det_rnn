import sys, copy
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from det_rnn import *
import det_rnn.analysis as da
import det_rnn.train as dt

##############################################################################
# Evolution of information with binary codes
##############################################################################
stimulus = Stimulus(par)

## Parameters ################################################################
Nhidden  = 100
Nrule    = 2
NruleIn  = 2
NruleOut = 1
Nstim    = 24
Nout     = 24
pEI      = 0.7  # make sure Ninhibit > Nstim
overlap  = 3
dt       = 10
tau      = 50
privg    = 1e5
devi     = 1e-6
noise_rnn_sd = 0.5 
batch_size   = 128

Nexcite  = int(Nhidden*pEI)
Ninhibit = Nhidden - Nexcite
tuning_matrix = stimulus.tuning_input.reshape((24,24))
# tuning_matrix = stimulus.tuning_input.reshape((24,24)).astype(np.float32) ## float32
alpha    = dt/tau


## Make hidden states ########################################################
H = np.zeros((Nhidden,Nstim))
rollvec = np.zeros(Nstim)
rollvec[:overlap] = 1.
for i in range(Nstim): H[Nrule:(Nrule+Nstim),i] =  np.roll(rollvec,i-int(overlap/2))
for i in range(Nstim): H[i+Nexcite,i] = 1.

## Make Wrnn #################################################################
### Initialize
w_rnn0 = np.random.rand(Nhidden,Nhidden) 
# w_rnn0 = np.zeros((Nhidden,Nhidden)) 
w_rnn0[:,Nexcite:] *= (-1.)
for i in range(Nhidden): w_rnn0[i,i] = -1.

### Make (24,24) block
stim_block = np.ones((Nstim,Nstim))
for i in range(Nstim): stim_block[i,i] = -1.
w_rnn0[Nrule:(Nrule+Nstim),Nrule:(Nrule+Nstim)] = stim_block

### Make (30,24) block
w_rnn0[Nexcite:,Nrule:(Nrule+Nstim)] = 1./overlap

### Challenging part
rollvec = -np.ones(Nstim)*overlap
rollvec[:overlap] = -overlap + 2.
for i in range(Nstim): w_rnn0[(i+Nrule),Nexcite:(Nexcite+Nstim)] =  np.roll(rollvec,i-int(overlap/2))

### 
w_rnn0[Nexcite:,Nexcite:(Nexcite+Nstim)] = - np.ones((Ninhibit,Nstim))
w_rnn0[(Nrule+Nstim):Nexcite,Nrule:(Nrule+Nstim)] = 1.
w_rnn0[(Nrule+Nstim):Nexcite,Nexcite:(Nexcite+Nstim)] = -float(overlap)

### Make rule strings(TODO: may not work if Nrule > 2)
w_rnn0[:Nrule,:] = 0
w_rnn0[:,:Nrule] = 0
for i in range(Nrule): w_rnn0[i,i] = -1.
w_rnn0[0,1] = 1.
w_rnn0[1,0] = 1.

### Complete!!!
w_roll = alpha*w_rnn0 + np.eye(Nhidden)
w_rnn  = w_rnn0 + np.eye(Nhidden)

### Sanity Check
np.linalg.matrix_rank(w_rnn0)
np.linalg.matrix_rank(w_rnn)
np.linalg.matrix_rank(w_roll)

abs((w_rnn @ H) - H).sum()

_H = copy.deepcopy(H)
_H[:Nrule,:] = 1
abs((w_rnn @ _H) - _H).sum()



## Make Win ##################################################################
# "Rule 1" is trivial, thus ruled out 
Resp_input_trunc = np.zeros(NruleIn - 1 + Nstim); Resp_input_trunc[0] = 0.8
Resp_h = np.zeros((Nhidden,1)); Resp_h[:NruleIn,:] = -1.
Htotal = np.concatenate((Resp_h,H*devi),axis=1) # _H : retain, H : respond

tr_mat_tilde = np.concatenate((Resp_input_trunc.reshape((-1,1)),
                               np.concatenate((np.zeros((1,Nstim)), tuning_matrix))),axis=1)
tr_mat_full  = np.concatenate((0.8*np.ones((1,NruleIn - 1 + Nstim)),tr_mat_tilde),axis=0)

INV0 = np.linalg.inv(tuning_matrix)
OneZero = np.zeros((Nstim+1,1)); OneZero[0,0] = 1/0.8
INV  = np.concatenate((OneZero, np.concatenate((np.zeros((1,Nstim)),INV0))),axis=1)

w_in_tilde = Htotal @ INV
w_in = np.concatenate((np.zeros((Nhidden,1)),w_in_tilde), axis=1)
abs(w_in_tilde@tr_mat_tilde- Htotal).sum()

## Make Wout #################################################################
w_out = np.zeros((Nhidden,Nout+NruleOut))
# w_out[Nrule:(Nrule+Nstim),NruleOut:(NruleOut+Nstim)] = H[Nrule:(Nstim+Nrule),:Nstim]
w_out[Nrule:(Nrule+Nstim),NruleOut:(NruleOut+Nstim)] = tuning_matrix
w_out[0,0] = privg

## var_dict ##################################################################
var_dict = {}
h0 = np.zeros(Nhidden)
h0[:Nrule] = 1e-5
var_dict['h'] = h0
var_dict['w_in'] = w_in
var_dict['w_rnn'] = w_rnn
var_dict['w_out'] = w_out

## functions #################################################################
def relu(x):
    return np.maximum(x,0.)

def rnn_cell(h, rnn_input, var_dict=var_dict):
    h = relu((1.-alpha)*h + \
             alpha*(rnn_input @ var_dict['w_in'].T + h @ var_dict['w_rnn'].T))
             # np.random.normal(size=Nhidden, loc=0,scale=2.*alpha*noise_rnn_sd))
    return h

def rnn_model(input_data, var_dict=var_dict):
	h = np.tile(var_dict['h'], (batch_size, 1))
	h_stack = []; y_stack = []
	*_input_data, = input_data

	for _iter, rnn_input in enumerate(_input_data):
		h = rnn_cell(h, rnn_input)
		h_stack.append(h)
		y_stack.append(h @ var_dict['w_out'])

	h_stack = np.stack(h_stack)
	y_stack = np.stack(y_stack)
	
	return h_stack, y_stack


## Run RNN ###################################################################
par['noise_sd'] = 0
par['design'].update({'iti'     : (0,  1.5),
                      'stim'    : (1.5,3.0),
                      'delay'   : (3.0,89.0),
                      'estim'   : (89.0,93.5)})
par = update_parameters(par)
stimulus = Stimulus(par)
trial_info = stimulus.generate_trial()
Hrnn, Yrnn = rnn_model(trial_info['neural_input'])
Yrnn_normalize = Yrnn/(np.sum(Yrnn,axis=-1,keepdims=True)+np.finfo(float).eps)

fig, axes = plt.subplots(3,1, figsize=(10,8))
TEST_TRIAL = np.random.randint(par['batch_size'])
axes[0].imshow(trial_info['neural_input'][:,TEST_TRIAL,:].T, aspect='auto'); axes[0].set_title("Neural Input")
axes[1].imshow(trial_info['desired_output'][:,TEST_TRIAL,:].T, aspect='auto'); axes[1].set_title("Desired Output")
axes[2].imshow(Yrnn_normalize[:,TEST_TRIAL,:].T,  aspect='auto', vmax=0.13)
fig.tight_layout(pad=2.0)
plt.show()

        
## Sanity check ###############################################################
plt.imshow(tr_mat_full)
plt.imshow(w_in@tr_mat_full)

plt.imshow((H.T @ w_out).T/np.sum(H.T @ w_out, axis=1))
plt.imshow((_H.T @ w_out).T/np.sum(_H.T @ w_out, axis=1))
plt.colorbar()
plt.show()


### 1. Does ITI input induce invariance? : Yes
ITI_input = np.zeros(NruleIn + Nstim);  ITI_input[0] = 0.8
plt.plot(ITI_input)
plt.plot(ITI_input @ var_dict['w_in'].T)
plt.plot(rnn_cell(var_dict['h'], ITI_input))

h = rnn_cell(var_dict['h'], ITI_input)
for i in range(2000):
    plt.plot(h)
    h = rnn_cell(h, ITI_input) 
plt.show()

### 2. Does stimulus period induce gradual evidence accumulation?
#### No, instability of population code causes exponential growth
stim_input = tr_mat_full[:,1]
h = rnn_cell(h0, stim_input)
for i in range(10):
    plt.plot(h)
    h = rnn_cell(h, stim_input) 
plt.show()
ArithmeticErrorh = rnn_cell(h0, stim_input)
for i in range(150):
    plt.plot(h @ var_dict['w_out'])
    h = rnn_cell(h, stim_input) 
plt.show()

### 3. Does the delay period stably maintain information? 
#### This depends on "robustness(sparseness)" of Wrnn
h = rnn_cell(h, ITI_input)
for i in range(800):
    plt.plot(h)
    h = rnn_cell(h, ITI_input) 
plt.show()

for i in range(24):
    i_stim_input = tr_mat_full[:,i+1]
    h = ( w_in @ i_stim_input).reshape((1,-1)) @ var_dict['w_rnn'].T
    plt.plot(h.T)

### 4. Does the response period evoke response?
resp_input = copy.deepcopy(ITI_input) 
resp_input[1] = 0.8






