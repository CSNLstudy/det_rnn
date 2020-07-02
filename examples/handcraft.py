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
NruleIn  = 2
NruleOut = 1
Nstim    = 24
Nout     = 24
pEI      = 0.7  # make sure Ninhibit > Nstim
overlap  = 13
dt       = 10
tau      = 50
privg    = 30.
noise_rnn_sd = 0.5 
batch_size   = 128

Nexcite  = int(Nhidden*pEI)
Ninhibit = Nhidden - Nexcite
tuning_matrix = stimulus.tuning_input.reshape((24,24))
tuning_matrix_sigmoid = np.exp(tuning_matrix)/np.sum(np.exp(tuning_matrix), axis=1)
for i in range(Nstim): tuning_matrix_sigmoid[i,i] = 1.
alpha    = dt/tau


## Make hidden states ########################################################
H = np.zeros((Nhidden,Nstim))
rollvec = np.zeros(Nstim)
rollvec[:overlap] = 1.
for i in range(Nstim): H[NruleIn:(NruleIn+Nstim),i] =  np.roll(rollvec,i-int(overlap/2))
for i in range(Nstim): H[i+Nexcite,i] = 1.

## Make Wrnn #################################################################
### Initialize
w_rnn0 = np.random.rand(Nhidden,Nhidden) 
w_rnn0[:,Nexcite:] *= (-1.)
for i in range(Nhidden): w_rnn0[i,i] = -1.

### Make (24,24) block
stim_block = np.ones((Nstim,Nstim))
for i in range(Nstim): stim_block[i,i] = -1.
w_rnn0[NruleIn:(NruleIn+Nstim),NruleIn:(NruleIn+Nstim)] = stim_block

### Make (30,24) block
w_rnn0[Nexcite:,NruleIn:(NruleIn+Nstim)] = 1./overlap

### Challenging part
rollvec = -np.ones(Nstim)*overlap
rollvec[:overlap] = -overlap + 2.
for i in range(Nstim): w_rnn0[(i+NruleIn),Nexcite:(Nexcite+Nstim)] =  np.roll(rollvec,i-int(overlap/2))

### 
w_rnn0[Nexcite:,Nexcite:(Nexcite+Nstim)] = - np.ones((Ninhibit,Nstim))
w_rnn0[(NruleIn+Nstim):Nexcite,NruleIn:(NruleIn+Nstim)] = 1.
w_rnn0[(NruleIn+Nstim):Nexcite,Nexcite:(Nexcite+Nstim)] = -float(overlap)

### Make rule strings(TODO: may not work if NruleIn > 2)
w_rnn0[:NruleIn,:] = 0
w_rnn0[:,:NruleIn] = 0
for i in range(NruleIn): w_rnn0[i,i] = -1.
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
_H[:NruleIn,:] = 1
abs((w_rnn @ _H) - _H).sum()


ITI_h  = np.zeros((Nhidden,1)); ITI_h[:NruleIn,:] = 1.
ITI_input = np.zeros(NruleIn + Nstim)
ITI_input[:1] = 0.8
Htotal = np.concatenate((ITI_h,_H,H),axis=1) # _H : retain, H : respond


## Make Win ##################################################################
# past version
# w_in = H @ np.linalg.inv(tuning_matrix)
# abs((w_in)@tuning_matrix - H).sum()

tr_mat = np.concatenate((ITI_input.reshape((-1,1)),
                        np.concatenate((np.ones((1,Nstim))*0.8,np.zeros((1,Nstim)),tuning_matrix)),
                        np.concatenate((np.ones((2,Nstim))*0.8,tuning_matrix))),axis=1)

# w_in = Htotal @ (tr_mat.T) @ np.linalg.inv(tr_mat @ (tr_mat.T)) # why not work?
w_in = Htotal @ np.linalg.pinv(tr_mat)
abs(w_in@tr_mat- Htotal).sum()

## Make Wout #################################################################
w_out = np.zeros((Nhidden,Nout+NruleOut))
w_out[NruleIn:(NruleIn+Nstim),NruleOut:(NruleOut+Nstim)] = H[NruleIn:(Nstim+NruleIn),:Nstim]
w_out[0,0] = privg


## Sanity check ##############################################################
plt.imshow(tr_mat)
plt.imshow(w_in@tr_mat)

plt.imshow((H.T @ w_out).T/np.sum(H.T @ w_out, axis=1))
plt.colorbar()
plt.show()


## var_dict ##################################################################
var_dict = {}
var_dict['h'] = np.zeros(Nhidden)
var_dict['w_in'] = w_in
var_dict['w_rnn'] = w_rnn
var_dict['w_out'] = w_out


## functions #################################################################
def relu(x):
    return np.maximum(x,0.)

def rnn_cell(h, rnn_input, var_dict=var_dict):
    h = relu((1.-alpha)*h + \
             alpha*(rnn_input @ var_dict['w_in'].T) + h @ var_dict['w_rnn'].T)
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
        

## Sanity check ###############################################################
### 1. Does ITI input induce invariance?
plt.plot(ITI_input)
plt.plot(ITI_input @ var_dict['w_in'].T)
plt.plot(rnn_cell(var_dict['h'], ITI_input))

h = rnn_cell(var_dict['h'], ITI_input)
for i in range(8):
    plt.plot(h)
    h = rnn_cell(h, ITI_input) 
plt.show()



### 2. Does
stim_input = tr_mat[:,12]

plt.plot(stim_input)
plt.plot(stim_input @ var_dict['w_in'].T)
plt.plot(rnn_cell(var_dict['h'], stim_input))


### 3. Does stimulus period induce gradual evidence accumulation?
#### No, instability of population code causes exponential growth
stim_input = tr_mat[:,12]
h = rnn_cell(var_dict['h'], stim_input)
for i in range(8):
    plt.plot(h)
    h = rnn_cell(h, stim_input) 
plt.show()


### 4. Does the delay period stably maintain information?
h = rnn_cell(var_dict['h'], stim_input)
h = rnn_cell(h, ITI_input)

for i in range(3):
    plt.plot(h)
    h = rnn_cell(h, ITI_input) 
plt.show()


### 5. Does the response period evoke response?
resp_input = copy.deepcopy(ITI_input) 
resp_input[1] = 0.8

for i in range(8):
    plt.plot(h)
    h = rnn_cell(h, resp_input) 
plt.show()

plt.imshow((H.T @ w_out).T/np.sum(H.T @ w_out, axis=1))





stim_input
tr_mat.shape




htest = np.tile(var_dict['h'], (batch_size, 1))






h = rnn_cell(var_dict['h'], tr_mat[:,12])
for i in range(25):
    plt.plot(h)
    h = rnn_cell(h, tr_mat[:,12])
plt.show()

h.shape
h = iti_input.reshape((1,-1))
h = h.reshape((1,-1))
for i in range(300):
    plt.plot(h.T)
    h = h @ var_dict['w_in'].T
plt.show()

var_dict['w_in'].shape

var_dict['w_in'].shape

h.reshape((1,-1)).shape
var_dict['w_in'].shape

h = rnn_cell(var_dict['h'], iti_input)
for i in range(300):
    plt.plot(h)
    h = rnn_cell(h, iti_input)
plt.show()


h = rnn_cell(h, tr_mat[:,12])

plt.plot(rnn_cell(h, tr_mat[:,12]))




## Run RNN ###################################################################
par['noise_sd'] = 0
par = update_parameters(par)
stimulus = Stimulus(par)
trial_info = stimulus.generate_trial()
Hrnn, Yrnn = rnn_model(trial_info['neural_input'][298:,:,:])
Yrnn_normalize = Yrnn/np.sum(Yrnn,axis=-1,keepdims=True)

plt.imshow(trial_info['neural_input'][299:,TEST_TRIAL,:].T, aspect='')

plt.plot(Yrnn_normalize[301,TEST_TRIAL,:])

plt.imshow(Hrnn[:20,TEST_TRIAL,:].T, vmax=1)

fig, axes = plt.subplots(3,1, figsize=(10,8))
TEST_TRIAL = np.random.randint(par['batch_size'])
axes[0].imshow(trial_info['neural_input'][:,TEST_TRIAL,:].T, aspect='auto'); axes[0].set_title("Neural Input")
axes[1].imshow(trial_info['desired_output'][:,TEST_TRIAL,:].T, aspect='auto'); axes[1].set_title("Desired Output")
axes[2].imshow(Yrnn_normalize[:10,TEST_TRIAL,:].T,  aspect='auto',vmax=0.1)
fig.tight_layout(pad=2.0)
plt.show()


fig, axes = plt.subplots(3,1, figsize=(10,8))
TEST_TRIAL = np.random.randint(par['batch_size'])
axes[0].imshow(trial_info['neural_input'][:,TEST_TRIAL,:].T, aspect='auto'); axes[0].set_title("Neural Input")
axes[1].imshow(trial_info['desired_output'][:,TEST_TRIAL,:].T, aspect='auto'); axes[1].set_title("Desired Output")
axes[2].imshow(Hrnn[:,TEST_TRIAL,:].T,  aspect='auto', vmax=0.01)
fig.tight_layout(pad=2.0)
plt.show()


input_ = trial_info['neural_input'][200,TEST_TRIAL,:]
tr_mat.shape


plt.plot(input_)
plt.plot(tr_mat[:,12])



plt.plot(input_ @ var_dict['w_in'].T)
plt.plot(tr_mat[:,12] + np.random.normal(loc=0) @ var_dict['w_in'].T)

np.linalg.cond(var_dict['w_in'])


Hrnn[:,TEST_TRIAL,:]


fig, axes = plt.subplots(3,1, figsize=(10,8))
TEST_TRIAL = np.random.randint(par['batch_size'])
axes[0].imshow(trial_info['neural_input'][:,TEST_TRIAL,:].numpy().T, aspect='auto'); axes[0].set_title("Neural Input")
axes[1].imshow(trial_info['desired_output'][:,TEST_TRIAL,:].numpy().T, aspect='auto'); axes[1].set_title("Desired Output")
axes[2].imshow(cenoutput[:,TEST_TRIAL,:].T,  aspect='auto', vmin=0, vmax=0.15)
fig.tight_layout(pad=2.0)
plt.show()



var_dict['w_in'].T.shape

rnn_cell(var_dict['h'], H[:,0])

tf.sqrt(, dtype=tf.float32)


par['design'].update({'iti'     : (0,  1.5),
                      'stim'    : (1.5,3.0),
                      'delay'   : (3.0,18.5),
                      'estim'   : (18.5,23.0)})
par   = update_parameters(par)
stimulus    = Stimulus(par)

trial_info  = dt.tensorize_trial(stimulus.generate_trial())
pred_output, H, _, _ = model.rnn_model(trial_info['neural_input'], dt.hp)
pred_output = da.softmax_pred_output(pred_output)

trial_info  = dt.tensorize_trial(stimulus.generate_trial())
pred_output, H, _, _ = model2.rnn_model(trial_info['neural_input'], dt.hp)
pred_output = da.softmax_pred_output(pred_output)

fig, axes = plt.subplots(3,1, figsize=(10,8))
TEST_TRIAL = np.random.randint(par['batch_size'])
axes[0].imshow(trial_info['neural_input'][:,TEST_TRIAL,:].numpy().T, aspect='auto'); axes[0].set_title("Neural Input")
axes[1].imshow(trial_info['desired_output'][:,TEST_TRIAL,:].numpy().T, aspect='auto'); axes[1].set_title("Desired Output")
axes[2].imshow(pred_output[:,TEST_TRIAL,:].T,  aspect='auto', vmin=0, vmax=0.15)
fig.tight_layout(pad=2.0)
plt.show()



plt.imshow(w_in.T)
plt.show()


plt.imshow((w_rnn @ H))
plt.show()

plt.imshow((w_rnn @ _H))
plt.show()


plt.imshow(w_rnn)
plt.show()

plt.imshow(w_out.T)
plt.colorbar()
plt.show()

plt.imshow((_H.T @ w_out).T/np.sum(_H.T @ w_out, axis=1))
plt.colorbar()
plt.show()

plt.imshow(np.exp(H.T @ w_out)/np.sum(np.exp(H.T @ w_out), axis=1))
plt.show()




plt.plot(model.var_dict['h'].numpy().T)
plt.show()

## 






##############################################################################
# Like-suppress-like motifs
##############################################################################


