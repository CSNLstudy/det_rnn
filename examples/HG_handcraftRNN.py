import copy, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')
from det_rnn import *

##############################################################################
# RNN Pygmalion : Fixed-point evolution of information with binary codes
##############################################################################

# WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
## This RNN is very sensitive to noise as small as floating-point precision
## Thus, _stimulus.py should be modified
## From
### return {'neural_input'  : neural_input.astype(np.float32),
###         'stimulus_ori'  : stimulus_ori,
###         'desired_output': desired_output.astype(np.float32),
###         'mask'          : mask}

## To
### return {'neural_input'  : neural_input,
###         'stimulus_ori'  : stimulus_ori,
###         'desired_output': desired_output,
###         'mask'          : mask}
##############################################################################

## Parameters ################################################################
Nhidden = 100  # number of neurons
Nrule = 2  # number of "rule neurons"
NruleIn = 2  # number of rule inputs
NruleOut = 1  # number of rule outputs
Nstim = 24  # number of stimulus
Nout = 24  # number of stimulus output
pEI = 0.7  # make sure Ninhibit > Nstim (I think )
overlap = 3  # coding length (ex. if overlap=3 then pi: [...,0,1,1,1,0,...])
dt = 10
tau = 50
privg = 1e5  # output "suppression" signal corresponding to "rule neurons"
devi = 0.1  # evidence accumulation scaler(speed), acronym for derivative_evidence
noise_rnn_sd = 0.005  # noise : see model part
batch_size = 128

## Dependent parameters ######################################################
stimulus = Stimulus(par)
Nexcite = int(Nhidden * pEI)
Ninhibit = Nhidden - Nexcite
tuning_matrix = stimulus.tuning_input.reshape((24, 24))
alpha = dt / tau

## Make hidden states (eigenvectors) #########################################
H = np.zeros((Nhidden, Nstim))
rollvec = np.zeros(Nstim)
rollvec[:overlap] = 1.
for i in range(Nstim): H[Nrule:(Nrule + Nstim), i] = np.roll(rollvec, i - int(overlap / 2))
for i in range(Nstim): H[i + Nexcite, i] = 1.

## Make Wrnn #################################################################
### Initialize
R = np.random.rand(Nhidden, Nhidden)  ## random backgrounds
R[:, Nexcite:] *= (-1.)
for i in range(Nhidden): R[i, i] = -1.

### Make excitatory block
stim_block = np.ones((Nstim, Nstim))
for i in range(Nstim): stim_block[i, i] = -1.
R[Nrule:(Nrule + Nstim), Nrule:(Nrule + Nstim)] = stim_block

### Make sub-excitatory block
R[Nexcite:, Nrule:(Nrule + Nstim)] = 1. / overlap

### Make inhibitory block
rollvec = -np.ones(Nstim) * overlap
rollvec[:overlap] = -overlap + 2.
for i in range(Nstim): R[(i + Nrule), Nexcite:(Nexcite + Nstim)] = np.roll(rollvec, i - int(overlap / 2))

### Make other blocks
R[Nexcite:, Nexcite:(Nexcite + Nstim)] = - np.ones((Ninhibit, Nstim))
R[(Nrule + Nstim):Nexcite, Nrule:(Nrule + Nstim)] = 1.
R[(Nrule + Nstim):Nexcite, Nexcite:(Nexcite + Nstim)] = -float(overlap)

### Make block corresponding to "rule neurons"
R[:Nrule, :] = 0
R[:, :Nrule] = 0
for i in range(Nrule): R[i, i] = -1.
R[0, 1] = 1.
R[1, 0] = 1.

### Make W_rnn
w_rnn = R + np.eye(Nhidden)

### Sanity check
#### 1. matrix rank
np.linalg.matrix_rank(R)
np.linalg.matrix_rank(w_rnn)

#### 2. make sure H(respond) are eigenvectors
abs((w_rnn @ H) - H).sum()

#### 3. make sure H(suppress) are eigenvectors
_H = copy.deepcopy(H)
_H[:Nrule, :] = 1
abs((w_rnn @ _H) - _H).sum()

## Make W_inin ##################################################################
### 1. Make truncated tuning input matrix ("Rule 1" is trivial, thus ruled out )
Resp_input_trunc = np.zeros(NruleIn - 1 + Nstim);
Resp_input_trunc[0] = 0.8
Resp_h = np.zeros((Nhidden, 1));
Resp_h[:NruleIn, :] = -1.
Htotal = np.concatenate((Resp_h, H * devi), axis=1)  # _H : retain, H : respond

tr_mat_tilde = np.concatenate((Resp_input_trunc.reshape((-1, 1)),
                               np.concatenate((np.zeros((1, Nstim)), tuning_matrix))), axis=1)
tr_mat_full = np.concatenate((0.8 * np.ones((1, NruleIn - 1 + Nstim)), tr_mat_tilde), axis=0)

### 2. Do inversion (very high condition number)
INV0 = np.linalg.inv(tuning_matrix)
OneZero = np.zeros((Nstim + 1, 1));
OneZero[0, 0] = 1 / 0.8
INV = np.concatenate((OneZero, np.concatenate((np.zeros((1, Nstim)), INV0))), axis=1)

### 3. Make W_in
w_in_tilde = Htotal @ INV
w_in = np.concatenate((np.zeros((Nhidden, 1)), w_in_tilde), axis=1)
abs(w_in_tilde @ tr_mat_tilde - Htotal).sum()  # check mapping works

## Make Wout #################################################################
w_out = np.zeros((Nhidden, Nout + NruleOut))
w_out[Nrule:(Nrule + Nstim), NruleOut:(NruleOut + Nstim)] = tuning_matrix
w_out[0, 0] = privg  ## output "suppression" signal corresponding to "rule neurons"

## Make var_dict #############################################################
var_dict = {}
h0 = np.zeros(Nhidden)
h0[:Nrule] = 1.  ## Note that h0 corresponds to initial suppress rule eigenvector
var_dict['h'] = h0
var_dict['w_in'] = w_in
var_dict['w_rnn'] = w_rnn
var_dict['w_out'] = w_out


## Define functions ##########################################################
def relu(x):
    return np.maximum(x, 0.)


def rnn_cell(h, rnn_input, var_dict=var_dict):
    h = relu((1. - alpha) * h + \
             alpha * (rnn_input @ var_dict['w_in'].T + h @ var_dict['w_rnn'].T) + \
             np.random.normal(size=Nhidden, loc=0, scale=2. * alpha * noise_rnn_sd))
    return h


def rnn_model(input_data, var_dict=var_dict):
    h = np.tile(var_dict['h'], (batch_size, 1))
    h_stack = [];
    y_stack = []
    *_input_data, = input_data

    for _iter, rnn_input in enumerate(_input_data):
        h = rnn_cell(h, rnn_input)
        h_stack.append(h)
        y_stack.append(h @ var_dict['w_out'])

    h_stack = np.stack(h_stack)
    y_stack = np.stack(y_stack)

    return h_stack, y_stack


## Run RNN ###################################################################
### Note that W_in has extremely high condition number
### Meaning that "stimulus noise" may show a snowball-like destructive behavior
### Therefore, it may not be safe to interpret our RNN is "sensitive to input noise"
### Sensitivity to stimulus noise may be solely due to W_in instability
### Therefore we should use par['noise_sd'] = 0
### Meanwhile, a fairer way to examine noise-sensitivity is "internal noise"
### Which is modulated by "noise_rnn_sd"

### Define "internal noise"
noise_rnn_sd = 0.005  ## Feel free to modulate it. As it gets larger, more instable RNN becomes.
par['noise_sd'] = 0

### Update parameters
par['design'].update({'iti': (0, 1.5),
                      'stim': (1.5, 3.0),
                      'delay': (3.0, 4.5),
                      'estim': (4.5, 9.0)})  # vanilla setting
# 'delay'   : (3.0,183.0),
# 'estim'   : (183.0,187.5)}) # 3min delay(less noise_rnn_sd recommended)
par = update_parameters(par)
stimulus = Stimulus(par)
trial_info = stimulus.generate_trial()

### Run model
Hrnn, Yrnn = rnn_model(trial_info['neural_input'])  # run model
Yrnn_normalize = Yrnn / (np.sum(Yrnn, axis=-1, keepdims=True) + np.finfo(float).eps)  # linear normalization

### Plot!
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
TEST_TRIAL = np.random.randint(par['batch_size'])
axes[0].imshow(trial_info['neural_input'][:, TEST_TRIAL, :].T, aspect='auto');
axes[0].set_title("Neural Input")
axes[1].imshow(trial_info['desired_output'][:, TEST_TRIAL, :].T, aspect='auto');
axes[1].set_title("Desired Output")
axes[2].imshow(Yrnn_normalize[:, TEST_TRIAL, :].T, aspect='auto', vmax=0.13)
fig.tight_layout(pad=2.0)
plt.show()

## Sanity check ###############################################################
### Visualize designed parameters
plt.imshow(w_in)  # w_rnn, w_out
plt.colorbar()
plt.show()

### "Active" output
plt.imshow((H.T @ w_out).T / np.sum(H.T @ w_out, axis=1))
plt.colorbar()
plt.show()

### "Suppressed" output
plt.imshow((_H.T @ w_out).T / np.sum(_H.T @ w_out, axis=1), vmax=0.0001)
plt.colorbar()
plt.show()

### Visualization of H: the population codes for each stimulus
for i in range(24):
    i_stim_input = tr_mat_full[:, i + 1]
    h = (w_in @ i_stim_input).reshape((1, -1)) @ var_dict['w_rnn'].T
    plt.plot(h.T)

### 1. Does ITI input induce invariance?
#### Yes, ITI input(=[1,0,...,0]) maps to [0,0,...,0]
#### Therefore, ITI period just preservers initial h0
ITI_input = np.zeros(NruleIn + Nstim);
ITI_input[0] = 0.8

plt.plot(ITI_input @ var_dict['w_in'].T)  ## note that ITI input |-> 0
plt.plot(rnn_cell(var_dict['h'], ITI_input))  ## preservation

h = rnn_cell(var_dict['h'], ITI_input)  # starts from h0
for i in range(2000):
    plt.plot(h)
    h = rnn_cell(h, ITI_input)
plt.show()

### 2. Does stimulus period induce gradual evidence accumulation?
#### Yes, note the growth of population code as a function of iteration
#### This growth speed is modulated by the parameter devi
stim_input = tr_mat_full[:, 1]
h = rnn_cell(h0, stim_input)
for i in range(150):
    plt.plot(h)
    h = rnn_cell(h, stim_input)
plt.show()

### 3. Does the delay period stably maintain information?
#### Yes, seems delay input(=[1,0,...,0], same as ITI input) maps to [0,...,0]
#### Thus, there is no additive information during ITI
h = rnn_cell(h, ITI_input)  # inherits h from 2.
for i in range(800):
    plt.plot(h)
    h = rnn_cell(h, ITI_input)
plt.show()

### 4. Does the response period evoke response?
#### Yes, Response signal input(=[1,1,0,...,0]) maps to [-1,-1,0,...,0]
#### This diminishes 1st and 2nd component of h (rule neurons)
#### However, there is lower bound of 0 due to ReLU
#### In this sense, Resp_input just exploits ReLU
#### Therefore, during response period, first and second component of h remains 0
#### This "releases" internal response
#### However, this RNN is not flexible enough to "re-suppress" response when response cue is nonpresent
Resp_input = np.zeros(NruleIn + Nstim);
Resp_input[:NruleIn] = 0.8
h = rnn_cell(h, Resp_input)  # inherits h from 3.

plt.plot(Resp_input @ var_dict['w_in'].T)  ## note that Resp_input |-> [-1,-1,0,...,0]

for i in range(800):
    plt.plot(h.reshape((-1,)))
    h = rnn_cell(h, Resp_input)
plt.show()

