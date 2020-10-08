import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append('../')
from det_rnn import *
import det_rnn.train as dt
import det_rnn.analysis as da

# Rational account of cardinal bias: RNNs trained with discrimination cost
discrim_template = np.kron(np.eye(2), np.ones(12)).T

def GirshickPrior(theta, radian=True):
    if radian:
        p_theta = (2. - np.abs(np.sin(theta * 2.))) / (2. * (np.pi - 1.))
    else: # degree
        p_theta = (2. - np.abs(np.sin(theta * np.pi/90. ))) / (360./np.pi * (np.pi - 1.))
    return p_theta

def desired_discrim(theta, power=10):
    # arr = np.zeros((12,1)); arr[0] = 1. # "infinity" case
    arr = (GirshickPrior(np.arange(0., 24., step=2.) / 48. * np.pi)[:, np.newaxis] + .5) ** power
    res = arr * np.roll(discrim_template, int(13 + theta), axis=0)[:12]
    return res

def decisionize(trial_info, par=par):
    decision_output = np.zeros(trial_info['desired_estim'].shape[:2] + (12, 3))
    decision_output[:,:,:,0] = 1. # pad rule component
    decision_output[par['design_rg']['estim'],:,:,0] = 0
    for i,s in enumerate(trial_info['stimulus_ori']):
        decision_output[par['design_rg']['estim'],i,:,1:] = desired_discrim(s)
    return decision_output

def decisionize_ti(trial_info, par=par):
    trial_info['desired_output_it'] = decisionize(trial_info, par=par)
    trial_info['mask_it'] = trial_info['mask'][:,:,:int(par['n_output']/2),np.newaxis]
    return trial_info


# Analysis

## 0. Load model (Compare the following)
load_model = tf.saved_model.load("/Volumes/Data_CSNL/project/RNN_study/20-07-24/HG/alternate_rnns/rnn1")
# load_model = tf.saved_model.load("/Volumes/Data_CSNL/project/RNN_study/20-07-24/HG/alternate_rnns/rnn2")
# load_model = tf.saved_model.load("/Volumes/Data_CSNL/project/RNN_study/20-07-24/HG/alternate_rnns/rnn3")

## 1. Sanity check
par = update_parameters(par)
stimulus = Stimulus(par)
trial_info = dt.tensorize_trial(decisionize_ti(stimulus.generate_trial()))

### Performance
plt.plot(load_model.model_performance['perf'].numpy())
plt.show()

### Wrnn
plt.imshow((load_model.var_dict['w_rnn'] * dt.hp['EI_mask']).numpy().T)
plt.colorbar(); plt.show()

### Wout_discrim 
plt.imshow(load_model.var_dict['w_out_int'].numpy().transpose(1,0,2)[:,:,::-1])
plt.show()

### Wout_estim
plt.imshow(load_model.var_dict['w_out_est'].numpy().T)
plt.colorbar(); plt.show()

### Estimation Figure 
dt.hp['task_type'] = 1
pred_output, H, _, _ = load_model.rnn_model(trial_info['neural_input'], dt.hp)
pred_output = da.softmax_pred_output(pred_output)

fig, axes = plt.subplots(3,1, figsize=(10,8))
TEST_TRIAL = np.random.randint(par['batch_size'])
axes[0].imshow(trial_info['neural_input'][:,TEST_TRIAL,:].numpy().T, aspect='auto'); axes[0].set_title("Neural Input")
axes[1].imshow(trial_info['desired_estim'][:,TEST_TRIAL,:].numpy().T, aspect='auto'); axes[1].set_title("Desired Output")
axes[2].imshow(pred_output[:,TEST_TRIAL,:].T,  aspect='auto', vmin=0, vmax=0.15)
fig.tight_layout(pad=2.0)
plt.show()

### Discrimination Figure 
dt.hp['task_type'] = 0
pred_output, H, _, _ = load_model.rnn_model(trial_info['neural_input'], dt.hp)
pred_output = da.softmax_pred_output(pred_output)

fig, axes = plt.subplots(3,1, figsize=(10,8))
TEST_TRIAL = np.random.randint(par['batch_size'])
axes[0].imshow(trial_info['neural_input'][:,TEST_TRIAL,:].numpy().T, aspect='auto'); axes[0].set_title("Neural Input")
axes[1].imshow(trial_info['desired_output_it'][:,TEST_TRIAL,:,::-1].numpy().transpose((1,0,2)), aspect='auto'); axes[1].set_title("Desired Output")
axes[2].imshow(pred_output[:,TEST_TRIAL,:,::-1].transpose((1,0,2)),  aspect='auto')
fig.tight_layout(pad=2.0)
plt.show()


## 2. Behavior analysis
dt.hp['task_type'] = 1

### Accumulate 128 pred_outputs for each stimulus (128 x 24 in total)
### Accumulate 10 Hs for each stimulus (10 x 24 in total)
stim_list   = np.arange(24)
pred_output_total  = np.zeros((par['n_timesteps'], 128 * 24, 25))
stimulus_ori_total = np.zeros((128*24,))
H_total = np.zeros((par['n_timesteps'], 10 * 24, par['n_hidden']))
for i_s, st in enumerate(stim_list):
    stim_dist = np.zeros(24); stim_dist[i_s] = 1.
    par['stim_dist'] = stim_dist
    par = update_parameters(par)
    stimulus = Stimulus(par)
    trial_info = dt.tensorize_trial(decisionize_ti(stimulus.generate_trial()))
    pred_output, H, _, _  = load_model.rnn_model(trial_info['neural_input'], dt.hp)
    pred_output_total[:,(128*i_s):(128*(i_s+1)),:] = pred_output.numpy()
    stimulus_ori_total[(128*i_s):(128*(i_s+1))] = trial_info['stimulus_ori'].numpy()
    H_total[:,(10*i_s):(10*(i_s+1)),:] = H.numpy()[:,:10,:]

### Behavior Figure
ground_truth, estim_mean, raw_error, beh_perf = \
    da.behavior_summary({'stimulus_ori': stimulus_ori_total}, pred_output_total, par=par)
da.behavior_figure(ground_truth, estim_mean, raw_error, beh_perf)


## 3. Neural analysis
fig, ax = plt.subplots(1,4, figsize=(15,5))

ax[0].imshow(np.corrcoef(H_total[100,:,:])); ax[0].set_title("ITI")
ax[1].imshow(np.corrcoef(H_total[200,:,:])); ax[1].set_title("Stimulus")
ax[2].imshow(np.corrcoef(H_total[400,:,:])); ax[2].set_title("Delay")
ax[3].imshow(np.corrcoef(H_total[500,:,:])); ax[3].set_title("Estimation")

plt.tight_layout(pad=2.0)
plt.show()





