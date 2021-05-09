import sys
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as mc

sys.path.append('../')
from det_rnn import *
import det_rnn.train as dt
import det_rnn.analysis as da

# ========================
# Preparation
# ========================

## Load model (shorter version)
model_dir = "/Volumes/ROOT/CSNL_temp/HG/Analysis/RNN/modular/networks/short_separate/sigmoid/network00"
model     = tf.saved_model.load(model_dir)

par['design'].update({'iti'  : (0, 0.3),
                      'stim' : (0.3, 0.6),                      
                      'decision': (0.9, 1.2),
                      'delay'   : ((0.6, 0.9),(1.2, 1.5)),
                      'estim' : (1.5, 1.8)})

## parameter adjustment 
par['dt']             = 20
dt.hp['dt']           = 20
dt.hp['gain']         = 1e-2  # amplitude of random initialization (makes dynamics chaotic) 
dt.hp['w_out_dm_fix'] = True  # assume linear voting from two separate populations
dt.hp['DtoE_off']     = True  # for simplicity

dt.hp                 = dt.update_hp(dt.hp)
par                   = update_parameters(par)
stimulus              = Stimulus()
ti_spec               = dt.gen_ti_spec(stimulus.generate_trial())
trial_info            = dt.tensorize_trial(stimulus.generate_trial())

## numpy version of forward pass
def rnn_model(input_data1, input_data2, hp, var_dict, with_noise):
    _h1 = np.zeros((neural_input1.shape[1], 50))
    _h2 = np.zeros((neural_input2.shape[1], 50))
    h1_stack    = []
    h2_stack    = []
    y_dm_stack  = []
    y_em_stack  = []
    for i_t in range(len(input_data1)):
        rnn_input1  = input_data1[i_t,:,:]
        rnn_input2  = input_data2[i_t,:,:]
        _h1, _h2    = rnn_cell(_h1, _h2, rnn_input1, rnn_input2, hp=hp, var_dict=var_dict, with_noise=with_noise)
        h1_stack.append(_h1.astype(np.float32))
        h2_stack.append(_h2.astype(np.float32))
        y_dm_stack.append((_h1.astype(np.float32) @ hp['w_out_dm']).astype(np.float32))
        y_em_stack.append((_h2.astype(np.float32) @ var_dict['w_out_em']).astype(np.float32))
        
    h1_stack   = np.stack(h1_stack) 
    h2_stack   = np.stack(h2_stack) 
    y_dm_stack = np.stack(y_dm_stack) 
    y_em_stack = np.stack(y_em_stack) 
    return y_dm_stack, y_em_stack, h1_stack, h2_stack

def rnn_cell(_h1, _h2, rnn_input1, rnn_input2, hp, var_dict, with_noise=False):

    _h1 = _h1.astype(np.float32) * (1. - hp['alpha_neuron1']) \
        + hp['alpha_neuron1'] * tf.nn.sigmoid(rnn_input1 @ var_dict['w_in1'] \
            + _h1.astype(np.float32) @ var_dict['w_rnn11'] + _h2.astype(np.float32) @ var_dict['w_rnn21'] \
            + with_noise*np.random.normal(size=_h1.shape, loc=0, scale=np.sqrt(2*hp['alpha_neuron1'])*hp['noise_rnn_sd'])).numpy()

    _h2 = _h2.astype(np.float32) * (1. - hp['alpha_neuron2']) \
        + hp['alpha_neuron2'] * tf.nn.sigmoid(rnn_input2 @ var_dict['w_in2'] \
            + _h2.astype(np.float32) @ var_dict['w_rnn22'] \
            + with_noise*np.random.normal(size=_h2.shape, loc=0, scale=np.sqrt(2*hp['alpha_neuron2'])*hp['noise_rnn_sd'])).numpy()

    return _h1, _h2

## inherit trained/hyper-parameters 
var_dict = {k:model.var_dict[k].numpy() for k in model.var_dict}
hp       = {k:dt.hp[k].numpy() for k in dt.hp if tf.is_tensor(dt.hp[k])}
hp['alpha_neuron1'] = dt.hp['alpha_neuron1']
hp['alpha_neuron2'] = dt.hp['alpha_neuron2']
hp['noise_rnn_sd']  = dt.hp['noise_rnn_sd']

## check model responses 
neural_input1      = trial_info['neural_input1'].numpy().astype(np.float32)
neural_input2      = trial_info['neural_input2'].numpy().astype(np.float32)
Y_DM, Y_EM, H1, H2 = rnn_model(neural_input1, neural_input2, hp, var_dict, with_noise=False)
pred_output_DM     = da.softmax_pred_output(Y_DM)
pred_output_EM     = da.softmax_pred_output(Y_EM)

TEST_TRIAL = np.random.randint(par['batch_size'])
fig, axes = plt.subplots(8,1, figsize=(10,12))
axes[0].imshow(neural_input1[:,TEST_TRIAL,:].T, aspect='auto'); axes[0].set_title("Neural Input")
axes[1].imshow(neural_input2[:,TEST_TRIAL,:].T, aspect='auto'); axes[1].set_title("Neural Input")
axes[2].imshow(trial_info['desired_decision'][:,TEST_TRIAL,:].numpy().T, aspect='auto', interpolation='none'); axes[2].set_title("Desired Decision")
axes[3].imshow(trial_info['desired_estim'][:,TEST_TRIAL,:].numpy().T, aspect='auto'); axes[3].set_title("Desired Estimation")
axes[4].imshow(pred_output_DM[:,TEST_TRIAL,:].T,  aspect='auto', interpolation='none'); axes[4].set_title("Predicted Decision")
axes[5].imshow(pred_output_EM[:,TEST_TRIAL,:].T,  aspect='auto', vmin=0, vmax=0.15); axes[5].set_title("Predicted Estimation")
axes[6].imshow(H1[:,TEST_TRIAL,:].T,  aspect='auto', vmin=0, vmax=3.); axes[6].set_title("H1")
axes[7].imshow(H2[:,TEST_TRIAL,:].T,  aspect='auto', vmin=0, vmax=3.); axes[7].set_title("H2")
fig.tight_layout(pad=2.0)
plt.show()




# ========================
# Analysis
# ========================
## stimulate every case (without noise)
down_trial          = 1
stim_list           = np.arange(24)
ref_list            = np.array([-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11])
ds_timesteps        = np.arange(par['n_timesteps'])
par['reference']    = ref_list
par                 = update_parameters(par)
dm_output_total     = np.zeros((len(par['design_rg']['decision']), 24, 23, 2)) # save only a single point
em_output_total     = np.zeros((len(par['design_rg']['estim']),    24, 23, 24))
stimulus_ori_total  = np.zeros((24, 23))
reference_ori_total = np.zeros((24, 23))
H1_total            = np.zeros((par['n_timesteps'], 24, 23, par['n_hidden1']))
H2_total            = np.zeros((par['n_timesteps'], 24, 23, par['n_hidden2']))

for i_s, st in enumerate(stim_list):
    stim_dist = np.zeros(24); stim_dist[i_s] = 1.;    par['stim_dist'] = stim_dist
    for i_r, re in enumerate(ref_list):
        ref_dist   = np.zeros(23); ref_dist[i_r] = 1.; par['ref_dist']  = ref_dist
        par        = update_parameters(par); stimulus = Stimulus(par)
        trial_info = dt.tensorize_trial(stimulus.generate_trial())
        neural_input1 = trial_info['neural_input1'].numpy().astype(np.float32)
        neural_input2 = trial_info['neural_input2'].numpy().astype(np.float32)
        pred_output_dm, pred_output_em, H1, H2  = rnn_model(neural_input1, neural_input2, hp, var_dict, False)

        dm_output_total[:, i_s, i_r, :]  = pred_output_dm[par['design_rg']['decision'],0,:]
        em_output_total[:, i_s, i_r, :]  = pred_output_em[par['design_rg']['estim'],0,:]
        stimulus_ori_total[i_s, i_r]     = trial_info['stimulus_ori'].numpy()[0]
        reference_ori_total[i_s, i_r]    = trial_info['reference_ori'].numpy()[0]
        H1_total[:, i_s, i_r, :]         = H1[:,0,:][ds_timesteps,:]
        H2_total[:, i_s, i_r, :]         = H2[:,0,:][ds_timesteps,:]
    print('stimulus', st)

par['stim_dist'] = np.ones(shape=(24,))
par['ref_dist']  = np.ones(shape=(23,))
par              = update_parameters(par); stimulus = Stimulus(par)
trial_info       = dt.tensorize_trial(stimulus.generate_trial())

## sanity check: decision-making evidence as a function of relative reference 
cw = mc.ListedColormap(sns.color_palette("coolwarm",23))
fig = plt.figure(figsize=(8, 5))
for stim_idx in range(24):
    for i_ref in range(23):
        CW_mean  = np.mean(dm_output_total[:,stim_idx,i_ref,0],axis=0)
        CCW_mean = np.mean(dm_output_total[:,stim_idx,i_ref,1],axis=0)
        plt.scatter((i_ref-11)*7.5, CCW_mean-CW_mean, color=cw.colors[i_ref])   
plt.xlabel("Relative Reference(deg)"); plt.ylabel("Evidence CCW")
plt.show()


## input-driven stimulus and reference representations
tuning_input = stimulus.tuning_input[:,0,:]
ref_input    = np.eye(len(tuning_input)) * stimulus.strength_ref
ref_input_vector  = ref_input @ var_dict['w_in1']
stim_input_vector = tuning_input @ var_dict['w_in2']

fig, ax = plt.subplots(1,2,figsize=(13,6))
ax[0].imshow(ref_input_vector[:,np.argsort(np.argmax(ref_input_vector,axis=0))])
ax[1].imshow(stim_input_vector[:,np.argsort(np.argmax(stim_input_vector,axis=0))])
for i in range(2):
    ax[i].set_title("Input driven representation of "+["reference", "stimulus"][i] + " vectors")
    ax[i].set_xlabel("Module "+str(i+1) + "(" + ["DM", "EM"][i]  + ")" + " neuron label(sorted by tuning)")
    ax[i].set_ylabel("Tuning center(deg)")
    ax[i].set_yticks([0,0,5,10,15,20])
    ax[i].set_yticklabels(np.round(np.array([0,0,5,10,15,20])*180/24,1))
plt.show()


## check two flanking templates
any_ref  = 0   # does not matter, any refererence
pre_DM   = 44  # one step before the reference onset
from_H2  = H2_total[pre_DM,:,any_ref,:] @ var_dict['w_rnn21']
from_ref = ref_input_vector

# sort by reference tuning; within each readout-corresponding population
ref_sort = np.argsort(np.argmax(ref_input_vector,axis=0)+np.repeat([0,25],25))
plt.figure(figsize=(9,4))
plt.imshow(from_H2[:,ref_sort])
plt.xlabel("Module 1(DM) neuron label (sorted within population)")
plt.ylabel("Tuning center(deg)")
plt.yticks([0,0,5,10,15,20],np.round(np.array([0,0,5,10,15,20])*180/24,1))
plt.colorbar()
plt.show()


# check template-matching with flankers 
fig, ax = plt.subplots(2,1,figsize=(9,8))
im0 = ax[0].imshow((from_H2+np.roll(from_ref,-6,axis=0))[:,ref_sort])
im1 = ax[1].imshow((from_H2+np.roll(from_ref,6, axis=0))[:,ref_sort])

ax[0].set_title("When reference input is -45$^\circ$")
ax[1].set_title("When reference input is +45$^\circ$")

for i in range(2):
    plt.colorbar([im0,im1][i], ax=ax[i])
    plt.xlabel("Module 1(DM) neuron label (sorted within population)")
    ax[i].set_ylabel("Tuning center(deg)")
    ax[i].set_yticks([0,0,5,10,15,20])
    ax[i].set_yticklabels(np.round(np.array([0,0,5,10,15,20])*180/24,1))

plt.tight_layout()
plt.show()
