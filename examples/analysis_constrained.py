import sys
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import kde

sys.path.append('../')
from det_rnn import *
import det_rnn.train as dt
import det_rnn.analysis as da

# ========================
# Preparation
# ========================
## Load model (shorter version)
model_dir = "/Volumes/Data_CSNL/project/RNN_study/21-05-27/constrained_networks/network00"
model     = tf.saved_model.load(model_dir)

par['design'].update({'iti'  : (0, 0.3),
                      'stim' : (0.3, 0.6),                      
                      'decision': (0.9, 1.2),
                      'delay'   : ((0.6, 0.9),(1.2, 1.5)),
                      'estim' : (1.5, 1.8)})

## parameter adjustment 
par['dt']             = 20
dt.hp['dt']           = 20
dt.hp['gain']         = 0  # 1e-2 : amplitude of random initialization (makes dynamics chaotic) 

## contrain the network(fixing some of the weights)
dt.hp['w_in_dm_fix']  = True  
dt.hp['w_in_em_fix']  = True  

dt.hp['w_rnn11_fix']  = True  # fix DM-to-DM recurrent matrix
dt.hp['w_rnn21_fix']  = True  # fix EM-to-DM recurrent matrix
dt.hp['w_rnn22_fix']  = False # fix EM-to-EM recurrent matrix

dt.hp['w_out_dm_fix'] = True  # fix linear voting from two separate populations
dt.hp['w_out_em_fix'] = True  # fix circular voting from two separate populations

dt.hp['EtoD_off']     = False # True: set W21=0, False: train W21
dt.hp['DtoE_off']     = False # True: set W12=0, False: train W12 # THIS IS IMPORTANT

dt.hp                 = dt.update_hp(dt.hp) # Do not forget this line
par                   = update_parameters(par)
stimulus              = Stimulus()
ti_spec               = dt.gen_ti_spec(stimulus.generate_trial())
trial_info            = dt.tensorize_trial(stimulus.generate_trial())


# ================================================================
# Check Bimodality
# ================================================================ 

## 1. Bump center density as a function of reference duration
DM_lens    = np.arange(3.0,step=0.15)
N_repeat   = 5
refs       = [-4,-3,-2,-1,0,1,2,3,4]
idx_ref0   = int(len(refs)/2)
errors     = np.zeros((len(DM_lens), len(refs), N_repeat * 128)) * np.nan
dms        = np.zeros((len(DM_lens), len(refs), N_repeat * 128)) * np.nan

for i_dm, dm_len in enumerate(DM_lens):
    par['design'].update({'iti'     : (0, 0.3),
                          'stim'    : (0.3, 0.6),                      
                          'decision': (0.9, 0.9+dm_len), # 0.3s to 3s
                          'delay'   : ((0.6, 0.9),(0.9+dm_len, 1.2+dm_len)), # 0.3 to 3s
                          'estim'   : (1.2+dm_len, 1.5+dm_len)})

    # 
    for i_ref, _ in enumerate(refs):
        par['reference']  = np.arange(-4,4+1)
        ref_dist          = np.zeros(len(par['reference']))
        ref_dist[i_ref]   = 1.
        # ref_dist[3:6]     = 1. 
        par['ref_dist']   = ref_dist
        par               = update_parameters(par)
        stimulus          = Stimulus(par)

        for i_repeat in range(N_repeat):
            trial_info        = dt.tensorize_trial(stimulus.generate_trial())
            pred_output_dm, pred_output_em, H1, H2  = model.rnn_model(trial_info['neural_input1'], trial_info['neural_input2'], dt.hp)
            stim      = trial_info['stimulus_ori'].numpy()
            output_dm = pred_output_dm.numpy()[par['design_rg']['decision'],:,:]
            output_em = pred_output_em.numpy()[par['design_rg']['estim'],:,:]
            dv_L, dv_R, choice                        = da.behavior_summary_dm(output_dm, par=par)
            ground_truth, estim_mean, error, beh_perf = da.behavior_summary_em({'stimulus_ori': stim}, output_em, par=par)

            errors[i_dm, i_ref, (i_repeat*128):((i_repeat+1)*128)]  = error - np.pi/24/2
            dms[i_dm, i_ref, (i_repeat*128):((i_repeat+1)*128)]     = choice * 1

    print('simulate reference duration: ', dm_len, '(sec)')

## plot density
x,y     = np.repeat(DM_lens, 128*N_repeat*3), errors[:,(idx_ref0-1):(idx_ref0+2),:].reshape((-1,))*180/np.pi
rg      = 45
xmin, xmax, ymin, ymax = x.min(), x.max(), -rg, rg
nbins   = 80
nlabels = 6
k       = kde.gaussian_kde([x,y])
xi, yi  = np.mgrid[xmin:xmax:nbins*1j, ymin:ymax:nbins*1j]
zi      = k(np.vstack([xi.flatten(), yi.flatten()]))

plt.figure(figsize=(10,3))
plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap=plt.cm.gray)
plt.xlabel('Reference Duration')
plt.ylabel('Estimation Error')
plt.ylim([-rg,rg]) # for aesthetics
plt.show()


## 2. Bias plot as a function of reference
DF = pd.concat([pd.DataFrame({'Reference': ref,
                              'Error'     : errors[:,i_r,:].reshape((-1,)), 
                              'Duration'  : np.round(np.repeat(DM_lens, errors.shape[-1]),1)}) \
                                  for i_r, ref in enumerate(refs)])

DF['AbsRef'] = np.abs(DF['Reference']) * 7.5
DF['Bias']   = -np.sign(DF['Reference']) * DF['Error'] * 180/np.pi

plt.figure(figsize=[8,5.5])
sns.pointplot(x='AbsRef', y='Bias', hue='Duration', data=DF[DF.AbsRef > 0])
plt.legend(title='Reference Duration(sec)', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('Absolute Reference(deg)')
plt.ylabel('Bias(deg')
plt.show()


## 3. Density: within-trial bump pattern as a function of reference duration(this is "safer" version)
DM_lens    = np.arange(3.0,step=0.15)
N_repeat   = 1
refs       = [-4,-3,-2,-1,0,1,2,3,4]
idx_ref0   = int(len(refs)/2)

# for i_dm, dm_len in enumerate(DM_lens):
dm_len = DM_lens[-1]
par['design'].update({'iti'     : (0, 0.3),
                      'stim'    : (0.3, 0.6),                      
                      'decision': (0.9, 0.9+dm_len), # 0.3s to 3s
                      'delay'   : ((0.6, 0.9),(0.9+dm_len, 1.2+dm_len)), # 0.3 to 3s
                      'estim'   : (1.2+dm_len, 1.5+dm_len)})

for i_ref, _ in enumerate(refs):
    par['reference']  = np.arange(-4,4+1)
    ref_dist          = np.zeros(len(par['reference']))
    ref_dist[i_ref]   = 1.
    # ref_dist[3:6]     = 1. 
    par['ref_dist']   = ref_dist
    par               = update_parameters(par)
    stimulus          = Stimulus(par)

    for i_repeat in range(N_repeat):
        trial_info        = dt.tensorize_trial(stimulus.generate_trial())
        pred_output_dm, pred_output_em, H1, H2  = model.rnn_model(trial_info['neural_input1'], trial_info['neural_input2'], dt.hp)
        stim      = trial_info['stimulus_ori'].numpy()
        pred_output_EM = da.softmax_pred_output(pred_output_em)
        pred_output_EM_roll = np.zeros_like(pred_output_EM)
        for i_trial in range(pred_output_EM_roll.shape[1]):
            pred_output_EM_roll[:,i_trial,:] = np.roll(pred_output_EM[:,i_trial,:], 12-stim[i_trial], axis=-1)

        if (i_ref == 0) & (i_repeat == 0):
            pred_output_EM_rolls = np.zeros((len(refs), N_repeat) + pred_output_EM_roll.shape)
        
        pred_output_EM_rolls[i_ref, i_repeat, :, :, :] = pred_output_EM_roll

plt.figure(figsize=[10,3])
plt.imshow(np.mean(pred_output_EM_rolls[np.array([3,5]),0,:,:,:],axis=(0,2)).T, aspect='auto', clim=[0,0.15], cmap='gray')
plt.colorbar()



