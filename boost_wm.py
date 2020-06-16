#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 19:25:51 2020

@author: hyunwoogu
"""
# Package
import re, os, sys, pickle
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

#
sys.path.append('/Users/hyunwoogu/Dropbox/CSNL/Projects/RNN/neuroRNN/rnn/det_rnn/')
from det_rnn import *

#%% 
with open('/Volumes/Data_CSNL/project/RNN_study/20-06-19/HG/output/HL_Masse_mask_moderate.pkl','rb') as f:
    boost_model = pickle.load(f)

#
par = update_parameters(par)
stimulus    = Stimulus(par)
trial_info  = stimulus.generate_trial()
pred_output, H = boost_model.rnn_model(trial_info['neural_input'])

plt.plot(boost_model.model_performance['loss'])
plt.show()
plt.plot(boost_model.model_performance['perf'])
plt.show()

#%%
ground_truth, estim_mean, raw_error, beh_perf = behavior_summary(trial_info, pred_output, par=par)
behavior_figure(ground_truth, estim_mean, raw_error, beh_perf)

# How to boost memory? 
boost_model.syn_x_init.shape


#%% Noise corr
H = H.numpy()
extended_perf_HL_Masse_mask.pkl


#%%
base_path  = "/Volumes/Data_CSNL/project/RNN_study/20-06-19/HG/boost_wm/"
model_code = "HL_booster1_resume"

model_list = os.listdir(base_path+"HL_booster1")
model_list = [m for m in model_list if m.endswith(".pkl")]
latest = base_path+"HL_booster1/"+sorted(model_list)[-1]

with open(latest,'rb') as f:
    childrnn = pickle.load(f)


par['design'].update({'iti'     : (0, 1.5),
                      'stim'    : (1.5,3.0),
                      'delay'   : (3.0,4.5),
                      'estim'   : (4.5,6.0)})

par = update_parameters(par)
stimulus = Stimulus()
pred_output, H = childrnn.rnn_model(trial_info['neural_input'])

plt.plot(childrnn.model_performance['loss'])
plt.show()
plt.plot(childrnn.model_performance['perf'])
plt.show()

ground_truth, estim_mean, raw_error, beh_perf = behavior_summary(trial_info, pred_output, par=par)
behavior_figure(ground_truth, estim_mean, raw_error, beh_perf)

childrnn.optimizer.weights


#%% Performance extension #####################################################
# N_iter      = 30
# extend_time = np.arange(0,17.0,step=0.5)
# perf_res    = np.empty((N_iter*len(extend_time),)) * np.nan

# for i_t, et in enumerate(extend_time):
#     for i_iter in range(N_iter):
        
#         par['design'].update({'iti'     : (0,  1.5),
#                               'stim'    : (1.5,3.0),
#                               'delay'   : (3.0,4.5+et),
#                               'estim'   : (4.5+et,6.0+et)})
#         par = update_parameters(par)
#         stimulus    = Stimulus(par)
#         trial_info  = stimulus.generate_trial()
#         pred_output, _ = boost_model.rnn_model(trial_info['neural_input'])
#         _, _, _, beh_perf = behavior_summary(trial_info, pred_output, par=par)
#         perf_res[int(i_t*N_iter+i_iter)] = np.mean(beh_perf)
#     print(et, "complete")

# res_DF = pd.DataFrame({'ExtendedTime': np.repeat(extend_time, N_iter),
#                         'Performance' : perf_res})
# res_DF.to_pickle("/Volumes/Data_CSNL/project/RNN_study/20-06-19/HG/analysis/extended_perf_HL_Masse_mask_moderate.pkl")

res_DF = pd.read_pickle("/Volumes/Data_CSNL/project/RNN_study/20-06-19/HG/analysis/extended_perf_HL_Masse_mask_moderate.pkl")
sns.lineplot(x="ExtendedTime", y="Performance", data=res_DF)
plt.ylim([-0.1,1.0]); plt.show()

###############################################################################


#%%




