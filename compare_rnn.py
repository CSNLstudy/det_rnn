#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:34:23 2020

@author: hyunwoogu
"""

## Package
import re, os, sys, pickle
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

##
sys.path.append('/Users/hyunwoogu/Dropbox/CSNL/Projects/RNN/neuroRNN/rnn/det_rnn/')
from det_rnn import *

## 
output_path = "/Volumes/Data_CSNL/project/RNN_study/20-06-19/HG/output/"
HL   = os.listdir(output_path)
print(HL)

## Functions
def circ_mean(x,p):
    # Assumes wrap-around period = pi(rad)
    sinr = np.sum(np.sin(2*x) * p)
    cosr = np.sum(np.cos(2*x) * p)
    return np.arctan2(sinr, cosr)/2

def cos_perf(pred, true):
    return(np.cos(2.*(pred-true)))

## Load models 
models = {}
for i,m in enumerate(HL):
    with open(output_path + m, 'rb') as f:
        models[re.sub('.pkl', '', m)] = pickle.load(f)

for m,v in models.items():
    plt.plot(v.model_performance['iteration'], v.model_performance['loss'], label=m)
plt.xlim((0,500))
plt.legend(loc='upper right')
plt.show()

def behavior_summary(trial_info, pred_output, par=par):
    # softmax the pred_output
    sout = np.sum(pred_output[0], axis=2)
    sout = np.expand_dims(sout, axis=2)
    noutput = pred_output[0] / np.repeat(sout,par['n_output'],axis=2)
    cenoutput = tf.nn.softmax(pred_output[0], axis=2)
    cenoutput = cenoutput.numpy()
    
    # posterior mean as a function of time
    post_prob = cenoutput[:,:,par['n_rule_output']:]
    post_prob = post_prob/(np.sum(post_prob, axis=2, keepdims=True)+np.finfo(np.float32).eps) # Dirichlet normaliation
    post_support = np.linspace(0,np.pi,par['n_ori'],endpoint=False) + np.pi/par['n_ori']/2
    post_sinr = np.sin(2*post_support)
    post_cosr = np.cos(2*post_support)
    pseudo_mean = np.arctan2(post_prob @ post_sinr, post_prob @ post_cosr)/2
    
    # posterior mean collapsed along time
    estim_sinr = (np.sin(2*pseudo_mean[par['design_rg']['estim'],:])).mean(axis=0)
    estim_cosr = (np.cos(2*pseudo_mean[par['design_rg']['estim'],:])).mean(axis=0)
    estim_mean = np.arctan2(estim_sinr, estim_cosr)/2
    
    ## Quantities for plotting
    ground_truth  = trial_info['stimulus_ori']
    ground_truth  = ground_truth * np.pi/par['n_ori']
    raw_error = estim_mean - ground_truth
    beh_perf  = cos_perf(ground_truth, estim_mean)

    return ground_truth, estim_mean, raw_error, beh_perf

def behavior_figure(ground_truth, estim_mean, raw_error, beh_perf):
    cos_supp  = np.linspace(0,np.pi,1000)
    fig, ax = plt.subplots(3,2, figsize=[15,10])
    plt.subplots_adjust(hspace=0.4)
    
    ax[0,0].hist(beh_perf, bins=30); ax[0,0].set_title("Performance Distribution")
    ax[0,0].axvline(x=0, color='r', linestyle='--', label="Chance Level")
    ax[0,0].set_xlabel("Performance"); ax[0,0].set_ylabel("Count"); ax[0,0].legend()
    
    sns.scatterplot(x='GT', y='Perf', data=pd.DataFrame({'GT':ground_truth, 'Perf':beh_perf}), ax=ax[0,1])
    ax[0,1].set_title("Performance as a function of ground truth")
    ax[0,1].axhline(y=0, color='r', linestyle='--', label="Chance Level")
    ax[0,1].set_xlabel(r"$\theta$(rad)"); ax[0,1].set_ylabel("Performance"); ax[0,1].legend()
    
    ax[1,0].plot(cos_supp,np.cos(cos_supp*2), linewidth=2, color='darkgreen', label=r"$\cos(2\theta)$")
    ax[1,0].set_title("Estimation(Cosine Overlay)")
    sns.scatterplot(x='GT', y='CosBehav', data=pd.DataFrame({'GT':ground_truth, 'CosBehav':np.cos(estim_mean*2)}), ax=ax[1,0])
    ax[1,0].set_xlabel(r"$\theta$(rad)"); ax[1,0].set_ylabel(r"$\cos(2\hat{\theta}$)"); ax[1,0].legend()
    
    ax[1,1].set_title("Estimation(Cosine Transformed)")
    sns.regplot(x='GT', y='CosBehav', data=pd.DataFrame({'GT':np.cos(ground_truth*2), 'CosBehav':np.cos(estim_mean*2)}), ax=ax[1,1])
    ax[1,1].set_xlabel(r"$\cos(2\theta$)"); ax[1,1].set_ylabel(r"$\cos(2\hat{\theta}$)"); ax[1,1].legend()
    
    ax[2,0].set_title("Error Distribution")
    sns.scatterplot(x='GT', y='Error', data=pd.DataFrame({'GT':ground_truth, 'Error':np.arcsin(np.sin(2*raw_error))}), ax=ax[2,0])
    ax[2,0].set_xlabel(r"$\theta$(rad)"); ax[2,0].set_ylabel(r"$\hat{\theta} - \theta$"); plt.legend()
    
    ax[2,1].set_title("Estimation Distribution")
    ax[2,1].hist(estim_mean%(np.pi), bins=30)
    ax[2,1].set_xlabel(r"$\theta$(rad)"); ax[2,1].set_ylabel("Count"); ax[2,1].legend()
    
    plt.show()



## Prediction  ##########################################################################
par = update_parameters(par)
stimulus    = Stimulus(par)
trial_info  = stimulus.generate_trial()
pred_output = models['HL_Masse_mask'].rnn_model(trial_info['neural_input'])

## Quantities for plotting
ground_truth, estim_mean, raw_error, beh_perf = behavior_summary(trial_info, pred_output, par=par)
behavior_figure(ground_truth, estim_mean, raw_error, beh_perf)




## Performance extension ################################################################
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
#         pred_output = models['HL_Masse_mask'].rnn_model(trial_info['neural_input'])
#         _, _, _, beh_perf = behavior_summary(trial_info, pred_output, par=par)
#         perf_res[int(i_t*N_iter+i_iter)] = np.mean(beh_perf)
#     print(et, "complete")


# res_DF = pd.DataFrame({'ExtendedTime': np.repeat(extend_time, N_iter),
#                        'Performance' : perf_res})
# res_DF.to_pickle("/Volumes/Data_CSNL/project/RNN_study/20-06-19/HG/analysis/extended_perf_HL_Masse_mask.pkl")

res_DF = pd.read_pickle("/Volumes/Data_CSNL/project/RNN_study/20-06-19/HG/analysis/extended_perf_HL_Masse_mask.pkl")
sns.lineplot(x="ExtendedTime", y="Performance", data=res_DF)
plt.show()





# Trash bin ########################################################################################
# Conventional correciton to map estim into [0,pi]
# def wrap_correct():
    
# Functions: Debug
# p_test = np.array([0.4, 0.6])
# x_test = np.array([30,  160]) * np.pi/180
# circ_mean(p_test, x_test) * 180/np.pi

# sns.scatterplot(x='GT', y='Error', data=pd.DataFrame({'GT':ground_truth, 'Error':raw_error}))
# plt.xlabel(r"$\theta$"); plt.ylabel(r"$\hat{\theta} - \theta$"); plt.legend()
# plt.show()

# wrong way to recover mean
# pseudo_mean = post_prob @ post_support
# estim_expectation  = pseudo_expectation[par['design_rg']['estim'],:].mean(axis=0)


# Circular variance
# from scipy.stats import circvar
# circvar([0, 2*np.pi/3, 5*np.pi/3])

####################################################################################################
