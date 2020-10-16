import sys, pickle, copy
import scipy.stats
import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append('../')
from det_rnn import *
import det_rnn.train as dt
import det_rnn.analysis as da

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

##
par = update_parameters(par)
stimulus = Stimulus(par)
trial_info = dt.tensorize_trial(stimulus.generate_trial())


##
for i in range(20):
    model = tf.saved_model.load("/Volumes/Data_CSNL/project/RNN_study/20-10-15/HG/output/decision/success"+str(i))
    stim_list   = np.arange(24)
    ref_list    = np.arange(-4,4+1)
    par['reference'] = ref_list;  par = update_parameters(par)
    dm_output_total  = np.zeros((par['n_timesteps'], 128 * 24 * 9, 3))
    em_output_total  = np.zeros((par['n_timesteps'], 128 * 24 * 9, 25))
    stimulus_ori_total  = np.zeros((128*24*9,))
    reference_ori_total = np.zeros((128*24*9,))
    H_total = np.zeros((par['n_timesteps'], 10 * 24 * 9, par['n_hidden']))
    
    for i_s, st in enumerate(stim_list):
        stim_dist = np.zeros(24); stim_dist[i_s] = 1.; par['stim_dist'] = stim_dist
        for i_r, re in enumerate(ref_list):
            ref_dist  = np.zeros(9); ref_dist[i_r] = 1.;  par['ref_dist'] = ref_dist
            par = update_parameters(par); stimulus = Stimulus(par)
            trial_info = dt.tensorize_trial(stimulus.generate_trial())
            pred_output_dm, pred_output_em, H, _, _  = model.rnn_model(trial_info['neural_input'], dt.hp)
            dm_output_total[:, (128*9*i_s+128*i_r):(128*9*i_s+128*(i_r+1)),:] = pred_output_dm.numpy()
            em_output_total[:, (128*9*i_s+128*i_r):(128*9*i_s+128*(i_r+1)),:] = pred_output_em.numpy()
            stimulus_ori_total[(128*9*i_s+128*i_r):(128*9*i_s+128*(i_r+1))]   = trial_info['stimulus_ori'].numpy()
            reference_ori_total[(128*9*i_s+128*i_r):(128*9*i_s+128*(i_r+1))]  = trial_info['reference_ori'].numpy()
            H_total[:,(10*9*i_s+10*i_r):(10*9*i_s+10*(i_r+1)),:] = H.numpy()[:,:10,:]
        print(st)


    DV = dm_output_total[475,:,2]-dm_output_total[475,:,1]
    ground_truth, estim_mean, raw_error, beh_perf = da.behavior_summary({'stimulus_ori': stimulus_ori_total}, em_output_total, par=par)
    corrError = (raw_error - np.pi/2.) % (np.pi) - np.pi/2.

    behav = {'stimulus_ori_total' : stimulus_ori_total,
             'reference_ori_total': reference_ori_total,
             'ground_truth':ground_truth,
             'estim_mean'  : estim_mean,
             'raw_error'   : raw_error,
             'beh_perf'    : beh_perf,
             'corrError'   : corrError,
             'choice'      : DV > 0}
    behav = pd.DataFrame(behav)
    with open('/Volumes/Data_CSNL/project/RNN_study/20-10-15/HG/output/behavior/sequential'+str(i)+'.pkl', 'wb') as f:
        pickle.dump(behav, f)


    ## Pattern similarity
    H_ref  = np.tile(np.repeat(ref_list,10),24)
    Hsimil = np.ones((9,75,24,24)) * np.nan
    for i_r, ref in enumerate(ref_list):
        for i_t in range(75):
            corrH = np.corrcoef(H_total[i_t*10,H_ref == ref,:])
            np.fill_diagonal(corrH, np.nan)    
            corrHreduc = np.ones((24,24)) * np.nan
            for istim in range(24):
                for jstim in range(24):
                    corrHreduc[istim,jstim] = np.nanmean(corrH[(istim*10):((istim+1)*10),
                                                               (jstim*10):((jstim+1)*10)])
            Hsimil[i_r,i_t,:,:] = corrHreduc
        print(ref)
    with open('/Volumes/Data_CSNL/project/RNN_study/20-10-15/HG/output/pattern_similarity/sequential'+str(i)+'.pkl', 'wb') as f:
        pickle.dump(Hsimil, f)

    ## Decoding
    # stim_decode  = np.ones((9, 75, 10*24)) * np.nan  # LDA    
    # post_support = np.linspace(0,np.pi,par['n_ori'],endpoint=False) + np.pi/par['n_ori']/2
    # post_sinr = np.sin(2*post_support)
    # post_cosr = np.cos(2*post_support)
    
    # stim_gt = np.repeat(np.arange(24),10)
    # for i_r in range(9):
    #     curr_trials = np.concatenate([np.arange(10 * 9 * i_s + 10 * i_r, 10 * 9 * i_s + 10 * (i_r + 1)) for i_s in range(24)], axis=0)
    #     for i_t in range(75):
    #         readout = np.ones((240,24)) * np.nan
    #         for i_trial in range(240):
    #             clf = LinearDiscriminantAnalysis()
    #             clf.fit(H_total[i_t*10,curr_trials[i_trial != np.arange(240)],:],
    #                     stim_gt[i_trial != np.arange(240)])
    #             readout[i_trial,:] = clf.predict_proba(H_total[i_t*10,curr_trials[i_trial],:].reshape((1, -1)))
    #         stim_decode[i_r, i_t, :] = np.arctan2(readout @ post_sinr, readout @ post_cosr) / 2.
    #         print(i_r, i_t)
    
    
    # with open('/Volumes/Data_CSNL/project/RNN_study/20-10-15/HG/output/decoded/sequential'+str(i)+'.pkl', 'wb') as f:
    #     pickle.dump(stim_decode, f)

    ## Decoding
    stim_decode  = np.ones((9, 75, 10*24)) * np.nan  # LDA    
    post_support = np.linspace(0,np.pi,par['n_ori'],endpoint=False) + np.pi/par['n_ori']/2
    post_sinr = np.sin(2*post_support)
    post_cosr = np.cos(2*post_support)

    ### time of interest = 25    
    toi     = 25
    stim_gt = np.repeat(np.arange(24),10*9)
    clf = LinearDiscriminantAnalysis()
    clf.fit(H_total[toi*10,:,:],stim_gt)
    for i_t in range(75):
        readout = np.ones((240*9,24)) * np.nan
        for i_trial in range(240*9):
            readout[i_trial,:] = clf.predict_proba(H_total[i_t*10,i_trial,:].reshape((1, -1)))
        for i_r in range(9):
            curr_trials = np.concatenate([np.arange(10 * 9 * i_s + 10 * i_r, 10 * 9 * i_s + 10 * (i_r + 1)) for i_s in range(24)], axis=0)
            stim_decode[i_r, i_t, :] = np.arctan2(readout[curr_trials,:] @ post_sinr, readout[curr_trials,:] @ post_cosr) / 2.
        print(i_t)
    
    
    with open('/Volumes/Data_CSNL/project/RNN_study/20-10-15/HG/output/gen_decoded/sequential'+str(i)+'.pkl', 'wb') as f:
        pickle.dump(stim_decode, f)

