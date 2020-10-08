# Alternation training of decision and estimation
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
fig, axes = plt.subplots(3,1, figsize=(10,8))
TEST_TRIAL = np.random.randint(par['batch_size'])
print(trial_info['reference_ori'][TEST_TRIAL])
axes[0].imshow(trial_info['neural_input'][:,TEST_TRIAL,:].numpy().T, aspect='auto', interpolation='none'); axes[0].set_title("Neural Input")
axes[1].imshow(trial_info['desired_decision'][:,TEST_TRIAL,:].numpy().T, aspect='auto', interpolation='none'); axes[1].set_title("Desired Decision")
axes[2].imshow(trial_info['desired_estim'][:,TEST_TRIAL,:].numpy().T, aspect='auto', interpolation='none'); axes[2].set_title("Desired Estimation")
fig.tight_layout(pad=2.0)
plt.show()


##
model_performance = {'perf_dm': [], 'perf_em': [], 'loss': [], 'perf_loss_dm': [], 'perf_loss_em': [], 'spike_loss': []}
ti_spec = dt.gen_ti_spec(stimulus.generate_trial())
model   = dt.initialize_rnn(ti_spec)
for iter in range(5000):
    trial_info = dt.tensorize_trial(stimulus.generate_trial())
    Y, Loss = model(trial_info, dt.hp)
    model_performance = dt.append_model_performance(model_performance, trial_info, Y, Loss, par)
    dt.print_results(model_performance, iter)
    if iter < 3000:
        if iter % 30 == 0:
            dt.print_results(model_performance, iter)
            if iter % 60 == 0:
                dt.hp['lam_estim'] = 0; dt.hp['lam_decision'] = 2400.
            else:
                dt.hp['lam_estim'] = 300.; dt.hp['lam_decision'] = 0
    else:
        dt.hp['lam_estim'] = 300.; dt.hp['lam_decision'] = 2400.

#
# # for mk in model_performance.keys():
# #     model_performance[mk] = model_performance[mk].numpy().tolist()
#
# model.model_performance = dt.tensorize_model_performance(model_performance)
# tf.saved_model.save(model, "/Volumes/Data_CSNL/project/RNN_study/20-10-08/HG/output/decision/success10")
#
#
# ##
# model = tf.saved_model.load("/Volumes/Data_CSNL/project/RNN_study/20-10-08/HG/output/joint_tasks/sequential03")
#
# plt.plot(model.model_performance['loss'].numpy())
# plt.ylim([0,5e5])
# # plt.plot(model7.model_performance['loss'].numpy())
# plt.show()
#
#
#
# ##
# model = tf.saved_model.load("/Volumes/Data_CSNL/project/RNN_study/20-10-08/HG/output/decision/success7")
# pred_output_dm, pred_output_em, H, _, _ = model.rnn_model(trial_info['neural_input'], dt.hp)
# pred_output_dm = da.softmax_pred_output(pred_output_dm)
# pred_output_em = da.softmax_pred_output(pred_output_em)
#
# ##
# fig, axes = plt.subplots(3,1, figsize=(10,8))
# TEST_TRIAL = np.random.randint(par['batch_size'])
# axes[0].imshow(trial_info['neural_input'][:,TEST_TRIAL,:].numpy().T, aspect='auto'); axes[0].set_title("Neural Input")
# axes[1].imshow(trial_info['desired_decision'][:,TEST_TRIAL,:].numpy().T, aspect='auto', interpolation='none'); axes[1].set_title("Desired Decision")
# axes[2].imshow(pred_output_dm[:,TEST_TRIAL,:].T, aspect='auto', interpolation='none'); axes[2].set_title("Predicted Decision")
# # axes[1].imshow(trial_info['desired_estim'][:,TEST_TRIAL,:].numpy().T, aspect='auto'); axes[1].set_title("Desired Estimation")
# # axes[2].imshow(pred_output_em[:,TEST_TRIAL,:].T, aspect='auto', vmax=0.13, vmin=0); axes[2].set_title("Predicted Estimation")
# fig.tight_layout(pad=2.0)
# plt.show()
#
#
# ####################################
# # Behavior analysis
# ####################################
# ### Accumulate 128 pred_outputs for each (stimulus x reference) (128 x 24 x 9 in total)
# ### Accumulate 10 Hs for each (stimulus x reference) (10 x 24 x 9 in total)
# stim_list   = np.arange(24)
# ref_list    = np.arange(-4,4+1)
# par['reference'] = ref_list;  par = update_parameters(par)
# dm_output_total  = np.zeros((par['n_timesteps'], 128 * 24 * 9, 3))
# em_output_total  = np.zeros((par['n_timesteps'], 128 * 24 * 9, 25))
# stimulus_ori_total  = np.zeros((128*24*9,))
# reference_ori_total = np.zeros((128*24*9,))
# H_total = np.zeros((par['n_timesteps'], 10 * 24 * 9, par['n_hidden']))
#
# for i_s, st in enumerate(stim_list):
#     stim_dist = np.zeros(24); stim_dist[i_s] = 1.; par['stim_dist'] = stim_dist
#     for i_r, re in enumerate(ref_list):
#         ref_dist  = np.zeros(9); ref_dist[i_r] = 1.;  par['ref_dist'] = ref_dist
#         par = update_parameters(par); stimulus = Stimulus(par)
#         trial_info = dt.tensorize_trial(stimulus.generate_trial())
#         pred_output_dm, pred_output_em, H, _, _  = model.rnn_model(trial_info['neural_input'], dt.hp)
#         dm_output_total[:, (128*9*i_s+128*i_r):(128*9*i_s+128*(i_r+1)),:] = pred_output_dm.numpy()
#         em_output_total[:, (128*9*i_s+128*i_r):(128*9*i_s+128*(i_r+1)),:] = pred_output_em.numpy()
#         stimulus_ori_total[(128*9*i_s+128*i_r):(128*9*i_s+128*(i_r+1))]   = trial_info['stimulus_ori'].numpy()
#         reference_ori_total[(128*9*i_s+128*i_r):(128*9*i_s+128*(i_r+1))]  = trial_info['reference_ori'].numpy()
#         H_total[:,(10*9*i_s+10*i_r):(10*9*i_s+10*(i_r+1)),:] = H.numpy()[:,:10,:]
#     print(st)
#
#
#
# # Behavior analysis: reference impact on decision(DV)?
# DV = dm_output_total[475,:,2]-dm_output_total[475,:,1]
# plt.scatter(reference_ori_total,DV); plt.show()
#
# sns.boxplot(x='ref', y='DV', data=pd.DataFrame({'ref':reference_ori_total, 'DV':np.minimum(np.exp(DV),25)}))
# plt.show()
#
# ground_truth, estim_mean, raw_error, beh_perf = da.behavior_summary({'stimulus_ori': stimulus_ori_total}, em_output_total, par=par)
# da.behavior_figure(ground_truth, estim_mean, raw_error, beh_perf)
# corrError = (raw_error - np.pi/2.) % (np.pi) - np.pi/2.
#
# behav = {'stimulus_ori_total': stimulus_ori_total,
#          'reference_ori_total': reference_ori_total,
#          'ground_truth':ground_truth,
#          'estim_mean': estim_mean,
#          'raw_error': raw_error,
#          'beh_perf': beh_perf,
#          'corrError': corrError}
# behav = pd.DataFrame(behav)
# with open('/Volumes/Data_CSNL/project/RNN_study/20-10-08/HG/output/joint_tasks/behavior/sequential03.pkl', 'wb') as f:
#     pickle.dump(behav, f)
#
# ##
# pd.DataFrame({'Choice':DV>0,'ref':reference_ori_total, 'err':corrError})
# with plt.style.context('fivethirtyeight'):
#     fig = plt.figure(figsize=(8, 5))
#     g = sns.violinplot(x="Ref", y="MM", hue="D", data=nFF[~nFF['Lapse']], palette="Set1", linewidth=1., showfliers = False, bw=.005)
#     g.legend(loc='center left', bbox_to_anchor=(1., 0.5), ncol=1)
#     plt.show()
#
# ## DIV? No!
# # sns.violinplot(x='ref', y='err', bw=.05, data=pd.DataFrame({'ref':reference_ori_total[(stimulus_ori_total>=12) & (stimulus_ori_total<24)],
# #                                                             'err':corrError[(stimulus_ori_total>=12) & (stimulus_ori_total<24)]}))
# # plt.show()
#
# ######
#
# ######
#
# with open('/Volumes/Data_CSNL/project/RNN_study/20-10-08/HG/output/joint_tasks/stimulus_decoder/sequential01.pkl', 'rb') as f:
#     Decode1 = pickle.load(f)
# with open('/Volumes/Data_CSNL/project/RNN_study/20-10-08/HG/output/joint_tasks/stimulus_decoder/sequential02.pkl', 'rb') as f:
#     Decode2 = pickle.load(f)
# with open('/Volumes/Data_CSNL/project/RNN_study/20-10-08/HG/output/joint_tasks/stimulus_decoder/sequential03.pkl', 'rb') as f:
#     Decode3 = pickle.load(f)
# with open('/Volumes/Data_CSNL/project/RNN_study/20-10-08/HG/output/joint_tasks/stimulus_decoder/sequential04.pkl', 'rb') as f:
#     Decode4 = pickle.load(f)
#
#
#
# sns.violinplot(x='ref', y='err', bw=.05, data=pd.DataFrame({'ref':Behav2.reference_ori_total,'err':Behav2.corrError}))
# plt.show()
#
# sns.boxplot(x='ref', y='err', hue='Choice', data=pd.DataFrame({'Choice':DV>0,'ref':reference_ori_total,'err':corrError}))
# plt.show()
#
#
#
# test_IQR1 = np.ones((24,9))
# test_IQR2 = np.ones((24,9))
# test_IQR3 = np.ones((24,9))
# test_IQR4 = np.ones((24,9))
# for i_s, s_t in enumerate(stim_list):
#     for i_r, ref in enumerate(ref_list):
#         # test_IQR[i_s, i_r] = scipy.stats.iqr(corrError[(reference_ori_total==ref) & (stimulus_ori_total==s_t)])
#         test_IQR1[i_s, i_r] = scipy.stats.iqr(Behav1.corrError[(Behav1.reference_ori_total == ref) & (Behav1.stimulus_ori_total == s_t)])
#         test_IQR2[i_s, i_r] = scipy.stats.iqr(Behav2.corrError[(Behav2.reference_ori_total == ref) & (Behav2.stimulus_ori_total == s_t)])
#         test_IQR3[i_s, i_r] = scipy.stats.iqr(Behav3.corrError[(Behav3.reference_ori_total == ref) & (Behav3.stimulus_ori_total == s_t)])
#         test_IQR4[i_s, i_r] = scipy.stats.iqr(Behav4.corrError[(Behav4.reference_ori_total == ref) & (Behav4.stimulus_ori_total == s_t)])
#
# for i_r, ref in enumerate(ref_list):
#     print(np.std(corrError[reference_ori_total==ref]))
#
# plt.imshow(test_IQR)
# plt.show()
#
# plt.plot(np.mean(test_IQR1,axis=0))
# plt.plot(np.mean(test_IQR2,axis=0))
# plt.plot(np.mean(test_IQR3,axis=0))
# plt.plot(np.mean(test_IQR4,axis=0))
# plt.show()
#
# ##
# H_ref  = np.tile(np.repeat(ref_list,10),24)
# H_stim = np.repeat(np.arange(24),90)
#
# for ref in ref_list:
#     plt.plot(np.arange(750), np.mean(H_total[:, H_ref == ref, :], axis=(1,2)), label=str(ref))
# plt.legend()
# plt.show()
#
#
# ##
# fig, ax = plt.subplots(1,5, figsize=(18,5))
# ax[0].imshow(np.corrcoef(H_total[100,H_ref == 4.,:])); ax[0].set_title("ITI")
# ax[1].imshow(np.corrcoef(H_total[200,H_ref == 4.,:])); ax[1].set_title("Stimulus")
# ax[2].imshow(np.corrcoef(H_total[380,H_ref == 4.,:])); ax[2].set_title("Delay")
# ax[3].imshow(np.corrcoef(H_total[480,H_ref == 4.,:])); ax[3].set_title("Delay")
# ax[4].imshow(np.corrcoef(H_total[600,H_ref == 4.,:])); ax[4].set_title("Estimation")
# plt.tight_layout(pad=2.0); plt.show()
#
#
# ##
# fig, ax = plt.subplots(1,5, figsize=(18,5))
# ax[0].imshow(np.corrcoef(H_total[580,H_ref == -4.,:])); ax[0].set_title("-4")
# ax[1].imshow(np.corrcoef(H_total[580,H_ref == -2.,:])); ax[1].set_title("-2")
# ax[2].imshow(np.corrcoef(H_total[580,H_ref == 0.,:]));  ax[2].set_title("0")
# ax[3].imshow(np.corrcoef(H_total[580,H_ref == 2.,:]));  ax[3].set_title("2")
# ax[4].imshow(np.corrcoef(H_total[580,H_ref == 4.,:]));  ax[4].set_title("4")
# plt.tight_layout(pad=2.0); plt.show()
#
#
# #############################################
# # MVPA decoding
# #############################################
# import cmath
# import scipy.stats as stats
#
# def mean(angles, deg=True):
#     '''Circular mean of angle data(default to degree)
#     '''
#     a = np.deg2rad(angles) if deg else np.array(angles)
#     angles_complex = np.frompyfunc(cmath.exp, 1, 1)(a * 1j)
#     mean = cmath.phase(angles_complex.sum()) % (2 * np.pi)
#     return round(np.rad2deg(mean) if deg else mean, 7)
#
# def corrcoef(x, y, deg=True, test=False):
#     '''Circular correlation coefficient of two angle data(default to degree)
#     Set `test=True` to perform a significance test.
#     '''
#     convert = np.pi / 180.0 if deg else 1
#     sx = np.frompyfunc(np.sin, 1, 1)((x - mean(x, deg)) * convert)
#     sy = np.frompyfunc(np.sin, 1, 1)((y - mean(y, deg)) * convert)
#     r = (sx * sy).sum() / np.sqrt((sx ** 2).sum() * (sy ** 2).sum())
#
#     if test:
#         l20, l02, l22 = (sx ** 2).sum(),(sy ** 2).sum(), ((sx ** 2) * (sy ** 2)).sum()
#         test_stat = r * np.sqrt(l20 * l02 / l22)
#         p_value = 2 * (1 - stats.norm.cdf(abs(test_stat)))
#         return tuple(round(v, 7) for v in (r, test_stat, p_value))
#     return round(r, 7)
#
#
# ## Decoding
# stim_decode = np.ones((9, 75, 10*24)) * np.nan  # LDA
# dv_decode   = np.ones((5, 75, 3072)) * np.nan   # SVM
# # dm_decode   = np.ones((5, 150, 3072)) * np.nan   # SVM
#
# post_support = np.linspace(0,np.pi,par['n_ori'],endpoint=False) + np.pi/par['n_ori']/2
# post_sinr = np.sin(2*post_support)
# post_cosr = np.cos(2*post_support)
#
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
#
# # plt.imshow(stim_decode[4,:,:].T, aspect='auto')
# # plt.show()
#
# with open('/Volumes/Data_CSNL/project/RNN_study/20-10-08/HG/output/joint_tasks/stimulus_decoder/sequential03.pkl', 'wb') as f:
#     pickle.dump(stim_decode, f)
#
# stim_perf = np.ones((9, 75)) * np.nan
# for i_r in range(9):
#     for i_t in range(75):
#         stim_perf[i_r, i_t] = corrcoef(np.repeat(np.arange(24),10)/24.*np.pi,
#                                        Decode4[i_r, i_t, :] %np.pi, deg=False, test=False)
#     print(i_r, i_t)
#
# ##
# for i_r in range(9):
#     plt.plot(stim_perf[i_r,:], label=str(ref_list[i_r]))
# plt.legend()
# plt.show()
#
#
# # Behavior analysis: reference impact on estimation distribution?
# ground_truth, estim_mean, raw_error, beh_perf = da.behavior_summary({'stimulus_ori': stimulus_ori_total}, pred_output_total, par=par)
#
#

#
# data_dic = {'dm_output_total': dm_output_total,
#             'em_output_total': em_output_total,
#             'stimulus_ori_total': stimulus_ori_total,
#             'reference_ori_total': reference_ori_total,
#             'H_total': H_total,
#             'ground_truth':ground_truth,
#             'estim_mean': estim_mean,
#             'raw_error': raw_error,
#             'beh_perf': beh_perf}
#
# with open('/Volumes/Data_CSNL/project/RNN_study/20-10-08/HG/data/decision_success7/data_dic.pkl', 'wb') as f:
#     pickle.dump(data_dic, f)
#


########################################################################
########################################################################



