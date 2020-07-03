import pickle
from det_rnn import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from det_rnn.train import hyper as hp
import det_rnn.analysis as da
import det_rnn.train as dt
import pandas as pd
from celluloid import Camera
import seaborn as sns
# from os import makedirs,path
# from det_rnn.base.functions import convert_to_rg

save_dir = "/Users/eva/Dropbox/det_rnn_addingDM_200625/results"
# makedirs(save_dir + '/basicBehav')
# makedirs(save_dir + '/mov')

filename = '/basic_DMplugged_longIt.pkl'
with open(save_dir + filename, 'rb') as f:
    model = pickle.load(f)
#     dat = pickle.load(f)


N_iter      = 10
extend_time = np.arange(0,17.0,step=0.5)
perf_res    = np.empty((N_iter*len(extend_time),)) * np.nan
true_res = np.empty((N_iter*len(extend_time),)) * np.nan
est_res  = np.empty((N_iter*len(extend_time),)) * np.nan
raw_res = np.empty((N_iter*len(extend_time),)) * np.nan
# est_traj = np.empty((par['batch_size'],len(extend_time), N_iter) ) * np.nan
est_traj = np.empty((par['batch_size'],N_iter*len(extend_time),)) * np.nan
stim_traj =  np.empty((par['batch_size'],N_iter*len(extend_time),)) * np.nan

for i_t, et in enumerate(extend_time[0:10]):
    # i_t = 2; et = 2.
        for i_iter in range(N_iter):
            # i_iter = 1
            par['design'].update({'iti'     : (0,  1.5),
                                  'stim'    : (1.5,3.0),
                                  'delay'   : (3.0,4.5+et),
                                  'dm'      : (4.5+et, 5.0+et), # Eva
                                  # 'delay': (3.0, 4.5),
                                  # 'dm': (4.5 , 5.0 ),  # Eva
                                  # 'delay2': (5.0, 5.0+et),
                                  'estim'   : (5.0+et,6.5+et)})
            par = update_parameters(par)
            stimulus    = Stimulus(par)
            k=23
            while k < 24:
                trial_info = stimulus.generate_trial()
                k=np.unique( trial_info['stimulus_ori']).shape[0]

            y_pred_output, h_pred_output, syn_x_stack, syn_u_stack = model.rnn_model(trial_info['neural_input'], dt.hp)
            cenoutput = tf.nn.softmax(y_pred_output, axis=2)

            round_truth, estim_mean, raw_error, beh_perf = da.behavior_summary(trial_info, y_pred_output,
                                                                               par=par)
            perf_res[int(i_t * N_iter + i_iter)] = np.mean(beh_perf)
            true_res[int(i_t * N_iter + i_iter)] = np.mean(round_truth)
            est_res[int(i_t * N_iter + i_iter)] = np.mean(estim_mean)
            raw_res[int(i_t * N_iter + i_iter)] = np.mean(raw_error)
            est = estim_mean[np.argsort(round_truth)] % (np.pi)
            est_traj[:,int(i_t * N_iter + i_iter)] = est
            stim_traj[:, int(i_t * N_iter + i_iter)] = trial_info['stimulus_ori'][np.argsort(round_truth)]

            # if (i_t == 0 and i_iter == 0):
            plt.clf()
            iT = np.random.randint(12)
            plt.subplot(222)
            a = cenoutput.numpy()[:, iT, :]
            plt.imshow(a.T, aspect='auto', vmin=0, vmax=0.15)
            plt.colorbar()

            out_target = trial_info['desired_output']
            starget = np.sum(out_target, axis=2)
            starget = np.expand_dims(starget, axis=2)
            ntarget = out_target / np.repeat(starget, par['n_output'], axis=2)

            plt.subplot(224)
            a = ntarget[:, iT, :]
            plt.imshow(a.T, aspect='auto', vmin=0, vmax=0.15)
            plt.colorbar()
            plt.show()
            # #
            plt.savefig(save_dir + '/basicBehav_DlyimDec/' + 'i_t' + str(i_t) + '_i_iter' + str( i_iter) + '.png')
            #
            #
            if (i_t == 0 and i_iter == 0):
                da.behavior_figure(round_truth, estim_mean, raw_error, beh_perf)
            # plt.savefig(save_dir + '/dasummary/' + 'i_t' + str(i_t) + '_i_iter' + str(i_iter) + '.png')
            #
            nTrial = trial_info['stimulus_ori'].shape[0]
            nTpoint = y_pred_output.shape[0]
            nStim = np.unique( trial_info['stimulus_ori']).shape[0]
            idx_stim = np.empty((24,2), dtype = int)
            corrmat = np.empty((nTrial, nTrial))
            rsamat = np.empty((nStim, nStim))

            fig = plt.figure()
            camera = Camera(fig)
            for tp in range(nTpoint):
                fin_idx = 0
                trwsmat = np.empty((24,nTrial))
                for st in np.unique(trial_info['stimulus_ori']):
                    sz = np.where(trial_info['stimulus_ori'] == st)
                    y = cenoutput.numpy()
                    strt_idx = fin_idx
                    trwsmat[:,strt_idx:strt_idx+sz[0].shape[0]] = y[tp,sz[0],2:].T
                    fin_idx = strt_idx + sz[0].shape[0]
                    idx_stim[st, :] = strt_idx, fin_idx

                # np.fill_diagonal(trwsmat, np.nan)
                df = pd.DataFrame(data=trwsmat)
                corrmat = df.corr()
                for i in corrmat.index:
                    corrmat.loc[i, i] = np.nan
                for st in np.unique(trial_info['stimulus_ori']):
                    for st2 in np.unique(trial_info['stimulus_ori']):
                        parmat = corrmat.iloc[idx_stim[st,0]:idx_stim[st,1], idx_stim[st2,0]:idx_stim[st2,1]]
                        rsamat[st, st2]     =  parmat.mean().mean()
                plt.imshow(rsamat)
                plt.text(25, 25, "time step: " + str(tp*10) +  "ms", fontsize= 8)
                plt.subplots_adjust()
                camera.snap()
            animation = camera.animate()
            animation.save(save_dir + '/mov_DlyimDec/' + 'basic_DMplugged_longIt1_corrmat' + 'i_t' + str(i_t) + '_i_iter' + str(i_iter)  + '.mp4')
        print(et, "complete")

plt.clf()
iT = np.random.randint(12)
plt.subplot(222)
a = cenoutput.numpy()[:, iT, :]
plt.imshow(a.T, aspect='auto', vmin=0, vmax=0.15)
plt.colorbar()

out_target = trial_info['desired_output']
starget = np.sum(out_target, axis=2)
starget = np.expand_dims(starget, axis=2)
ntarget = out_target / np.repeat(starget, par['n_output'], axis=2)

plt.subplot(224)
a = ntarget[:, iT, :]
plt.imshow(a.T, aspect='auto', vmin=0, vmax=0.15)
plt.colorbar()
plt.show()
da.behavior_figure(round_truth, estim_mean, raw_error, beh_perf)

tp_until = 10
all_est = []
all_stnm = []
all_tp = []
for i_s in range(24):
    for i_t, et in enumerate(extend_time[0:tp_until]):
        for i_iter in range(N_iter):
            add_arry = est_traj[:, int(i_t * N_iter + i_iter)][np.where(stim_traj[:, int(i_t * N_iter + i_iter)] == i_s)]
            all_est = np.append(all_est, add_arry)
            all_stnm = np.append(all_stnm, np.repeat(i_s, add_arry.shape))
            all_tp = np.append(all_tp,  np.repeat(et, add_arry.shape))
plt.figure()
sns.lineplot(x='ext_time', y='est', hue='ori',data=pd.DataFrame({'ext_time': all_tp, 'est': all_est, 'ori':all_stnm}) )
plt.plot(extend_time[0:tp_until], np.mean(est_traj[:,0:tp_until], axis=0))
plt.show()
    ######
plt.figure()
for i_s in range(24):
    plt.plot((0,3), (0,3))
    plt.plot( i_s*np.pi/24, np.mean(all_est[np.where(all_stnm ==i_s)]) ,'o')
plt.axes().set_aspect('equal')
plt.show()

stimulus  = Stimulus()
trial_info  = stimulus.generate_trial()
y_pred_output,h_pred_output, syn_x_stack, syn_u_stack= model.rnn_model(trial_info['neural_input'],dt.hp)
out_target = trial_info['desired_output']

# sout = np.sum(output, axis=2)
#     sout = np.expand_dims(sout, axis=2)
#     noutput = output / np.repeat(sout,par['n_output'],axis=2)
#     cenoutput = tf.nn.softmax(output, axis=2)
#     cenoutput = cenoutput.numpy()



sout = np.sum(y_pred_output, axis=2)
sout = np.expand_dims(sout, axis=2)

# noutput = y_pred_output / np.repeat(sout, par['n_output'], axis=2)
cenoutput = tf.nn.softmax(y_pred_output, axis=2)
cenoutput = cenoutput.numpy()

starget = np.sum(out_target, axis=2)
starget = np.expand_dims(starget, axis=2)
ntarget = out_target / np.repeat(starget, par['n_output'], axis=2)


plt.clf()
iT = np.random.randint(12)
plt.subplot(222)
a = cenoutput[:, iT, :]
plt.imshow(a.T, aspect='auto', vmin=0, vmax=0.15)
plt.colorbar()

plt.subplot(223)
a = out_target[:, iT, :]
plt.imshow(a.T, aspect='auto')
plt.colorbar()
#
plt.subplot(224)
a = ntarget[:, iT, :]
plt.imshow(a.T, aspect='auto', vmin=0, vmax=0.15)
plt.colorbar()
plt.show()

#
ground_truth, estim_mean, raw_error, beh_perf = da.behavior_summary(trial_info, y_pred_output, par=par)
da.behavior_figure(ground_truth, estim_mean, raw_error, beh_perf)


cenoutput = tf.nn.softmax(y_pred_output, axis=2)

nTrial = trial_info['stimulus_ori'].shape[0]
nTpoint = y_pred_output.shape[0]
nStim = np.unique( trial_info['stimulus_ori']).shape[0]
idx_stim = np.empty((24,2), dtype = int)
corrmat = np.empty((nTrial, nTrial))
rsamat = np.empty((nStim, nStim))

fig = plt.figure()
camera = Camera(fig)
for tp in range(nTpoint):
    fin_idx = 0
    trwsmat = np.empty((24,nTrial))
    for st in np.unique(trial_info['stimulus_ori']):
        sz = np.where(trial_info['stimulus_ori'] == st)
        y = cenoutput.numpy()
        strt_idx = fin_idx
        trwsmat[:,strt_idx:strt_idx+sz[0].shape[0]] = y[tp,sz[0],2:].T
        fin_idx = strt_idx + sz[0].shape[0]
        idx_stim[st, :] = strt_idx, fin_idx

    # np.fill_diagonal(trwsmat, np.nan)
    df = pd.DataFrame(data=trwsmat)
    corrmat = df.corr()
    for i in corrmat.index:
        corrmat.loc[i, i] = np.nan
    for st in np.unique(trial_info['stimulus_ori']):
        for st2 in np.unique(trial_info['stimulus_ori']):
            parmat = corrmat.iloc[idx_stim[st,0]:idx_stim[st,1], idx_stim[st2,0]:idx_stim[st2,1]]
            rsamat[st, st2]     =  parmat.mean().mean()
    plt.imshow(rsamat)
    plt.text(25, 25, "time step: " + str(tp*10) +  "ms", fontsize= 8)
    plt.subplots_adjust()
    camera.snap()
animation = camera.animate()
animation.save('basic_DMplugged_longIt1_corrmat.mp4')




#
# plt.ion()
# for tp in range(nTpoint):
#     fin_idx = 0
#     trwsmat = np.empty((24,nTrial))
#     for st in np.unique(trial_info['stimulus_ori']):
#         sz = np.where(trial_info['stimulus_ori'] == st)
#         y = cenoutput.numpy()
#         strt_idx = fin_idx
#         trwsmat[:,strt_idx:strt_idx+sz[0].shape[0]] = y[tp,sz[0],2:].T
#         fin_idx = strt_idx + sz[0].shape[0]
#         idx_stim[st, :] = strt_idx, fin_idx
#
#     # np.fill_diagonal(trwsmat, np.nan)
#     df = pd.DataFrame(data=trwsmat)
#     corrmat = df.corr()
#     for i in corrmat.index:
#         corrmat.loc[i, i] = np.nan
#     for st in np.unique(trial_info['stimulus_ori']):
#         for st2 in np.unique(trial_info['stimulus_ori']):
#             parmat = corrmat.iloc[idx_stim[st,0]:idx_stim[st,1], idx_stim[st2,0]:idx_stim[st2,1]]
#             rsamat[st, st2]     =  parmat.mean().mean()
#     plt.imshow(rsamat)
#     plt.title("time step: " + str(tp*10) +  "ms")
#     # plt.show(False)
#     plt.pause(0.001)
#     plt.clf()




N_iter      = 10
extend_time = np.arange(0,17.0,step=0.5)
perf_res    = np.empty((N_iter*len(extend_time),)) * np.nan
true_res = np.empty((N_iter*len(extend_time),)) * np.nan
est_res  = np.empty((N_iter*len(extend_time),)) * np.nan
raw_res = np.empty((N_iter*len(extend_time),)) * np.nan
for i_t, et in enumerate(extend_time):
    for i_iter in range(N_iter):
        par['design'].update({'iti'     : (0,  1.5),
                              'stim'    : (1.5,3.0),
                              'delay'   : (3.0,4.5+et),
                              'estim'   : (4.5+et,6.0+et)})
        par = update_parameters(par)
        stimulus    = Stimulus(par)
        trial_info  = dt.tensorize_trial(stimulus.generate_trial())
        pred_output, H, _, _ = model.rnn_model(trial_info['neural_input'], dt.hp)

        round_truth, estim_mean, raw_error, beh_perf = da.behavior_summary(dt.numpy_trial(trial_info), pred_output, par=par)
        perf_res[int(i_t*N_iter+i_iter)] = np.mean(beh_perf)
        true_res[int(i_t * N_iter + i_iter)] = np.mean(round_truth)
        est_res[int(i_t * N_iter + i_iter)] = np.mean(estim_mean)
        raw_res[int(i_t * N_iter + i_iter)] = np.mean(raw_error)
    print(et, "complete")

fig, axes = plt.subplots(4,1, figsize=(10,8))
TEST_TRIAL = np.random.randint(stimulus.batch_size)
axes[0].imshow(trial_info['neural_input'][:,TEST_TRIAL,:].T, aspect='auto'); axes[0].set_title("Neural Input")
axes[1].imshow(trial_info['desired_output'][:,TEST_TRIAL,:].T, aspect='auto'); axes[1].set_title("Desired Output")
# axes[2].imshow(pred_output[1].numpy()[:,TEST_TRIAL,:].T, aspect='auto'); axes[2].set_title("Model Output")
axes[2].imshow(cenoutput[:,TEST_TRIAL,:].T, aspect='auto'); axes[2].set_title("Model Output y")
# axes[3].imshow(pred_output[0].numpy()[:,TEST_TRIAL,:].T, aspect='auto'); axes[3].set_title("Model Output [0]")
axes[3].imshow(h_pred_output[:,TEST_TRIAL,:].numpy().T, aspect='auto'); axes[3].set_title("Model Output [0]")
fig.tight_layout(pad=2.0)
plt.show()
