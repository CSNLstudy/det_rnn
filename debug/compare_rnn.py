import re, os, sys, pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Inspect

/Volumes/Data_CSNL/project/RNN_study/20-06-19/HG/current_best

##
sys.path.append('/Users/hyunwoogu/Dropbox/CSNL/Projects/RNN/neuroRNN/rnn/det_rnn/')
from det_rnn import *

## 
output_path = "/Volumes/Data_CSNL/project/RNN_study/20-06-19/HG/output/"
HL   = os.listdir(output_path)
print(HL)


## Load models
models = {}
for i,m in enumerate(HL):
    with open(output_path + m, 'rb') as f:
        models[re.sub('.pkl', '', m)] = pickle.load(f)

for m,v in models.items():
    plt.plot(v.model_performance['loss'], label=m)
plt.xlim((0,500)); plt.xlabel('Iteration'); plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()






## Prediction  ##########################################################################
par = update_parameters(par)
stimulus    = Stimulus(par)
trial_info  = stimulus.generate_trial()
pred_output, _ = models['HL_Masse_mask'].rnn_model(trial_info['neural_input'])

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
plt.ylim([-0.1,1.0]); plt.show()




## Behavior inspection for SavedModel ################################################################
load_model = tf.saved_model.load("/Volumes/Data_CSNL/project/RNN_study/20-06-19/HG/boost_wm/HL_booster9/")
par = update_parameters(par)
stimulus    = Stimulus(par)
trial_info  = stimulus.generate_trial()

for k, v in trial_info.items():
    trial_info[k] = tf.constant(v, name=k)

for k, v in hp.items():
    hp[k] = tf.constant(v, name=k)


pred_output, H, _, _ = load_model.rnn_model(trial_info['neural_input'], hp)

for k, v in trial_info.items():
    trial_info[k] = v.numpy()

ground_truth, estim_mean, raw_error, beh_perf = behavior_summary(trial_info, pred_output.numpy(), par=par)
behavior_figure(ground_truth, estim_mean, raw_error, beh_perf)


plt.plot(load_model.model_performance['perf'].numpy())
plt.show()

## resume training


# ## 
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

#         #
#         for k, v in trial_info.items():
#             trial_info[k] = tf.constant(v, name=k)

#         # 
#         pred_output, H, _, _ = load_model.rnn_model(trial_info['neural_input'], hp)

#         # 
#         for k, v in trial_info.items():
#             trial_info[k] = v.numpy()

#         #
#         _, _, _, beh_perf = behavior_summary(trial_info, pred_output.numpy(), par=par)
#         perf_res[int(i_t*N_iter+i_iter)] = np.mean(beh_perf)
#     print(et, "complete")


# res_DF = pd.DataFrame({'ExtendedTime': np.repeat(extend_time, N_iter),
#                         'Performance' : perf_res})
# res_DF.to_pickle("/Volumes/Data_CSNL/project/RNN_study/20-06-19/HG/analysis/extended_perf_HL_Masse_mask_savedmodel.pkl")

res_DF = pd.read_pickle("/Volumes/Data_CSNL/project/RNN_study/20-06-19/HG/analysis/extended_perf_HL_Masse_mask_savedmodel.pkl")
sns.lineplot(x="ExtendedTime", y="Performance", data=res_DF)
plt.ylim([-0.1,1.0]); plt.show()



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




