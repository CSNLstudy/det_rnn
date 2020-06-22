import sys
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.append('../')
from det_rnn import *
import det_rnn.analysis as da
import det_rnn.train as dt


############################################################################################################
# Behavior inspection for the current best model
############################################################################################################
## Inspect current best model
## you can inspect your own model by slightly modifying the below code
curr_best_dir = "/Volumes/Data_CSNL/project/RNN_study/20-06-19/HG/current_best/"

par   = update_parameters(par)
stimulus    = Stimulus(par)
trial_info  = dt.tensorize_trial(stimulus.generate_trial())

model = tf.saved_model.load(curr_best_dir)
pred_output, H, _, _ = model.rnn_model(trial_info['neural_input'], dt.hp)
pred_output = da.softmax_pred_output(pred_output)

fig, axes = plt.subplots(3,1, figsize=(10,8))
TEST_TRIAL = np.random.randint(par['batch_size'])
axes[0].imshow(trial_info['neural_input'][:,TEST_TRIAL,:].numpy().T, aspect='auto'); axes[0].set_title("Neural Input")
axes[1].imshow(trial_info['desired_output'][:,TEST_TRIAL,:].numpy().T, aspect='auto'); axes[1].set_title("Desired Output")
axes[2].imshow(pred_output[:,TEST_TRIAL,:].T,  aspect='auto', vmin=0, vmax=0.15)
fig.tight_layout(pad=2.0)
plt.show()

trial_info = dt.numpy_trial(trial_info)
ground_truth, estim_mean, raw_error, beh_perf = da.behavior_summary(trial_info, pred_output, par=par)
da.behavior_figure(ground_truth, estim_mean, raw_error, beh_perf)


############################################################################################################
# Temporal generalizability of the current best model
############################################################################################################
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
#         trial_info  = dt.tensorize_trial(stimulus.generate_trial())
#         pred_output, H, _, _ = model.rnn_model(trial_info['neural_input'], dt.hp)
#
#         _, _, _, beh_perf = da.behavior_summary(dt.numpy_trial(trial_info), pred_output, par=par)
#         perf_res[int(i_t*N_iter+i_iter)] = np.mean(beh_perf)
#     print(et, "complete")

# res_DF = pd.DataFrame({'ExtendedTime': np.repeat(extend_time, N_iter),
#                        'Performance' : perf_res})
# res_DF.to_pickle("some/path")

# load pre-run dataset
res_DF = pd.read_pickle("/Volumes/Data_CSNL/project/RNN_study/20-06-19/HG/analysis/extended_perf_HL_Masse_mask_adultrnn.pkl")
sns.lineplot(x="ExtendedTime", y="Performance", data=res_DF)
plt.ylim([-0.1,1.0]); plt.show()



