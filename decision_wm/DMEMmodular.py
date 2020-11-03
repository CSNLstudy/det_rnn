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

# Main Idea
## Let us make an RNN with two separate modules: (sensory(SEN) + executive(EXE))

# from det_rnn.base.functions import random_normal_abs, alternating, modular_mask, w_rnn_mask
# EI_mask = modular_mask(par['connect_prob'], par['n_hidden']*2, par['exc_inh_prop'])
# EI_mask = w_rnn_mask(par['n_hidden'], par['exc_inh_prop'])
# plt.imshow(EI_mask)


par = update_parameters(par)
stimulus = Stimulus(par)
trial_info = dt.tensorize_trial(stimulus.generate_trial())

##
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
TEST_TRIAL = np.random.randint(par['batch_size'])
print(trial_info['reference_ori'][TEST_TRIAL])
axes[0].imshow(trial_info['neural_input'][:, TEST_TRIAL, :].numpy().T, aspect='auto', interpolation='none');
axes[0].set_title("Neural Input")
axes[1].imshow(trial_info['desired_decision'][:, TEST_TRIAL, :].numpy().T, aspect='auto', interpolation='none');
axes[1].set_title("Desired Decision")
axes[2].imshow(trial_info['desired_estim'][:, TEST_TRIAL, :].numpy().T, aspect='auto', interpolation='none');
axes[2].set_title("Desired Estimation")
fig.tight_layout(pad=2.0)
plt.show()

##
model_performance = {'perf_dm': [], 'perf_em': [], 'loss': [], 'perf_loss_dm': [], 'perf_loss_em': [], 'spike_loss': []}
ti_spec = dt.gen_ti_spec(stimulus.generate_trial())
model = dt.initialize_rnn(ti_spec)
for iter in range(5000):
    trial_info = dt.tensorize_trial(stimulus.generate_trial())
    Y, Loss = model(trial_info, dt.hp)
    model_performance = dt.append_model_performance(model_performance, trial_info, Y, Loss, par)
    dt.print_results(model_performance, iter)
    if iter < 3000:
        if iter % 30 == 0:
            dt.print_results(model_performance, iter)
            if iter % 60 == 0:
                dt.hp['lam_estim'] = 0;
                dt.hp['lam_decision'] = 2400.
            else:
                dt.hp['lam_estim'] = 300.;
                dt.hp['lam_decision'] = 0
    else:
        dt.hp['lam_estim'] = 300.;
        dt.hp['lam_decision'] = 2400.

model.model_performance = dt.tensorize_model_performance(model_performance)
tf.saved_model.save(model, "/Volumes/Data_CSNL/project/RNN_study/20-10-22/HG/output/decision/success2")

model2 = tf.saved_model.load("/Volumes/Data_CSNL/project/RNN_study/20-10-22/HG/output/decision/success1")

plt.plot(model2.model_performance['perf_dm'].numpy())
plt.plot(model2.model_performance['perf_em'].numpy())
plt.show()




