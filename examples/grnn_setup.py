import copy, os, sys
import time

# hyperparameters search
import pandas as pd
import itertools

# data manipulation
import numpy as np
import tensorflow as tf

# plotting
import seaborn as sns
import matplotlib.pyplot as plt
from utils.plotfnc import *

sys.path.append('../')
from det_rnn import *
import det_rnn.train as utils_train
import det_rnn.analysis as utils_analysis
#from det_rnn.train.model import Model

from models.base.analysis import behavior_summary, estimation_decision
from models.base.train import train
from models.gatedRNN.gatedRNN import gRNN
from models.gatedRNN.gatedRNN_hyper import grnn_hp

#niter = 1
#grnn.train(stim_train = stim_train, stim_test= stim_train, niter=niter)

# check stimulus
# for keepgoing in range(2000):
#     train_data = stim_train.generate_trial()
#     random_trial_num = np.random.randint(stim_train.batch_size)
#     target_ori = np.arange(0,180,180/par_train['n_ori'])[train_data['stimulus_ori'][random_trial_num]]
#     if tf.reduce_max(train_data['neural_input']) < 1:
#         print('wwtf wait a minute')
#     #print('orientation = ' + str(target_ori))
#     #plot_trial(train_data,stim_train,TEST_TRIAL=random_trial_num)

# # check plotting codes
# stim_test = stim_train
# test_data = stim_train.generate_trial()
# test_lossStruct, test_outputs   = grnn.evaluate(test_data)
# # plot rnn trial and outputs
# TEST_TRIAL = np.random.randint(stim_train.batch_size)
# #plot_trial(test_data, stim_train, TEST_TRIAL=TEST_TRIAL, savename=None)
# #plot_rnn_output(test_data,test_outputs,stim_train, TEST_TRIAL=TEST_TRIAL, savename=None)
# # estimation summary
# est_summary, dec_summary = behavior_summary(test_data, test_outputs, stim_train)
# #behavior_figure(est_summary)
# # plot rnn decision effects on estimation
# df_trials, df_sum = estimation_decision(test_data, test_outputs, stim_test)
# #plot_decision_effects(df_trials,df_sum)

# check loss function
# est_desired_out = test_data['desired_estim'][:, :, grnn.hp['n_rule_output_em']:]
# dec_desired_out = test_data['desired_decision'][:, :, grnn.hp['n_rule_output_dm']:]
# lossStruct = grnn.calc_loss(test_data,
#                             {'out_rnn': test_outputs['rnn_output'],
#                              'out_dm': dec_desired_out,
#                              'out_em': est_desired_out})
# # cross entropy loss is inherently nonzero (entropy of the distribution)..
# perfect_outputs = {'rnn_output': test_outputs['rnn_output'], 'rnn_states': test_outputs['rnn_states'],
#                    'dec_output': dec_desired_out, 'est_output': est_desired_out}
# est_summary, dec_summary = behavior_summary(test_data, perfect_outputs, stim_train)
# behavior_figure(est_summary)
# print("Perfect estimation cos error: {0:.2f}".format(np.mean(np.abs(est_summary['est_perf']))))
# print("Perfect decision performance: {0:.2f}".format(np.mean(dec_summary['dec_perf'])))

# basic without hp check
par_train = copy.deepcopy(par)
par_train['batch_size'] = 50  # smaller batch size
par_train['n_ori'] = 24
par_train = update_parameters(par_train)
stim_train = Stimulus(par_train)
train_data = stim_train.generate_trial()

# testing set
par_test = copy.deepcopy(par_train)
par_test['n_ori'] = 24
par_test['batch_size'] = 1000  # at least 9 refs * 24 orientations
par_test['reference'] = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
par_test = update_parameters(par_test)
stim_test = Stimulus(par_test)
test_data = stim_test.generate_trial()

grnn_hp = grnn_hp(par_train)
grnn = gRNN(grnn_hp)

niter = 1
grnn.train(stim_train = stim_train, stim_test= stim_train, niter=niter)
test_lossStruct, test_outputs   = grnn.evaluate(test_data)

# plot rnn trial and outputs
plot_trial(test_data, stim_train, TEST_TRIAL=0, savename=None)
plot_rnn_output(test_data,test_outputs,stim_train, TEST_TRIAL=0, savename=None)

# estimation summary
est_summary, dec_summary = behavior_summary(test_data, test_outputs, stim_train)
behavior_figure(est_summary)

# plot rnn decision effects on estimation
df_trials, df_sum = estimation_decision(test_data, test_outputs, stim_test)
plot_decision_effects(df_trials,df_sum)

print("Done.")
