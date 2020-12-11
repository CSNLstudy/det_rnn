import copy, os, sys
import time

import pandas as pd
import numpy as np
import tensorflow as tf

import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append('../')
from det_rnn import *
import det_rnn.train as utils_train
import det_rnn.analysis as utils_analysis
#from det_rnn.train.model import Model

from models.base.analysis import behavior_summary, estimation_decision

from models.base.train import train
from models.gatedRNN.gatedRNN import gRNN
from models.gatedRNN.gatedRNN_hyper import grnn_hp

from utils.plotfnc import *

# training set
par_train = copy.deepcopy(par)
par_train['n_ori'] = 24
par_train = update_parameters(par_train)
stim_train = Stimulus(par_train)

# check stimulus
# for keepgoing in range(2000):
#     train_data = stim_train.generate_trial()
#     random_trial_num = np.random.randint(stim_train.batch_size)
#     target_ori = np.arange(0,180,180/par_train['n_ori'])[train_data['stimulus_ori'][random_trial_num]]
#     if tf.reduce_max(train_data['neural_input']) < 1:
#         print('wwtf wait a minute')
#     #print('orientation = ' + str(target_ori))
#     #plot_trial(train_data,stim_train,TEST_TRIAL=random_trial_num)


grnn_hp = grnn_hp(par_train)
grnn_hp['neuron_stsp'] = False
grnn_hp['learning_rate'] = 0.05
grnn = gRNN(grnn_hp, par_train)

# check plotting codes
stim_test = stim_train
test_data = stim_train.generate_trial()
test_lossStruct, test_outputs   = grnn.evaluate(test_data)
# plot rnn trial and outputs
TEST_TRIAL = np.random.randint(stim_train.batch_size)
plot_trial(test_data, stim_train, TEST_TRIAL=TEST_TRIAL, savename=None)
plot_rnn_output(test_data,test_outputs,stim_train, TEST_TRIAL=TEST_TRIAL, savename=None)
# plot rnn decision effects on estimation
df_trials, df_sum = estimation_decision(test_data, test_outputs, stim_test)
plot_decision_effects(df_trials,df_sum)

niter = 5000
grnn.train(stim_train = stim_train, stim_test= stim_train, niter=1)

print("Done.")