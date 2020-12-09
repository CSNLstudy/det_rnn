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
train_data = stim_train.generate_trial()
random_trial_num = np.random.randint(stim_train.batch_size)
target_ori = np.arange(0,180,180/par_train['n_ori'])[train_data['stimulus_ori'][random_trial_num]]
# print('orientation = ' + str(target_ori))
# plot_trial(train_data,stim_train,TEST_TRIAL=random_trial_num)

grnn_hp = grnn_hp(par_train)
grnn_hp['neuron_stsp'] = True
grnn = gRNN(grnn_hp, par_train)
niter = 5000
grnn.train(stim_train = stim_train, stim_test= stim_train, niter=1)

print("Done.")