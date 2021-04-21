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

# hyperparameters
hp_dict = {'loss_ce_est': [1, 1e3],
           'loss_ce_dec': [1e5, 1e3],
           'scheduler': ['scheduler_estimFirst', None],
           'n_hidden': [300,500,100],
           'tau_max': [30000, 1000, 200],
           'tau_min': [10, 100, 1000],
           'gate_rnn': [True, False],
           'learning_rate': [5e-5, 5e-3]}

# # for debugging
# hp_dict = {'n_hidden': [50],
#            'learning_rate': [5e-3]}

hp_names = []
hp_lists = []
for (key, value) in hp_dict.items():
    hp_names += [key]
    hp_lists += [value]

hypsearch = itertools.product(*hp_lists)
print('Number of hyperparam combinations: ' + str(sum(1 for l in hypsearch)))

print('Training model for each combination of parameters....')
hypsearch = itertools.product(*hp_lists)

collist = copy.copy(hp_names)
collist.append('est_perf')
collist.append('dec_perf')
collist.append('model_number')
dflist = []

Nmaxtrain = 150
modelN = 0
lastrunend = 0 # if restarting

hp = grnn_hp(par)
dfpath = os.path.join(os.sep, *hp['output_base'].split(os.sep)[:-1], 'hp_df.pkl')
if os.path.isfile(dfpath):
    df1 = pd.read_pickle(dfpath)  # load any previous unfinished runs.
    dflist += [df1]

for params in hypsearch:
    if modelN < lastrunend:
        # skip how many ever models were already done?
        continue

    # training set
    par_train = copy.deepcopy(par)
    par_train['batch_size'] = 50  # smaller batch size
    par_train['n_ori'] = 24
    par_train['n_hidden'] = params[hp_names.index('n_hidden')]
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

    hp = grnn_hp(par_train)
    for idx in range(len(params)):
        if not idx == hp_names.index('n_hidden'):
            hp[hp_names[idx]] = params[idx]

    hp['model_number'] = modelN
    grnn = gRNN(hp)
    grnn.train(stim_train=stim_train, stim_test=stim_test, niter=Nmaxtrain)

    # evaluate on test set again
    test_lossStruct, test_outputs = grnn.evaluate(test_data)
    est_summary, dec_summary = behavior_summary(test_data, test_outputs, stim_test)
    dfdata = params + (np.mean(np.abs(est_summary['est_perf'])),
                       np.mean(np.abs(dec_summary['dec_perf'])),
                       grnn.hp['model_number'])
    dflist += [pd.DataFrame([dfdata], columns=collist)]
    modelN += 1

    # save df every iteration
    df = pd.concat(dflist, ignore_index=True)
    df.to_pickle(dfpath)

df1 = pd.read_pickle(dfpath)  # load back and check

