import copy, os, sys
import matplotlib.pyplot as plt
from utils.plotfnc import *
sys.path.append('../')
from det_rnn import *

import det_rnn.train as utils_train
import det_rnn.analysis as utils_analysis
#from det_rnn.train.model import Model

from models.base.analysis import behavior_summary, estimation_decision

par['design'] = {'iti': (0, 0.3),
                 'stim': (0.3, 0.6),
                 'delay': ((0.6, 0.8), (1.1, 1.2)),
                 'decision': (0.8, 1.1),
                 'estim': (1.2, 1.5)}

par_train = copy.deepcopy(par)
par_train['batch_size'] = 50  # smaller batch size
par_train['n_ori']      = 24
par_train['n_hidden']   = 100
par_train = update_parameters(par_train)
stim_train = Stimulus(par_train)
train_data = stim_train.generate_trial()

# check stimulus
for b in range(stim_train.batch_size):
    target_ori = np.arange(0,180,180/par_train['n_ori'])[train_data['stimulus_ori'][b]]
    ref_ori = np.arange(0,180,180/par_train['n_tuned_input'])[train_data['ref_neuron'][b]]
    print('orientation = ' + str(target_ori) + ', reference = ' + str(ref_ori))
    plot_trial(train_data, stim_train, TEST_TRIAL=b)
    print('\n')

