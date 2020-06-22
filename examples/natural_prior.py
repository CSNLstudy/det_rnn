import sys, os, time

import copy
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')
from det_rnn import *
import det_rnn.train as train_utils
import tensorflow as tf
from det_rnn.train.model import Model

model_dir = "/Users/JRyu/github/det_rnn/experiments/naturalprior/200622/"
os.makedirs(model_dir, exist_ok=True)


###### Generate stimulus ######

# training set
par_train = copy.deepcopy(par)
# use "natural prior"
par_train['stim_dist'] = 'natural'
par_train['natural_a'] = 2e-4
par_train['n_ori'] = 1000 # make it somewhat continuous
par_train = update_parameters(par_train)
stim_train = Stimulus(par_train)

# testing set; for testing the 24 orientations
par_test = copy.deepcopy(par)
par_test = update_parameters(par_test)
stim_test = Stimulus(par_test)   # the argument `par` may be omitted

###### Train network ######
N_iter = 3000
N_save = 50  # save model every N_save iterations

# define model
hp = train_utils.hp
hp['learning_rate'] = 5e-1
model = Model(hp=hp)

ti_spec = train_utils.gen_ti_spec(stim_train.generate_trial())
alllosses =[]
performance_train = {'perf': [], 'loss': [], 'perf_loss': [], 'spike_loss': []}
performance_test = {'perf': [], 'loss': [], 'perf_loss': [], 'spike_loss': []}

for iter_n in range(N_iter+1):
    # train on testing set
    train_data = train_utils.tensorize_trial(stim_train.generate_trial())
    Y, Loss = model(train_data, hp)

    performance_train = train_utils.append_model_performance(performance_train,
                                                             train_data, Y, Loss, par_train)
    # append performance on testing set
    test_data = train_utils.tensorize_trial(stim_test.generate_trial())
    performance_test = train_utils.append_model_performance(performance_test,
                                                             test_data, Y, Loss, par_test)

    if (iter_n % N_save) == 0:
        print('\nTraining performance: ')
        train_utils.print_results(performance_train, iter_n)
        #print('Testing performance: ')
        #train_utils.print_results(performance_test, iter_n)

        tf.saved_model.save(model, os.path.join(model_dir, "cont_iter" + str(iter_n)))
