import sys, os, time

import copy
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')
from det_rnn import *
import det_rnn.train as utils_train
import det_rnn.analysis as utils_analysis
import tensorflow as tf
from det_rnn.train.model import Model
from utils.plotfnc import *

#model_dir = "/Users/JRyu/github/det_rnn/experiments/naturalprior/200622/"
#model_dir = "D:/proj/det_rnn/experiments/naturalprior/200622/"
model_dir = "../experiments/naturalprior/200622/"
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
N_iter = 10000
N_save = N_iter/100  # save model every N_save iterations

# define model
hp = utils_train.hp
hp['learning_rate'] = 5e-1
model = Model(hp=hp)

ti_spec = utils_train.gen_ti_spec(stim_train.generate_trial())

alllosses =[]
performance_train = {'perf': [], 'loss': [], 'perf_loss': [], 'spike_loss': []}
performance_test = {'perf': [], 'loss': [], 'perf_loss': [], 'spike_loss': []}

for iter_n in range(N_iter+1):
    # train on testing set
    train_data = utils_train.tensorize_trial(stim_train.generate_trial())
    Y, Loss = model(train_data, hp)

    performance_train = utils_train.append_model_performance(performance_train,
                                                             train_data, Y, Loss, par_train)
    # append performance on testing set
    test_data = utils_train.tensorize_trial(stim_test.generate_trial(balanced=True))
    test_Y, test_loss = model.return_losses(test_data['neural_input'], test_data['desired_output'],
                                            test_data['mask'], hp)
    performance_test = utils_train.append_model_performance(performance_test,
                                                            test_data, test_Y, test_loss, par_test)

    print('\n Training performance: ')
    utils_train.print_results(performance_train, iter_n)
    print('Testing performance: ')
    utils_train.print_results(performance_test, iter_n)

    if (iter_n % N_save) == 0:
        # predict from testing set
        plot_rnn_output(par_test, test_data, test_Y, stim_test, TEST_TRIAL=None)
        test_Y = utils_analysis.softmax_pred_output(test_Y) # change to softmax
        ground_truth, estim_mean, raw_error, beh_perf = utils_analysis.behavior_summary(test_data, test_Y, par=par_test)
        utils_analysis.behavior_figure(ground_truth, estim_mean, raw_error, beh_perf)

        tf.saved_model.save(model, os.path.join(model_dir, "cont_iter" + str(iter_n)))

    if iter_n > 20 and np.sum(np.array(performance_test['perf'][-20:]) > 0.95) > 18:
        print('Good performance achieved... saving final model')
        break

# save final model
tf.saved_model.save(model, os.path.join(model_dir, "cont_iter" + str(iter_n)))

# plot performance
plt.figure()
plt.plot(np.arange(100,len(performance_train['perf'])),
         performance_train['perf'][100:], label='Train')
plt.plot(np.arange(100,len(performance_test['perf'])),
         performance_test['perf'][100:], label='Test (balanced)')
plt.ylabel('Performance')
plt.ylabel('Iterations')
plt.legend()
plt.show()

plt.figure()
plt.plot(np.arange(100,len(performance_train['perf'])),
         performance_train['loss'][100:], label='Train')
plt.plot(np.arange(100,len(performance_test['perf'])),
         performance_test['loss'][100:], label='Test (balanced)')
plt.ylabel('Loss')
plt.ylabel('Iterations')
plt.yscale('log')
plt.legend()
plt.show()