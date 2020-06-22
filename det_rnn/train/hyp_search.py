import os, time, sys
import numpy as np
import tensorflow as tf

sys.path.append('../../')
import det_rnn.train as dt
from det_rnn import *

def

def train(stimulus, train_params, hyper):
    ti_spec = dt.gen_ti_spec(stimulus.generate_trial())

    model_performance = {'perf': [], 'loss': [], 'perf_loss': [], 'spike_loss': []}

    model = dt.initialize_rnn(ti_spec)  # initialize RNN to be boosted
    for iter in range(3000):
        trial_info = dt.tensorize_trial(stimulus.generate_trial())
        Y, Loss = model(trial_info, dt.hp)
        model_performance = dt.append_model_performance(model_performance, trial_info, Y, Loss, par)
        if iter % 30 == 0:
            dt.print_results(model_performance, iter)

    model.model_performance = dt.tensorize_model_performance(model_performance)
    # tf.saved_model.save(model, save_dir)

if __name__ == '__main__':
    hyperparam_list = {}
    hyperparam_list['n_tuned_input'] = [24, 72, 128, 512] # belongs in params.
    hyperparam_list['n_hidden'] = [50, 100, 500, 1000, 3000]
    hyperparam_list['spike_cost'] = [2e-7, 2e-5, 2e-3, 2e-1] #i.e. sparsity
    hyperparam_list['exc_inh_prop'] = [0.9, 0.8, 0.6, 0.4]# EI proportion

    hyperdict = {} #
    for key, value in hyperparam_list.items():
        hyperdict[1]