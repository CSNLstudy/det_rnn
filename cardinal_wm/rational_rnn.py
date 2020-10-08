import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append('../')
from det_rnn import *
import det_rnn.train as dt
import det_rnn.analysis as da

## Rational account of cardinal bias: RNNs trained with discrimination cost
discrim_template = np.kron(np.eye(2), np.ones(12)).T

def GirshickPrior(theta, radian=True):
    if radian:
        p_theta = (2. - np.abs(np.sin(theta * 2.))) / (2. * (np.pi - 1.))
    else: # degree
        p_theta = (2. - np.abs(np.sin(theta * np.pi/90. ))) / (360./np.pi * (np.pi - 1.))
    return p_theta

def desired_discrim(theta, power=10):
    # arr = np.zeros((12,1)); arr[0] = 1. # "infinity" case
    arr = (GirshickPrior(np.arange(0., 24., step=2.) / 48. * np.pi)[:, np.newaxis] + .5) ** power
    res = arr * np.roll(discrim_template, int(13 + theta), axis=0)[:12]
    return res

def decisionize(trial_info, par=par):
    decision_output = np.zeros(trial_info['desired_estim'].shape[:2] + (12, 3))
    decision_output[:,:,:,0] = 1. # pad rule component
    decision_output[par['design_rg']['estim'],:,:,0] = 0
    for i,s in enumerate(trial_info['stimulus_ori']):
        decision_output[par['design_rg']['estim'],i,:,1:] = desired_discrim(s)
    return decision_output

def decisionize_ti(trial_info, par=par):
    trial_info['desired_output_it'] = decisionize(trial_info, par=par)
    trial_info['mask_it'] = trial_info['mask'][:,:,:int(par['n_output']/2),np.newaxis]
    return trial_info


## Training : please ignore WARNINGs (due to None gradient problems)
par = update_parameters(par)
stimulus = Stimulus()
trial_info = decisionize_ti(stimulus.generate_trial())

ti_spec  = dt.gen_ti_spec(decisionize_ti(stimulus.generate_trial()))
model_performance = {'perf': [], 'loss': [], 'perf_loss': [], 'spike_loss': []}

model = dt.initialize_rnn(ti_spec)
task_type = 'Discrim'
for iter in range(1500):
    trial_info = dt.tensorize_trial(decisionize_ti(stimulus.generate_trial()))
    Y, Loss = model(trial_info, dt.hp)
    model_performance = dt.append_model_performance(model_performance, trial_info, Y, Loss, par, dt.hp['task_type'])
    if iter % 30 == 0:
        dt.print_results(model_performance, iter, task_type)
        if iter % 60 == 0:
            dt.hp['task_type'] = 0; task_type = 'Discrim'
        else:
            dt.hp['task_type'] = 1; task_type = 'Estim'


## save model
model.model_performance = dt.tensorize_model_performance(model_performance)
tf.saved_model.save(model, "/Volumes/Data_CSNL/project/RNN_study/20-07-24/HG/output/example_alternation")




