import os, time
import numpy as np
import tensorflow as tf
from det_rnn import *

###########################################################################################################
# Preparation #############################################################################################
###########################################################################################################
par = update_parameters(par)
stimulus = Stimulus()
trial_info = stimulus.generate_trial()

model_performance = {'perf': [], 'loss': [], 'perf_loss': [], 'spike_loss': []}

def get_eval(trial_info, output, par):
    cenoutput = tf.nn.softmax(output, axis=2).numpy()
    post_prob = cenoutput[:, :, par['n_rule_output']:]
    post_prob = post_prob / (
            np.sum(post_prob, axis=2, keepdims=True) + np.finfo(np.float32).eps)  # Dirichlet normaliation
    post_support = np.linspace(0, np.pi, par['n_ori'], endpoint=False) + np.pi / par['n_ori'] / 2
    pseudo_mean = np.arctan2(post_prob @ np.sin(2 * post_support),
                             post_prob @ np.cos(2 * post_support)) / 2
    estim_sinr = (np.sin(2 * pseudo_mean[par['design_rg']['estim'], :])).mean(axis=0)
    estim_cosr = (np.cos(2 * pseudo_mean[par['design_rg']['estim'], :])).mean(axis=0)
    estim_mean = np.arctan2(estim_sinr, estim_cosr) / 2
    perf = np.mean(np.cos(2. * (trial_info['stimulus_ori'].numpy() * np.pi / par['n_ori'] - estim_mean)))
    return perf

def append_model_performance(model_performance, trial_info, Y, Loss, par):
    estim_perf = get_eval(trial_info, Y, par)
    model_performance['loss'].append(Loss['loss'].numpy())
    model_performance['perf_loss'].append(Loss['perf_loss'].numpy())
    model_performance['spike_loss'].append(Loss['spike_loss'].numpy())
    model_performance['perf'].append(estim_perf)
    return model_performance

def print_results(model_performance, iteration):
    print_res = 'Iter. {:4d}'.format(iteration)
    print_res += ' | Performance {:0.4f}'.format(model_performance['perf'][iteration]) + \
                 ' | Loss {:0.4f}'.format(model_performance['loss'][iteration])
    print_res += ' | Spike loss {:0.4f}'.format(model_performance['spike_loss'][iteration])
    print(print_res)

# Tensorize hp
for k, v in hp.items():
    hp[k] = tf.constant(v, name=k)

hp_spec = {}
for k, v in hp.items():
    hp_spec[k] = tf.TensorSpec(v.numpy().shape, tf.dtypes.as_dtype(v.numpy().dtype), name=k)

ti_spec = {}
for k, v in trial_info.items():
    _shape = list(v.shape)
    if len(_shape) > 1: _shape[0] = None
    ti_spec[k] = tf.TensorSpec(_shape, tf.dtypes.as_dtype(v.dtype), name=k)




###########################################################################################################
# Boosting ################################################################################################
###########################################################################################################
# Birth of a babyrnn
model = Model()
model.__call__.get_concrete_function(
    trial_info=ti_spec,
    hp=hp_spec
)
model.rnn_model.get_concrete_function(
    input_data=ti_spec['neural_input'],
    hp=hp_spec
)

## boost
N_boost_max = 100000
perf_crit   = 0.95 # Human mean performance level
recency     = 50   # Number of 'recent' epochs to be assayed
boost_step  = 1.5  # How much step should we increase

## prepare
extend_time = np.arange(boost_step,15.5,step=boost_step)
mileage_lim = len(extend_time)
milestones  = np.zeros((mileage_lim,), dtype=np.int64)
timestones  = np.zeros((mileage_lim,))

# base_path  = "/Volumes/Data_CSNL/project/RNN_study/20-06-19/HG/boost_wm/"
base_path  = "/Users/hyunwoo/Desktop/"
model_code = "HL_booster11"
os.makedirs(base_path+model_code, exist_ok=True)

##
mileage = -1
start_time = time.time()


## Start training #########################################################################################
print("RNN Booster started!")
for iter in range(N_boost_max):
    trial_info = stimulus.generate_trial()
    for k, v in trial_info.items():
        trial_info[k] = tf.constant(v, name=k)
    Y, Loss = model(trial_info, hp)
    model_performance = append_model_performance(model_performance, trial_info, Y, Loss, par)
    if iter % 30 == 0:
        print_results(model_performance, iter)

    indx_recent = np.arange(iter)[-recency:]
    indx_milest = np.arange(iter)[milestones[mileage]:]
    indx_inters = np.intersect1d(indx_recent,indx_milest)
    perf_vec    = np.array(model_performance['perf'])[indx_inters]

    if  (np.mean(perf_vec) > perf_crit) & (len(perf_vec) >= recency):
        check_time = time.time()
        mileage += 1
        if mileage >= mileage_lim:
            print("#"*80+"\nTraining criterion finally met!(Time Spent: {:0.2f}s)\t".format(check_time-start_time)+
                  "Now climb down the mountain!\n"+"#"*80)
            break
        milestones[mileage] = iter
        timestones[mileage] = check_time-start_time

        ## Attach to the model
        model.model_performance = {'perf': tf.Variable(model_performance['perf'], trainable=False),
                                   'loss': tf.Variable(model_performance['loss'], trainable=False),
                                   'perf_loss': tf.Variable(model_performance['perf_loss'], trainable=False),
                                   'spike_loss': tf.Variable(model_performance['spike_loss'], trainable=False)}

        model.milestones = tf.Variable(milestones, trainable=False)
        model.timestones = tf.Variable(timestones, trainable=False)

        ## save the model
        os.makedirs(base_path + model_code + "/model_level" + str(mileage), exist_ok=True)
        tf.saved_model.save(model, base_path + model_code + "/model_level" + str(mileage))


        ## upgrade to higher level
        par['design'].update({'iti': (0, 1.5),
                              'stim': (1.5, 3.0),
                              'delay': (3.0, 4.5 + extend_time[mileage]),
                              'estim': (4.5 + extend_time[mileage], 6.0 + extend_time[mileage])})
        par = update_parameters(par)
        stimulus = Stimulus(par)

        ## modulate hyperparameters
        # hp['spike_cost'] /= 2.

        ## Report an upgrade has been performed
        print("#"*80+"\nCriterion satisfied!(Time Spent: {:0.2f}s)\t".format(check_time-start_time)+
                     "Now extending: {:0.1f}\n".format(extend_time[mileage])+"#"*80)


