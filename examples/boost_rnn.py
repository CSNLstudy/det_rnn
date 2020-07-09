import os, time, sys
import numpy as np
import tensorflow as tf

sys.path.append('../')
import det_rnn.train as dt
from det_rnn import *

model_dir = "/Volumes/Data_CSNL/project/RNN_study/20-06-26/HG/boost_wm/boost_wm_example"
# model_dir = "/Volumes/Data_CSNL/project/RNN_study/20-07-10/HG/boost/boost_PPC_strong"
model_dir = "/Users/hyunwoogu/Desktop/boost_wm_example2"
os.makedirs(model_dir, exist_ok=True)

par = update_parameters(par)
stimulus = Stimulus()

model_performance = {'perf': [], 'loss': [], 'perf_loss': [], 'spike_loss': []}

def pad_rule(trial_info, par=par):
    ni = trial_info['neural_input']
    rule_pad = np.zeros((ni.shape[0],ni.shape[1],par['n_hidden']))
    rule_neuron = ni.shape[2]
    rule_pad[par['design_rg']['estim'],:,rule_neuron:(rule_neuron+7)] = par['input_rule_strength']
    trial_info['neural_input'] = np.concatenate((ni,rule_pad),axis=2)
    return trial_info
ti_spec  = dt.gen_ti_spec(pad_rule(stimulus.generate_trial(),par))


# Boosting RNN
N_boost_max = 100000
perf_crit   = 0.95 # Human mean performance level
recency     = 50   # Number of 'recent' epochs to be assayed
boost_step  = 1.5  # How much step should we increase

extend_time = np.arange(boost_step,15.5,step=boost_step)
mileage_lim = len(extend_time)
milestones  = np.zeros((mileage_lim,), dtype=np.int64)
timestones  = np.zeros((mileage_lim,))

# Start boosting
model = dt.initialize_rnn(ti_spec) # initialize RNN to be boosted

mileage = -1
start_time = time.time()
print("RNN Booster started!")
for iter in range(N_boost_max):
    trial_info = dt.tensorize_trial(pad_rule(stimulus.generate_trial(),par))
    Y, Loss = model(trial_info, dt.hp)
    model_performance = dt.append_model_performance(model_performance, trial_info, Y, Loss, par)

    if iter % 30 == 0:
        dt.print_results(model_performance, iter)

    if  dt.level_up_criterion(iter,perf_crit,recency,milestones[mileage],model_performance):
        check_time = time.time()
        mileage += 1
        if mileage >= mileage_lim:
            print("#"*80+"\nTraining criterion finally met!(Time Spent: {:0.2f}s)\t".format(check_time-start_time)+
                  "Now climb down the mountain!\n"+"#"*80)
            break
        milestones[mileage] = iter
        timestones[mileage] = check_time-start_time

        ## Attach to the model
        model.model_performance = dt.tensorize_model_performance(model_performance)
        model.milestones = tf.Variable(milestones, trainable=False)
        model.timestones = tf.Variable(timestones, trainable=False)

        ## save the model
        os.makedirs(model_dir + "/model_level" + str(mileage), exist_ok=True)
        tf.saved_model.save(model, model_dir  + "/model_level" + str(mileage))

        ## upgrade to higher level
        par['design'].update({'iti': (0, 1.5),
                              'stim': (1.5, 3.0),
                              'delay': (3.0, 4.5 + extend_time[mileage]),
                              'estim': (4.5 + extend_time[mileage], 6.0 + extend_time[mileage])})
        par = update_parameters(par)
        stimulus = Stimulus(par)

        ## modulate hyperparameters #######################################################
        # dt.hp['spike_cost'] /= 2.
        ###################################################################################

        ## Report an upgrade has been performed
        print("#"*80+"\nCriterion satisfied!(Time Spent: {:0.2f}s)\t".format(check_time-start_time)+
                     "Now extending: {:0.1f}\n".format(extend_time[mileage])+"#"*80)
