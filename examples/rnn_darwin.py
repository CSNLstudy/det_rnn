import os, sys
import numpy as np
import tensorflow as tf

sys.path.append('../')
import det_rnn.train as dt
from det_rnn import *

model_dir = "/Volumes/Data_CSNL/project/RNN_study/20-06-19/HG/boost_wm/HL_booster11"
os.makedirs(model_dir, exist_ok=True)

par = update_parameters(par)
stimulus = Stimulus()
ti_spec = dt.gen_ti_spec(stimulus.generate_trial())

model_performance = {'perf': [], 'loss': [], 'perf_loss': [], 'spike_loss': []}

# Resume boosting
N_boost_max = 100000
perf_crit   = 0.95 # Human mean performance level
fall_crit   = 0.7  # Resume the current level
recency     = 50   # Number of 'recent' epochs to be assayed
boost_step  = 1.5  # How much step should we increase

extend_time = np.arange(boost_step,15.5,step=boost_step)
mileage_lim = len(extend_time)

## Resume model
mileage = 7 # restart from level 7
model = tf.saved_model.load(model_dir + "/model_level" + str(mileage))

## load model_performance
for k,v in model.model_performance.items():
    model_performance[k] = v.numpy().tolist()

N_resume    = len(model_performance['perf'])
milestones  = model.milestones.numpy()
timestones  = model.timestones.numpy()

par['design'].update({'iti': (0, 1.5),
                      'stim': (1.5, 3.0),
                      'delay': (3.0, 4.5 + extend_time[mileage]),
                      'estim': (4.5 + extend_time[mileage], 6.0 + extend_time[mileage])})
## modulate hyperparameters #######################################################
# dt.hp['spike_cost'] = tf.constant(2e-10, name='spike_cost')
# par['mask']['estim'] = 8.
###################################################################################
par = update_parameters(par)
stimulus = Stimulus(par)

print("RNN Booster resumed from extend_time=" + str(extend_time[mileage]) + "!")
iter = N_resume
while iter < N_boost_max:
    trial_info = dt.tensorize_trial(stimulus.generate_trial())
    Y, Loss = model(trial_info, dt.hp)
    model_performance = dt.append_model_performance(model_performance, trial_info, Y, Loss, par)
    if iter % 30 == 0:
        dt.print_results(model_performance, iter)

    if  dt.level_up_criterion(iter, perf_crit, recency, milestones[mileage], model_performance):
        mileage += 1
        if mileage >= mileage_lim:
            print("#"*80+"\nTraining criterion finally met!\t"+
                  "Now climb down the mountain!\n"+"#"*80)
            break
        milestones[mileage] = iter

        ## Attach to the model
        model.model_performance = dt.tensorize_model_performance(model_performance)
        model.milestones = tf.Variable(milestones, trainable=False)

        ## save the model
        os.makedirs(model_dir + "/model_level" + str(mileage), exist_ok=True)
        tf.saved_model.save(model, model_dir + "/model_level" + str(mileage))

        ## upgrade to higher level
        par['design'].update({'iti': (0, 1.5),
                              'stim': (1.5, 3.0),
                              'delay': (3.0, 4.5 + extend_time[mileage]),
                              'estim': (4.5 + extend_time[mileage], 6.0 + extend_time[mileage])})
        ## modulate hyperparameters #######################################################
        # par['mask']['estim'] = 8.
        ###################################################################################
        par = update_parameters(par)
        stimulus = Stimulus(par)

        ## Report an upgrade has been performed
        print("#"*80+"\nCriterion satisfied!\t"+
                     "Now extending: {:0.1f}\n".format(extend_time[mileage])+"#"*80)

    iter += 1

    if dt.cull_criterion(iter, fall_crit, recency, milestones[mileage], model_performance):
        print("#" * 80 + "\nModel Failed! Retried!\t" +
              "Again training : {:0.1f}\n".format(extend_time[mileage]) + "#" * 80)
        model = tf.saved_model.load(model_dir + "/model_level" + str(mileage))
        model_performance = {'perf': [], 'loss': [], 'perf_loss': [], 'spike_loss': []}
        for k,v in model.model_performance.items():
            model_performance[k] = v.numpy().tolist()

        N_resume    = len(model_performance['perf'])
        milestones  = model.milestones.numpy()
        iter        = N_resume

