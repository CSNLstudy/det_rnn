from det_rnn import *
import det_rnn.train as dt
import os, sys
import tensorflow as tf
import pickle

par = update_parameters(par)
stimulus = Stimulus()
ti_spec  = dt.gen_ti_spec(stimulus.generate_trial())

model_performance = {'perf': [], 'loss': [], 'perf_loss': [], 'dec_loss':[], 'spike_loss': []}

model = dt.initialize_rnn(ti_spec) # initialize RNN to be boosted
for iter in range(5000):
    trial_info = dt.tensorize_trial(stimulus.generate_trial())
    Y, Loss = model(trial_info, dt.hp)
    model_performance = dt.append_model_performance(model_performance, trial_info, Y, Loss, par)
    if iter % 30 == 0:
        dt.print_results(model_performance, iter)

model.model_performance = dt.tensorize_model_performance(model_performance)
model_dir = "/Users/eva/Dropbox/det_rnn_addingDM_200625/results"
with open(model_dir + '/basic_DMplugged_longIt.pkl','wb') as f:
    pickle.dump(model,f)


# model_dir = "/Users/eva/Dropbox/det_rnn/results"
# os.makedirs(model_dir, exist_ok=True)
# os.makedirs(model_dir + "/basic" , exist_ok=True)
# tf.saved_model.save(model, model_dir)