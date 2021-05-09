import sys
sys.path.append('../')
from det_rnn import *
import det_rnn.train as dt

## "shorter" structure
par['design'].update({'iti'  : (0, 0.3),
                      'stim' : (0.3, 0.6),                      
                      'decision': (0.9, 1.2),
                      'delay'   : ((0.6, 0.9),(1.2, 1.5)),
                      'estim' : (1.5, 1.8)})

## parameter adjustment 
par['dt']             = 20
dt.hp['dt']           = 20
dt.hp['gain']         = 1e-2  # amplitude of random initialization (makes dynamics chaotic) 
dt.hp['w_out_dm_fix'] = True  # assume linear voting from two separate populations
dt.hp['DtoE_off']     = True  # for simplicity

dt.hp                 = dt.update_hp(dt.hp)
par                   = update_parameters(par)
stimulus              = Stimulus()
ti_spec               = dt.gen_ti_spec(stimulus.generate_trial())

## iteration: stop manually
max_iter          = 10000
n_print           = 10
model             = dt.initialize_rnn(ti_spec)
model_performance = {'perf_dm': [], 'perf_em': [], 'loss': [], 'perf_loss_dm': [], 'perf_loss_em': [], 'spike_loss': []}

for iter in range(max_iter):
    trial_info        = dt.tensorize_trial(stimulus.generate_trial())
    Y, Loss           = model(trial_info, dt.hp)
    model_performance = dt.append_model_performance(model_performance, trial_info, Y, Loss, par)
    
    # Print
    if iter % n_print == 0: dt.print_results(model_performance, iter)
    
model.model_performance = dt.tensorize_model_performance(model_performance)

# import tensorflow as tf
# save_dir = "/Volumes/ROOT/CSNL_temp/HG/Analysis/RNN/modular/networks/short_separate/sigmoid/network00"
# tf.saved_model.save(model, save_dir)
