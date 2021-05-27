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
dt.hp['gain']         = 0  # 1e-2 : amplitude of random initialization (makes dynamics chaotic) 

## contrain the network(fixing some of the weights)
dt.hp['w_in_dm_fix']  = True  
dt.hp['w_in_em_fix']  = True  

dt.hp['w_rnn11_fix']  = True  # fix DM-to-DM recurrent matrix
dt.hp['w_rnn21_fix']  = True  # fix EM-to-DM recurrent matrix
dt.hp['w_rnn22_fix']  = False # fix EM-to-EM recurrent matrix

dt.hp['w_out_dm_fix'] = True  # fix linear voting from two separate populations
dt.hp['w_out_em_fix'] = True  # fix circular voting from two separate populations

dt.hp['EtoD_off']     = False # True: set W21=0, False: train W21
dt.hp['DtoE_off']     = False # True: set W12=0, False: train W12 # THIS IS IMPORTANT

dt.hp                 = dt.update_hp(dt.hp) # Do not forget this line
par                   = update_parameters(par)
stimulus              = Stimulus()
ti_spec               = dt.gen_ti_spec(stimulus.generate_trial())

## iteration
max_iter          = 100 # note that it really learns fast
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
# save_dir = "/Volumes/Data_CSNL/project/RNN_study/21-05-27/constrained_networks/network00"
# tf.saved_model.save(model, save_dir)