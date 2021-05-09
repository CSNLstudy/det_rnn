from det_rnn import *
import det_rnn.train as dt

## "standard" structure
par['design'].update({'iti'  : (0, 1.5),
                      'stim' : (1.5, 3.0),                      
                      'decision': (4.5, 6.0),
                      'delay'   : ((3.0, 4.5),(6.0, 7.5)),
                      'estim' : (7.5,9.0)})

## parameter adjustment 
par['dt']      = 20
dt.hp['dt']    = 20
dt.hp['gain']  = 1e-2  # amplitude of random initialization (makes dynamics chaotic) 

dt.hp          = dt.update_hp(dt.hp)
par            = update_parameters(par)
stimulus       = Stimulus()
ti_spec        = dt.gen_ti_spec(stimulus.generate_trial())

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
# tf.saved_model.save(model, save_dir)