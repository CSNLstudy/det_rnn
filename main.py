from det_rnn import *
import det_rnn.train as dt

par = update_parameters(par)
stimulus = Stimulus()
ti_spec  = dt.gen_ti_spec(stimulus.generate_trial())

model_performance = {'perf': [], 'loss': [], 'perf_loss': [], 'spike_loss': []}

model = dt.initialize_rnn(ti_spec) # initialize RNN to be boosted
for iter in range(3000):
    trial_info = dt.tensorize_trial(stimulus.generate_trial())
    Y, Loss = model(trial_info, dt.hp)
    model_performance = dt.append_model_performance(model_performance, trial_info, Y, Loss, par)
    if iter % 30 == 0:
        dt.print_results(model_performance, iter)

model.model_performance = dt.tensorize_model_performance(model_performance)
# tf.saved_model.save(model, save_dir)