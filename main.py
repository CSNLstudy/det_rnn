from det_rnn import *
import det_rnn.train as dt

par = update_parameters(par)
stimulus = Stimulus()
ti_spec  = dt.gen_ti_spec(stimulus.generate_trial())

model_performance = {'perf_dm': [], 'perf_em': [], 'loss': [], 'perf_loss_dm': [], 'perf_loss_em': [], 'spike_loss': []}

model   = dt.initialize_rnn(ti_spec)
for iter in range(5000):
    trial_info = dt.tensorize_trial(stimulus.generate_trial())
    Y, Loss = model(trial_info, dt.hp)
    model_performance = dt.append_model_performance(model_performance, trial_info, Y, Loss, par)
    dt.print_results(model_performance, iter)
    if iter < 3000:
        if iter % 30 == 0:
            dt.print_results(model_performance, iter)
            if iter % 60 == 0:
                dt.hp['lam_estim'] = 0; dt.hp['lam_decision'] = 2400.  # note that lambda_estim and lambda_decision are heuristically set
            else:
                dt.hp['lam_estim'] = 300.; dt.hp['lam_decision'] = 0
    else:
        dt.hp['lam_estim'] = 300.; dt.hp['lam_decision'] = 2400.

model.model_performance = dt.tensorize_model_performance(model_performance)
# tf.saved_model.save(model, save_dir)
