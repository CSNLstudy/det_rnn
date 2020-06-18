import tensorflow as tf
import numpy as np
from det_rnn import *

par['design'].update({'iti'     : (0, 5.5),
                      'stim'    : (5.5,7.0),
                      'delay'   : (7.0,23.5),
                      'estim'   : (23.5,28.0)})

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

model = Model()
model.__call__.get_concrete_function(
    trial_info=ti_spec,
    hp=hp_spec
)
model.rnn_model.get_concrete_function(
    input_data=ti_spec['neural_input'],
    hp=hp_spec
)

for iter in range(4000):
    trial_info = stimulus.generate_trial()
    for k, v in trial_info.items():
        trial_info[k] = tf.constant(v, name=k)
    Y, Loss = model(trial_info, hp)
    model_performance = append_model_performance(model_performance, trial_info, Y, Loss, par)
    if iter % 30 == 0:
        print_results(model_performance, iter)

model.model_performance = {'perf': tf.Variable(model_performance['perf'], trainable=False),
                           'loss': tf.Variable(model_performance['loss'], trainable=False),
                           'perf_loss': tf.Variable(model_performance['perf_loss'], trainable=False),
                           'spike_loss': tf.Variable(model_performance['spike_loss'], trainable=False)}

tf.saved_model.save(model, "/Volumes/Data_CSNL/project/RNN_study/20-06-19/HG/boost_wm/HL_booster9/")


load_model = tf.saved_model.load("/Volumes/Data_CSNL/project/RNN_study/20-06-19/HG/boost_wm/HL_booster9/")

for k,v in load_model.model_performance.items():
    model_performance[k] = v.numpy()


for iter in range(4000,4200):
    trial_info = stimulus.generate_trial()
    for k, v in trial_info.items():
        trial_info[k] = tf.constant(v, name=k)
    Y, Loss = load_model(trial_info, hp)
    model_performance = append_model_performance(model_performance, trial_info, Y, Loss, par)
    if iter % 30 == 0:
        print_results(model_performance, iter)






for iter in range(2000,5000):
    trial_info = stimulus.generate_trial()
    for k, v in trial_info.items():
        trial_info[k] = tf.constant(v, name=k)
    Y, Loss = load_model(trial_info, hp)
    model_performance = append_model_performance(model_performance, trial_info, Y, Loss, par)
    if iter % 30 == 0:
        print_results(model_performance, iter)





Y, H, Sx, _ = load_model.rnn_model(trial_info['neural_input'], hp)
Y2, H2, Sx2, _ = model.rnn_model(trial_info['neural_input'], hp)



par['design'].update({'iti': (0, 1.5),
                      'stim': (1.5, 3.0),
                      'delay': (3.0, 7.5),
                      'estim': (7.5, 9.0)})

par = update_parameters(par)
stimulus = Stimulus(par)
trial_info = stimulus.generate_trial()
for k, v in trial_info.items():
    trial_info[k] = tf.constant(v, name=k)

Y, H, Sx, _ = load_model.rnn_model(trial_info['neural_input'], hp)
Y2, H2, Sx2, _ = model.rnn_model(trial_info['neural_input'], hp)

