import numpy as np
import tensorflow as tf
from .model import Model
from .hyper import hp_spec

__all__ = ['initialize_rnn', 'append_model_performance', 'print_results',
           'tensorize_trial', 'numpy_trial',
           'tensorize_model_performance',
           'level_up_criterion', 'cull_criterion', 'gen_ti_spec']

def initialize_rnn(ti_spec,hp_spec=hp_spec):
    model = Model()
    model.__call__.get_concrete_function(
        trial_info=ti_spec,
        hp=hp_spec
    )
    model.rnn_model.get_concrete_function(
        input_data=ti_spec['neural_input'],
        hp=hp_spec
    )
    return model

def append_model_performance(model_performance, trial_info, Y, Loss, par):
    estim_perf = _get_eval(trial_info, Y, par)
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

def tensorize_trial(trial_info):
    for k, v in trial_info.items():
        trial_info[k] = tf.constant(v, name=k)
    return trial_info

def numpy_trial(trial_info):
    for k, v in trial_info.items():
        trial_info[k] = v.numpy()
    return trial_info

def tensorize_model_performance(model_performance):
    tensor_mp = {'perf': tf.Variable(model_performance['perf'], trainable=False),
                 'loss': tf.Variable(model_performance['loss'], trainable=False),
                 'perf_loss': tf.Variable(model_performance['perf_loss'], trainable=False),
                 'spike_loss': tf.Variable(model_performance['spike_loss'], trainable=False)}
    return tensor_mp

def level_up_criterion(iter,perf_crit,recency,milestone,model_performance):
    indx_recent = np.arange(iter)[-recency:]
    indx_milest = np.arange(iter)[milestone:]
    indx_inters = np.intersect1d(indx_recent, indx_milest)
    perf_vec = np.array(model_performance['perf'])[indx_inters]
    return (np.mean(perf_vec) > perf_crit) & (len(perf_vec) >= recency)

def cull_criterion(iter,fall_crit,recency,milestone,model_performance):
    indx_recent = np.arange(iter)[-recency:]
    indx_milest = np.arange(iter)[milestone:]
    indx_inters = np.intersect1d(indx_recent, indx_milest)
    perf_vec = np.array(model_performance['perf'])[indx_inters]
    return (np.mean(perf_vec) < fall_crit) & (len(perf_vec) >= recency)


def gen_ti_spec(trial_info) :
    ti_spec = {}
    for k, v in trial_info.items():
        _shape = list(v.shape)
        if len(_shape) > 1: _shape[0] = None
        ti_spec[k] = tf.TensorSpec(_shape, tf.dtypes.as_dtype(v.dtype), name=k)
    return ti_spec

#
def _get_eval(trial_info, output, par):
    cenoutput = tf.nn.softmax(output, axis=2).numpy()
    post_prob = cenoutput[:, :, par['n_rule_output']:] #shape = T X B X neurons
    post_prob = post_prob / (
            np.sum(post_prob, axis=2, keepdims=True) + np.finfo(np.float32).eps)  # Dirichlet normalization
    post_support = np.linspace(0, np.pi, post_prob.shape[2], endpoint=False) + np.pi / post_prob.shape[2] / 2
    pseudo_mean = np.arctan2(post_prob @ np.sin(2 * post_support),
                             post_prob @ np.cos(2 * post_support)) / 2
    estim_sinr = (np.sin(2 * pseudo_mean[par['design_rg']['estim'], :])).mean(axis=0)
    estim_cosr = (np.cos(2 * pseudo_mean[par['design_rg']['estim'], :])).mean(axis=0)
    estim_mean = np.arctan2(estim_sinr, estim_cosr) / 2
    perf = np.mean(np.cos(2. * (trial_info['stimulus_ori'].numpy() * np.pi / post_prob.shape[2] - estim_mean)))
    return perf

