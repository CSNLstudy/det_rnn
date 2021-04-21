import sys, pickle, copy, cmath

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import logging, requests

from scipy.stats import circstd, circmean

#logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)
#logging.getLogger('requests').setLevel(logging.DEBUG)
# disable annoying debugs from matpltlib! do this before importing matplotlib
# https://stackoverflow.com/questions/35325042/python-logging-disable-logging-from-imported-modules
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib').disabled = True
logging.getLogger('matplotlib.font_manager').disabled = True
#logging.getLogger('DEBUG').disabled = True

def behavior_summary(trial_info, test_outputs, parOrStim, BatchIdx = None):
    '''
    trial_info:
    parOrStim: par or stim struct for testing.

    output:
    estimation:
    - est_mean: (range: -pi/2 to pi/2) The estimated angle in radians of the network output. Averaged across time and trials
    - est_target: (range: 0 to pi). The target of estimate
    - est_error: (range: -pi/2 to pi/2) est_mean - target
    - est_perf: (range: -1 to 1) cos(est_error). 1 is the best.

    decision:
    - dec_target
    - dec_mean
    - dec_end
    - dec_meanError
    dec_endError
    dec_mean_perf
    dec_end_perf
    '''
    if type(parOrStim) is not dict:  # inherit from stimulus class
        par = {}
        par['n_ori'] = parOrStim.n_ori
        par['n_tuned_output'] = parOrStim.n_tuned_output
        par['n_rule_output_em'] = parOrStim.n_rule_output_em
        par['n_rule_output_dm'] = parOrStim.n_rule_output_dm
        par['design_rg'] = parOrStim.design_rg
        par['input_rule_rg'] = parOrStim.input_rule_rg
    else:  # it the par struct
        par = parOrStim

    if BatchIdx is not None:
        # collect batch with the right idx
        est_output          = tf.gather(test_outputs['est_output'], BatchIdx,axis=1)
        dec_output          = tf.gather(test_outputs['dec_output'], BatchIdx,axis=1)
        ground_truth        = tf.gather(trial_info['stimulus_ori'],BatchIdx)
        desired_decision    = tf.gather(trial_info['desired_decision'],BatchIdx,axis=1)
    else:
        est_output          = test_outputs['est_output']
        dec_output          = test_outputs['dec_output']
        ground_truth        = trial_info['stimulus_ori']
        desired_decision    = trial_info['desired_decision']

        # estimation
    if len(est_output.shape) == 2:  # single tuning (est_output.shape is B x neurons)
        est_output = est_output
        cenoutput = tf.nn.softmax(est_output, axis=1)
        cenoutput = cenoutput.numpy()
        post_prob = cenoutput
    else:  # time series (pred_output.shape is B x T x neurons)
        est_output = tf.gather(est_output, par['input_rule_rg']['estimation'],
                               axis=0)  # collect estimation times
        cenoutput = tf.nn.softmax(est_output, axis=2)
        cenoutput = cenoutput.numpy()
        post_prob = cenoutput  # posterior mean as a function of time
        # Dirichlet normaliation todo (josh): remove, or add eps before softmax?
        post_prob = post_prob / (np.sum(post_prob, axis=2, keepdims=True) + np.finfo(np.float32).eps)

    # post_support = np.linspace(0, np.pi, par['n_tuned_output'], endpoint=False) + np.pi / par['n_ori'] / 2 # hmm??why add the np.pi?
    # Convert domain from (0,pi) => (0, 2pi) => circular mean of the posterior (-pi, pi)
    # take the (circular) posterior mean; then convert to 0 to pi.
    post_support = np.arange(0, np.pi, np.pi / par['n_tuned_output'])  # 0 to pi
    post_sinr = np.sin(2 * post_support)
    post_cosr = np.cos(2 * post_support)
    post_mean = np.arctan2(post_prob @ post_sinr, post_prob @ post_cosr) / 2  # -pi/2 to pi/2

    if len(est_output.shape) == 2:  # single tuning
        estim_mean = post_mean  # 0 to pi
    else:  # time series
        # posterior mean collapsed along time
        estim_sinr = (np.sin(2 * post_mean)).mean(axis=0)
        estim_cosr = (np.cos(2 * post_mean)).mean(axis=0)
        estim_mean = np.arctan2(estim_sinr, estim_cosr) / 2  # -pi to pi => -pi/2 to pi/2

    assert np.all(estim_mean <= np.pi/2) and np.all(estim_mean >= - np.pi/2)

      # 0 to (n_tuned_output-1)
    if tf.is_tensor(ground_truth):
        ground_truth = tf.make_ndarray(tf.make_tensor_proto(ground_truth))

    estim_target = ground_truth * np.pi / par['n_ori']  # range: 0 to pi
    estim_error = ((estim_mean - estim_target + np.pi/2) % np.pi) - np.pi/2  # -pi/2 to pi/2
    estim_perf = np.cos(2 * estim_error)  # -1 to 1

    est_summary = {'est_mean': estim_mean,
                   'est_target': estim_target,
                   'est_error': estim_error,
                   'est_perf': estim_perf}

    dec_output = tf.gather(dec_output, par['input_rule_rg']['decision'], axis=0)  # collect estimation times
    cenoutput = tf.nn.softmax(dec_output[:, :, :], axis=2)
    cenoutput = cenoutput.numpy()
    post_prob = cenoutput  # posterior mean as a function of time

    # Dirichlet normaliation todo (josh): remove, or add eps before softmax?
    dec_mean = post_prob / (np.sum(post_prob, axis=2, keepdims=True) + np.finfo(np.float32).eps)
    dec_mean = tf.reduce_mean(dec_mean,axis=0) # average over time.

    dec_target = tf.gather(desired_decision[:, :, par['n_rule_output_dm']:], par['input_rule_rg']['decision'],
                           axis=0)
    dec_target = tf.reduce_mean(dec_target,axis=0)
    dec_error = np.sum(dec_mean * dec_target, axis=-1)
    dec_perf = (np.argmax(dec_mean, axis=-1) == np.argmax(dec_target, axis=-1)).astype(float)

    dec_summary = {'dec_mean': dec_mean,
                   'dec_target': dec_target,
                   'dec_error': dec_error,
                   'dec_perf': dec_perf}

    # error: decision weighted by the probability

    return est_summary, dec_summary

def estimation_decision(test_data, test_outputs, stim_test):
    est_summary, dec_summary = behavior_summary(test_data, test_outputs, stim_test)

    # condition on reference
    a = dec_summary['dec_mean']
    CW_idx = (a[:, 0] > a[:, 1])  # unit 0 is activated if ref < 0 => stim is CW to the ref
    CCW_idx = (a[:, 0] < a[:, 1])  # unit 0 is activated if ref < 0 => stim is CW to the ref

    collist = ['CW', 'Ref',
               'est_mean', 'est_target', 'est_error', 'est_perf',
               'dec_error', 'dec_perf']
    dflist = []

    collist2 = ['CW', 'Ref', 'mean', 'sd']
    data2 = []

    # df = pd.DataFrame(data, columns = collist)
    for ref_ori in stim_test.reference:
        # clockwise
        batchmask_cw = ((test_data['reference_ori'] == ref_ori) & CW_idx)
        batchidx_cw = tf.squeeze(tf.where(batchmask_cw),axis=-1)
        est_summary, dec_summary = behavior_summary(test_data, test_outputs, stim_test, batchidx_cw)
        data = [(True, ref_ori * 180 / stim_test.n_ori,
                 est_summary['est_mean'][i] * 180 / np.pi, est_summary['est_target'][i],
                 est_summary['est_error'][i]* 180 / np.pi, est_summary['est_perf'][i],
                 dec_summary['dec_error'][i], dec_summary['dec_perf'][i])
                for i in range(sum(batchmask_cw.numpy()))]
        toappend = pd.DataFrame(data, columns=collist)
        dflist += [toappend]

        data2 += [(True, ref_ori * 180 / stim_test.n_ori,
                   circmean(est_summary['est_error'] * 180 / np.pi, low=-90,high=90),
                   circstd(est_summary['est_error'] * 180 / np.pi, low=0,high=180))]

        # counter clockwise
        batchmask_ccw = (test_data['reference_ori'] == ref_ori) & CCW_idx
        batchidx_ccw = tf.squeeze(tf.where(batchmask_ccw),axis=-1)
        est_summary, dec_summary = behavior_summary(test_data, test_outputs, stim_test, batchidx_ccw)
        data = [(False, ref_ori * 180 / stim_test.n_ori,
                 est_summary['est_mean'][i] * 180 / np.pi, est_summary['est_target'][i],
                 est_summary['est_error'][i]* 180 / np.pi, est_summary['est_perf'][i],
                 dec_summary['dec_error'][i], dec_summary['dec_perf'][i])
                for i in range(sum(batchmask_ccw.numpy()))]
        toappend = pd.DataFrame(data, columns=collist)
        dflist += [toappend]

        data2 += [(False, ref_ori * 180 / stim_test.n_ori,
                   circmean(est_summary['est_error'] * 180 / np.pi, low=-90,high=90),
                   circstd(est_summary['est_error'] * 180 / np.pi, low=0,high=180))]

    df = pd.concat(dflist, ignore_index=True)
    df2 = pd.DataFrame(data2, columns=collist2)

    if np.all(df['CW'].to_numpy()) or np.all(np.logical_not(df['CW'].to_numpy())):
        # if everything is true or false, cheat a bit
        # otherwise, we can't plot stuff.
        fakedf = pd.DataFrame([(False, 0, 0 , 0, 0, 0, 0, 0), (True, 0, 0, 0, 0, 0, 0, 0)],
                            columns=collist)
        dflist += [fakedf]
        df = pd.concat(dflist, ignore_index=True)

    return df, df2