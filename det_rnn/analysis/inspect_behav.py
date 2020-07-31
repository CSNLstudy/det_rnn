import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import logging, requests

#logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)
#logging.getLogger('requests').setLevel(logging.DEBUG)
# disable annoying debugs from matpltlib! do this before importing matplotlib
# https://stackoverflow.com/questions/35325042/python-logging-disable-logging-from-imported-modules
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib').disabled = True
logging.getLogger('matplotlib.font_manager').disabled = True
#logging.getLogger('DEBUG').disabled = True

import matplotlib.pyplot as plt
import matplotlib

EPSILON = 1e-7

__all__ = ['softmax_pred_output','behavior_summary', 'behavior_figure', 'biasvar_figure']

def softmax_pred_output(pred_output):
    # softmax the pred_output
    cenoutput = tf.nn.softmax(pred_output, axis=2)
    cenoutput = cenoutput.numpy()
    return cenoutput

def behavior_summary(trial_info, pred_output, parOrStim):
    '''
    parOrStim: par or stim struct for testing.
    '''
    if type(parOrStim) is not dict: # inherit from stimulus class
        par = {}
        par['n_ori']            = parOrStim.n_ori
        par['n_tuned_output']   = parOrStim.n_tuned_output
        par['n_rule_output']    = parOrStim.n_rule_output
        par['design_rg']        = parOrStim.design_rg
    else: # it the par struct
        par = parOrStim

    # find posterior probabilities, post_prob (B x neurons)
    if len(pred_output.shape) == 2: # single tuning (pred_output.shape is B x neurons)
        cenoutput   = tf.nn.softmax(pred_output, axis=1)
        cenoutput   = cenoutput.numpy()
        post_prob   = cenoutput
    else: # time series (pred_output.shape is B x T x neurons)
        cenoutput = softmax_pred_output(pred_output[:, :, par['n_rule_output']:])
        # posterior mean as a function of time
        post_prob = cenoutput[:, :, :]
        post_prob = post_prob / (
                    np.sum(post_prob, axis=2, keepdims=True) + np.finfo(np.float32).eps)  # Dirichlet normaliation todo (josh): remove, or add eps before softmax?

    # post_support = np.linspace(0, np.pi, par['n_tuned_output'], endpoint=False) + np.pi / par['n_ori'] / 2 # hmm??why add the np.pi?
    # Convert domain from (0,pi) => (0, 2pi) => circular mean of the posterior (-pi, pi)
    post_support    = np.arange(0, np.pi, np.pi / par['n_tuned_output']) # 0 to pi
    post_sinr       = np.sin(2 * post_support)
    post_cosr       = np.cos(2 * post_support)

    # take the (circular) posterior mean; then convert to 0 to pi.
    post_mean = np.arctan2(post_prob @ post_sinr, post_prob @ post_cosr)/2 # -pi/2 to pi/2

    if len(pred_output.shape) == 2: # single tuning
        estim_mean = post_mean # 0 to pi
    else:  # time series
        # todo: check this
        # posterior mean collapsed along time
        estim_sinr = (np.sin(2 * post_mean[par['design_rg']['estim'], :])).mean(axis=0)
        estim_cosr = (np.cos(2 * post_mean[par['design_rg']['estim'], :])).mean(axis=0)
        estim_mean = np.arctan2(estim_sinr, estim_cosr)/2     # -pi to pi, -pi/2 to pi

    ## Quantities for plotting
    ground_truth = trial_info['stimulus_ori'] # 0 to (n_tuned_output-1)
    if tf.is_tensor(ground_truth):
        ground_truth = tf.make_ndarray(tf.make_tensor_proto(ground_truth))

    ground_truth        = ground_truth * np.pi/par['n_ori'] # range: 0 to pi
    raw_error           = estim_mean - ground_truth # 0 to pi
    beh_perf            = np.cos(2*raw_error) # -1 to 1

    # check ranges
    if not (np.max(ground_truth) <= np.pi and np.min(ground_truth) >= 0):
        print('wtf something is wrong throwing assertion error')
        assert np.max(ground_truth) <= np.pi and np.min(ground_truth) >= 0
    if not (np.max(estim_mean) <= np.pi/2 and np.min(estim_mean) >= -np.pi/2):
        print('wtf something is wrong, throwing assertion error')
        assert np.max(estim_mean) <= np.pi/2 and np.min(estim_mean) >= -np.pi/2

    return ground_truth, estim_mean, raw_error, beh_perf

def behavior_figure(ground_truth, estim_mean, raw_error, beh_perf, filename = None):
    """
    ground_truth : 0 to pi
    estim_mean: -pi/2 to pi/2
    raw_error: 0 to pi
    beh_perf: -1 to 1
    """
    cos_supp  = np.linspace(0,np.pi,1000)
    fig, ax = plt.subplots(3,2, figsize=[15,10])
    plt.subplots_adjust(hspace=0.4)
    
    ax[0,0].hist(beh_perf, bins=30); ax[0,0].set_title("Performance Distribution")
    ax[0,0].axvline(x=0, color='r', linestyle='--', label="Chance Level");
    ax[0,0].set_xlabel("Performance"); ax[0,0].set_ylabel("Count");
    ax[0, 0].set_xlim([-1,1]); ax[0,0].legend();

    sns.scatterplot(x='GT', y='Perf', data=pd.DataFrame({'GT':ground_truth, 'Perf':beh_perf}), ax=ax[0,1])
    ax[0,1].set_title("Performance as a function of ground truth")
    ax[0,1].axhline(y=0, color='r', linestyle='--', label="Chance Level")
    ax[0,1].set_xlabel(r"$\theta$(rad)"); ax[0,1].set_ylabel("Performance"); ax[0,1].legend()
    ax[0, 1].set_xlim([0, np.pi]);

    ax[1,0].plot(cos_supp,np.cos(cos_supp*2), linewidth=2, color='darkgreen', label=r"$\cos(2\theta)$")
    ax[1,0].set_title("Estimation(Cosine Overlay)")
    sns.scatterplot(x='GT', y='CosBehav', data=pd.DataFrame({'GT':ground_truth, 'CosBehav':np.cos(estim_mean*2)}), ax=ax[1,0])
    ax[1,0].set_xlabel(r"$\theta$(rad)"); ax[1,0].set_ylabel(r"$\cos(2\hat{\theta}$)"); ax[1,0].legend()
    
    ax[1,1].set_title("Estimation(Cosine Transformed)")
    sns.regplot(x='GT', y='CosBehav',
                data=pd.DataFrame({'GT':np.cos(ground_truth*2), 'CosBehav':np.cos(estim_mean*2)}),
                ax=ax[1,1], label= "reg.")
    ax[1,1].set_xlabel(r"$\cos(2\theta$)"); ax[1,1].set_ylabel(r"$\cos(2\hat{\theta}$)"); ax[1,1].legend()
    ax[1, 1].set_xlim([-1, 1]);ax[1, 1].set_ylim([-1, 1]);
    
    ax[2,0].set_title("Error Distribution")
    sns.scatterplot(x='GT', y='Error', data=pd.DataFrame({'GT':ground_truth,
                                                          'Error':np.arcsin(np.sin(2*raw_error))}), ax=ax[2,0]) # todo (josh): doesn't this cut off the big errors i.e. raw_error = pi?
    ax[2,0].set_xlabel(r"$\theta$(rad)"); ax[2,0].set_ylabel(r"$\hat{\theta} - \theta$");
    #plt.legend()
    
    ax[2,1].set_title("Estimation Distribution")
    ax[2,1].hist(estim_mean%(np.pi), bins=np.arange(0,np.pi+0.01,np.pi/48))
    ax[2,1].set_xlabel(r"$\hat{\theta}$(rad)");
    ax[2,1].set_xlim([0,np.pi]);
    ax[2,1].set_ylabel("Count");
    #ax[2,1].legend()
    
    # save stuff or show and close
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()

def biasvar_figure(ground_truth, estim_mean, raw_error, stim_test, filename=None):
    """
    ground_truth : 0 to pi
    estim_mean: -pi/2 to pi/2
    raw_error: 0 to pi
    """

    # todo: or just bin based on true orientation....
    bin_center = np.arange(0, np.pi, np.pi / stim_test.n_ori)
    dt = np.pi / stim_test.n_ori / 2
    bin_edges = np.append(-dt, bin_center + dt)
    # bin_edges + (bin_edges[1]-bin_edges[0])/2
    indices = np.digitize(ground_truth, bin_edges, right=False)
    errorMean = []
    errorSTD = []
    binN = []
    for idx in range(1, stim_test.n_ori + 1):  # indices[i] = 0 are the ground_truth[i] < bin_edge[0]
        bin_errors = raw_error[indices == idx]
        n = np.sum(indices == idx)
        if n == 0:
            circ_mean = 0
            circ_std = 0
        else:
            sinesum = np.sum(np.sin(2 * bin_errors))
            cossum = np.sum(np.cos(2 * bin_errors))
            meanres = (np.sqrt(np.square(sinesum) + np.square(cossum))) / n
            circ_mean = np.arctan2(sinesum,cossum)
            circ_std = np.sqrt(-2 * np.log(meanres-EPSILON))
        errorMean.append(circ_mean)
        errorSTD.append(circ_std)
        binN.append(n)
    errors      = np.arctan2(np.sin(2*raw_error),np.cos(2*raw_error))/2
    # (-180,180), period pi => (-360,360),eriod 2pi=> apply atan2, (-pi,pi), period 2pi=> (-pi/2,pi/2), period pi.
    errorMean   = np.array(errorMean)/2
    errorSTD    = np.array(errorSTD)/2

    fig, ax = plt.subplots(2,2, figsize=[15,10])
    plt.subplots_adjust(hspace=0.4)

    ax[0,0].set_title("Bias (axes unscaled)")
    sns.scatterplot(x='GT', y='Bias',
                    data=pd.DataFrame({'GT':ground_truth * 180/np.pi, 'Bias': errors * 180/np.pi}),
                    ax=ax[0,0],
                    palette = 'orange')
    sns.scatterplot(x='GT', y='Average Bias',
                    data=pd.DataFrame({'GT':bin_center * 180/np.pi, 'Average Bias': errorMean * 180/np.pi}),
                    ax=ax[0,0], palette = 'blue')
    ax[0,0].set_xlabel(r"$\theta$(deg)"); ax[0,0].set_ylabel(r"$\hat{\theta} - \theta$ (deg)");
    ax[0,0].axhline(y=0, color='g', linestyle='--', label="Unbiased")
    ax[0,0].set_xlim([0,180]);ax[0,0].set_ylim([-90,90]);
    ax[0,0].legend();
    #plt.legend()

    ax[0,1].set_title("Variance (axes unscaled)")
    sns.scatterplot(x='GT', y='Std',
                    data=pd.DataFrame({'GT':bin_center * 180/np.pi, 'Std':(errorSTD * 180/np.pi)}),
                    ax=ax[0,1])
    ax[0,1].set_xlabel(r"$\theta$(deg)"); ax[0,1].set_ylabel('Variabiliy (deg)');
    #ax[1].set_yscale('log'); ax[1].set_ylim([0, 50]); ax[1].set_yticks([10, 30, 50]);
    #ax[1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[0,1].set_xlim([0, 180]);ax[0,1].set_ylim([0,50]);

    ax[1,0].set_title("Average Bias")
    sns.scatterplot(x='GT', y='Average Bias',
                    data=pd.DataFrame({'GT':bin_center * 180/np.pi, 'Average Bias':errorMean * 180/np.pi}),
                    ax=ax[1,0])
    ax[1,0].set_xlabel(r"$\theta$(deg)"); ax[1,0].set_ylabel(r"$\hat{\theta} - \theta$ (deg)");
    ax[1,0].axhline(y=0, color='g', linestyle='--', label="Unbiased")
    ax[1,0].axvline(x=90, color='g', linestyle='--', label="Cardinal")
    ax[1,0].set_xlim([0,180]);ax[1,0].set_ylim([-12,12]);
    ax[1,0].legend();
    #plt.legend()

    ax[1,1].set_title("Variance")
    sns.scatterplot(x='GT', y='Std',
                    data=pd.DataFrame({'GT':bin_center * 180/np.pi, 'Std':(errorSTD * 180/np.pi)}),
                    ax=ax[1,1])
    ax[1,1].set_xlabel(r"$\theta$(deg)"); ax[1,1].set_ylabel('Variabiliy (deg)');
    ax[1,1].axvline(x=90, color='g', linestyle='--', label="Cardinal")
    ax[1,1].set_yscale('log'); ax[1,1].set_ylim([1, 50]); ax[1,1].set_yticks([10, 30, 50]);
    ax[1,1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[1,1].set_xlim([0, 180]);

    # save stuff or show and close
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()