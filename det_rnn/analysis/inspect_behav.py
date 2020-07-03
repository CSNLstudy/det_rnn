import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

__all__ = ['softmax_pred_output','behavior_summary', 'behavior_figure']

def softmax_pred_output(pred_output):
    # softmax the pred_output
    cenoutput = tf.nn.softmax(pred_output, axis=2)
    cenoutput = cenoutput.numpy()
    return cenoutput

def behavior_summary(trial_info, pred_output, par):
    cenoutput = softmax_pred_output(pred_output)
    
    # posterior mean as a function of time
    post_prob = cenoutput[:,:,par['n_rule_output']:]
    post_prob = post_prob/(np.sum(post_prob, axis=2, keepdims=True)+np.finfo(np.float32).eps) # Dirichlet normaliation
    post_support = np.linspace(0,np.pi,par['n_ori'],endpoint=False) + np.pi/par['n_ori']/2
    post_sinr = np.sin(2*post_support)
    post_cosr = np.cos(2*post_support)
    pseudo_mean = np.arctan2(post_prob @ post_sinr, post_prob @ post_cosr)/2
    
    # posterior mean collapsed along time
    estim_sinr = (np.sin(2*pseudo_mean[par['design_rg']['estim'],:])).mean(axis=0)
    estim_cosr = (np.cos(2*pseudo_mean[par['design_rg']['estim'],:])).mean(axis=0)
    estim_mean = np.arctan2(estim_sinr, estim_cosr)/2
    
    ## Quantities for plotting
    ground_truth  = trial_info['stimulus_ori']
    ground_truth  = ground_truth * np.pi/par['n_ori']
    raw_error = estim_mean - ground_truth
    beh_perf  = np.cos(2.*(ground_truth - estim_mean))

    return ground_truth, estim_mean, raw_error, beh_perf

def behavior_figure(ground_truth, estim_mean, raw_error, beh_perf):
    cos_supp  = np.linspace(0,np.pi,1000)
    plt.clf()
    fig, ax = plt.subplots(3,2, figsize=[15,10])
    plt.subplots_adjust(hspace=0.4)
    
    ax[0,0].hist(beh_perf, bins=30); ax[0,0].set_title("Performance Distribution")
    ax[0,0].axvline(x=0, color='r', linestyle='--', label="Chance Level")
    ax[0,0].set_xlabel("Performance"); ax[0,0].set_ylabel("Count"); ax[0,0].legend()
    
    sns.scatterplot(x='GT', y='Perf', data=pd.DataFrame({'GT':ground_truth, 'Perf':beh_perf}), ax=ax[0,1])
    ax[0,1].set_title("Performance as a function of ground truth")
    ax[0,1].axhline(y=0, color='r', linestyle='--', label="Chance Level")
    ax[0,1].set_xlabel(r"$\theta$(rad)"); ax[0,1].set_ylabel("Performance"); ax[0,1].legend()
    
    ax[1,0].plot(cos_supp,np.cos(cos_supp*2), linewidth=2, color='darkgreen', label=r"$\cos(2\theta)$")
    ax[1,0].set_title("Estimation(Cosine Overlay)")
    sns.scatterplot(x='GT', y='CosBehav', data=pd.DataFrame({'GT':ground_truth, 'CosBehav':np.cos(estim_mean*2)}), ax=ax[1,0])
    ax[1,0].set_xlabel(r"$\theta$(rad)"); ax[1,0].set_ylabel(r"$\cos(2\hat{\theta}$)"); ax[1,0].legend()
    
    ax[1,1].set_title("Estimation(Cosine Transformed)")
    sns.regplot(x='GT', y='CosBehav', data=pd.DataFrame({'GT':np.cos(ground_truth*2), 'CosBehav':np.cos(estim_mean*2)}), ax=ax[1,1])
    ax[1,1].set_xlabel(r"$\cos(2\theta$)"); ax[1,1].set_ylabel(r"$\cos(2\hat{\theta}$)"); ax[1,1].legend()
    
    ax[2,0].set_title("Error Distribution")
    sns.scatterplot(x='GT', y='Error', data=pd.DataFrame({'GT':ground_truth, 'Error':np.arcsin(np.sin(2*raw_error))}), ax=ax[2,0])
    ax[2,0].set_xlabel(r"$\theta$(rad)"); ax[2,0].set_ylabel(r"$\hat{\theta} - \theta$"); plt.legend()
    
    ax[2,1].set_title("Estimation Distribution")
    ax[2,1].hist(estim_mean%(np.pi), bins=30)
    ax[2,1].set_xlabel(r"$\hat{\theta}$(rad)"); ax[2,1].set_ylabel("Count"); ax[2,1].legend()
    
    plt.show()
