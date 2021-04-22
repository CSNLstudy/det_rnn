import numpy as np
import tensorflow as tf

import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as mticker

from models.base.analysis import behavior_summary


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

EPSILON = 1e-7

def plot_rnn_output(trial_info,test_outputs,parOrStim,
                    TEST_TRIAL=None,
                    savename=None):
    if TEST_TRIAL is None:
        TEST_TRIAL = np.random.randint(parOrStim.batch_size)

    # trial_info,test_outputs,parOrStim, TEST_TRIAL

    if type(parOrStim) is not dict:  # inherit from stimulus class
        par = {}
        par['n_ori'] = parOrStim.n_ori
        par['n_rule_input'] = parOrStim.n_rule_input
        par['n_tuned_output'] = parOrStim.n_tuned_output
        par['n_rule_output_em'] = parOrStim.n_rule_output_em
        par['n_rule_output_dm'] = parOrStim.n_rule_output_dm
        par['design_rg'] = parOrStim.design_rg
        par['input_rule_rg'] = parOrStim.input_rule_rg
    else:  # it the par struct
        par = parOrStim

    axes = {}
    fig = plt.figure(constrained_layout=True, figsize=(10, 20))
    gs = fig.add_gridspec(21, 1)

    # neural input (3)
    axes[0] = fig.add_subplot(gs[0:3, :])

    neuralinput = trial_info['neural_input'][:, TEST_TRIAL, par['n_rule_input']:]
    if not isinstance(neuralinput,np.ndarray):
        neuralinput = neuralinput.numpy()

    im0 = axes[0].imshow(neuralinput.T,
                         origin='lower',
                         interpolation='none',
                         aspect='auto');
    axes[0].set_title("Neural Input")
    axes[0].set_xlabel("Time (frames)")
    axes[0].set_ylabel("Neuron (pref. ori. deg)")
    fig.colorbar(im0, ax=axes[0])

    # hidden neural activity (10)
    rnnoutput = test_outputs['rnn_output'][:, TEST_TRIAL, :]
    if not isinstance(rnnoutput,np.ndarray):
        rnnoutput = rnnoutput.numpy()

    axes[1] = fig.add_subplot(gs[3:13, :])
    im1 = axes[1].imshow(rnnoutput.T,
                         origin='lower',
                         interpolation='none',
                         aspect='auto');
    axes[1].set_title("Hidden neurons activity")
    axes[1].set_xlabel("Time (frames)")
    axes[1].set_ylabel("Neuron")
    fig.colorbar(im1, ax=axes[1])

    # decision desired output (1)
    decdesout = trial_info['desired_decision'][:, TEST_TRIAL, par['n_rule_output_dm']:]
    if not isinstance(decdesout,np.ndarray):
        decdesout = decdesout.numpy()
    axes[2] = fig.add_subplot(gs[13, :])
    im2 = axes[2].imshow(decdesout.T,
                         origin='lower',
                         interpolation='none',
                         aspect='auto');
    axes[2].set_title("Decision: Desired Output");
    axes[2].set_xlabel("Time (frames)")
    axes[2].set_ylabel("Neuron")
    fig.colorbar(im2, ax=axes[2])

    # decision activity (1)
    axes[3] = fig.add_subplot(gs[14, :])
    im3 = axes[3].imshow(test_outputs['dec_output'][:, TEST_TRIAL, :].numpy().T,
                         origin='lower',
                         interpolation='none',
                         aspect='auto');
    axes[3].set_title("Decision neurons activity");
    axes[3].set_xlabel("Time (frames)")
    axes[3].set_ylabel("Neuron")
    fig.colorbar(im3, ax=axes[3])

    # estimation desired output (3)
    decestout = trial_info['desired_estim'][:, TEST_TRIAL, par['n_rule_output_em']:]
    if not isinstance(decestout,np.ndarray):
        decestout = decestout.numpy()
    axes[4] = fig.add_subplot(gs[15:18, :])
    im4 = axes[4].imshow(decestout.T,
                         origin='lower',
                         interpolation='none',
                         aspect='auto')
    axes[4].set_title("Estimation: Desired Output")
    axes[4].set_xlabel("Time (frames)")
    axes[4].set_ylabel("Neuron (#)")
    fig.colorbar(im4, ax=axes[4])

    # estimation activity (3)
    axes[5] = fig.add_subplot(gs[18:21, :])
    im5 = axes[5].imshow(test_outputs['est_output'][:, TEST_TRIAL, :].numpy().T,
                         origin='lower',
                         interpolation='none',
                         aspect='auto');
    axes[5].set_title("Estimation neurons activity");
    axes[5].set_xlabel("Time (frames)")
    axes[5].set_ylabel("Neuron")
    fig.colorbar(im5, ax=axes[5])

    if savename is None:
        plt.show()
    else:
        plt.savefig(savename)
    plt.close()

def plot_trial(trial_info, stim, TEST_TRIAL=None, savename=None):
    if TEST_TRIAL is None:
        TEST_TRIAL = np.random.randint(stim.batch_size)

    axes = {}
    fig = plt.figure(constrained_layout=True, figsize=(10, 12))
    gs = fig.add_gridspec(10, 2)

    # input rule
    axes[0] = fig.add_subplot(gs[0, :])
    im0 = axes[0].imshow(trial_info['neural_input'][:, TEST_TRIAL, :stim.n_rule_input].T,
                         interpolation='none',
                         aspect='auto')
    axes[0].set_title("Input Rule")
    axes[0].set_xlabel("Time (frames)")
    fig.colorbar(im0, ax=axes[0])

    # mask rules
    axes[2] = fig.add_subplot(gs[2, 0])
    im2 = axes[2].imshow(trial_info['mask_decision'][:, TEST_TRIAL, :stim.n_rule_output_dm].T,
                         interpolation='none',
                         aspect='auto')
    axes[2].set_title("Decision: Training Mask rules")
    axes[2].set_xlabel("Time (frames)")
    fig.colorbar(im2, ax=axes[2])

    axes[3] = fig.add_subplot(gs[2, 1])
    im3 = axes[3].imshow(trial_info['mask_estim'][:, TEST_TRIAL, :stim.n_rule_output_em].T,
                         interpolation='none',
                         aspect='auto')
    axes[3].set_title("Estimation: Training Mask rules")
    axes[3].set_xlabel("Time (frames)")
    fig.colorbar(im3, ax=axes[3])

    axes[4] = fig.add_subplot(gs[3, 0])
    im4 = axes[4].imshow(trial_info['mask_decision'][:, TEST_TRIAL, stim.n_rule_output_dm:].T,
                         interpolation='none',
                         aspect='auto')
    axes[4].set_title("Decision: Training Mask")
    axes[4].set_xlabel("Time (frames)")
    fig.colorbar(im4, ax=axes[4])

    axes[5] = fig.add_subplot(gs[3, 1])
    im5 = axes[5].imshow(trial_info['mask_estim'][:, TEST_TRIAL, stim.n_rule_output_em:].T,
                         interpolation='none',
                         aspect='auto')
    axes[5].set_title("Estimation: Training Mask")
    axes[5].set_xlabel("Time (frames)")
    fig.colorbar(im5, ax=axes[5])

    # desired output rules, decision
    axes[6] = fig.add_subplot(gs[1, 0])
    im6 = axes[6].imshow(trial_info['desired_decision'][:, TEST_TRIAL, :stim.n_rule_output_dm].T,
                         interpolation='none',
                         aspect='auto');
    axes[6].set_title("Decision: Desired Output Rules");
    axes[6].set_xlabel("Time (frames)")
    axes[6].set_ylabel("Neuron")
    fig.colorbar(im6, ax=axes[6])

    # desired output rules, estimation
    axes[7] = fig.add_subplot(gs[1, 1])
    im7 = axes[7].imshow(trial_info['desired_estim'][:, TEST_TRIAL, :stim.n_rule_output_em].T,
                         interpolation='none',
                         aspect='auto');
    axes[7].set_title("Estimation: Desired Output Rules")
    axes[7].set_xlabel("Time (frames)")
    fig.colorbar(im7, ax=axes[7])

    # Neural input
    axes[8] = fig.add_subplot(gs[4:7, :])
    im8 = axes[8].imshow(trial_info['neural_input'][:, TEST_TRIAL, stim.n_rule_input:].T,
                         extent=[0, trial_info['neural_input'].shape[0], stim.pref_dirs[0], stim.pref_dirs[-1]],
                         origin='lower',
                         interpolation='none',
                         aspect='auto');
    axes[8].set_title("Neural Input")
    axes[8].set_xlabel("Time (frames)")
    axes[8].set_ylabel("Neuron (pref. ori. deg)")
    fig.colorbar(im8, ax=axes[8])

    # desired output, decision
    axes[9] = fig.add_subplot(gs[7, :])
    im9 = axes[9].imshow(trial_info['desired_decision'][:, TEST_TRIAL, stim.n_rule_output_dm:].T,
                         #extent=[0, trial_info['neural_input'].shape[0], stim.pref_dirs[0], stim.pref_dirs[-1]],
                         origin='lower',
                         interpolation='none',
                         aspect='auto');
    axes[9].set_title("Decision: Desired Output")
    axes[9].set_xlabel("Time (frames)")
    axes[9].set_ylabel("Neuron (#)")
    fig.colorbar(im9, ax=axes[9])

    # desired output, estimation
    axes[10] = fig.add_subplot(gs[8:10, :])
    im10 = axes[10].imshow(trial_info['desired_estim'][:, TEST_TRIAL, stim.n_rule_output_em:].T,
                         #extent=[0, trial_info['neural_input'].shape[0], stim.pref_dirs[0], stim.pref_dirs[-1]],
                         origin='lower',
                         interpolation='none',
                         aspect='auto')
    axes[10].set_title("Estimation: Desired Output")
    axes[10].set_xlabel("Time (frames)")
    axes[10].set_ylabel("Neuron (#)")
    fig.colorbar(im10, ax=axes[10])

    #fig.tight_layout(pad=2.0)
    if savename is None:
        plt.show()
    else:
        plt.savefig(savename)

def make_var_dict_figure(model):
    fig = plt.figure(figsize=(9.5, 12))

    ax1 = plt.subplot2grid((8,6), (0,1), rowspan=2,colspan=4); ax1.set_title("w_out")
    ax2 = plt.subplot2grid((8,6), (0,5), rowspan=2,colspan=1); ax2.set_title("b_out")
    ax3 = plt.subplot2grid((8,6), (2,0), rowspan=4,colspan=1); ax3.set_title("h")
    ax4 = plt.subplot2grid((8,6), (2,1), rowspan=4,colspan=4); ax4.set_title("w_rnn")
    ax5 = plt.subplot2grid((8,6), (2,5), rowspan=4,colspan=1); ax5.set_title("b_rnn")
    ax6 = plt.subplot2grid((8,6), (6,1), rowspan=2,colspan=4); ax6.set_title("w_in")

    im1 = ax1.imshow(model.var_dict['w_out'].numpy().T,
                     interpolation='none',
                     aspect='auto')
    im2 = ax2.imshow(model.var_dict['b_out'].numpy().reshape((-1,1)),
                     interpolation='none',
                     aspect='auto')
    im3 = ax3.imshow(model.var_dict['h'].numpy().T,
                     interpolation='none',
                     aspect='auto')
    im4 = ax4.imshow(model.var_dict['w_rnn'].numpy(),
                     interpolation='none',
                     aspect='auto')
    im5 = ax5.imshow(model.var_dict['b_rnn'].numpy().reshape((-1,1)),
                     interpolation='none',
                     aspect='auto')
    im6 = ax6.imshow(model.var_dict['w_in'].numpy(),
                     interpolation='none',
                     aspect='auto')

    cbarax1 = fig.add_axes([1.10, 6/8, 0.01, 2/8])
    cbarax2 = fig.add_axes([1.15, 6/8, 0.01, 2/8])
    cbarax3 = fig.add_axes([1.10, 3/8, 0.01, 2/8])
    cbarax4 = fig.add_axes([1.15, 3/8, 0.01, 2/8])
    cbarax5 = fig.add_axes([1.20, 3/8, 0.01, 2/8])
    cbarax6 = fig.add_axes([1.15, 0, 0.01, 2/8])

    fig.colorbar(im1, cax=cbarax1)
    fig.colorbar(im2, cax=cbarax2)
    fig.colorbar(im3, cax=cbarax3)
    fig.colorbar(im4, cax=cbarax4)
    fig.colorbar(im5, cax=cbarax5)
    fig.colorbar(im6, cax=cbarax6)

    #plt.tight_layout()
    plt.show()

def behavior_figure(est_summary, filename=None):
    """
    ground_truth : 0 to pi
    estim_mean: -pi/2 to pi/2
    raw_error: 0 to pi
    beh_perf: -1 to 1
    """
    ground_truth    = est_summary['est_target']
    estim_mean      = est_summary['est_mean']
    raw_error       = est_summary['est_error']
    beh_perf        = est_summary['est_perf']

    cos_supp = np.linspace(0, np.pi, 1000)
    fig, ax = plt.subplots(3, 2, figsize=[15, 10])
    plt.subplots_adjust(hspace=0.4)

    ax[0, 0].hist(beh_perf, bins=30);
    ax[0, 0].set_title("Performance Distribution")
    ax[0, 0].axvline(x=0, color='r', linestyle='--', label="Chance Level");
    ax[0, 0].set_xlabel("Performance");
    ax[0, 0].set_ylabel("Count");
    ax[0, 0].set_xlim([-1, 1]);
    ax[0, 0].legend();

    sns.scatterplot(x='GT', y='Perf', data=pd.DataFrame({'GT': ground_truth, 'Perf': beh_perf}), ax=ax[0, 1])
    ax[0, 1].set_title("Performance as a function of ground truth")
    ax[0, 1].axhline(y=0, color='r', linestyle='--', label="Chance Level")
    ax[0, 1].set_xlabel(r"$\theta$(rad)");
    ax[0, 1].set_ylabel("Performance");
    ax[0, 1].legend()
    ax[0, 1].set_xlim([0, np.pi]);

    ax[1, 0].plot(cos_supp, np.cos(cos_supp * 2), linewidth=2, color='darkgreen', label=r"$\cos(2\theta)$")
    ax[1, 0].set_title("Estimation(Cosine Overlay)")
    sns.scatterplot(x='GT', y='CosBehav', data=pd.DataFrame({'GT': ground_truth, 'CosBehav': np.cos(estim_mean * 2)}),
                    ax=ax[1, 0])
    ax[1, 0].set_xlabel(r"$\theta$(rad)");
    ax[1, 0].set_ylabel(r"$\cos(2\hat{\theta}$)");
    ax[1, 0].legend()

    ax[1, 1].set_title("Estimation(Cosine Transformed)")
    sns.regplot(x='GT', y='CosBehav',
                data=pd.DataFrame({'GT': np.cos(ground_truth * 2), 'CosBehav': np.cos(estim_mean * 2)}),
                ax=ax[1, 1], label="reg.")
    ax[1, 1].set_xlabel(r"$\cos(2\theta$)");
    ax[1, 1].set_ylabel(r"$\cos(2\hat{\theta}$)");
    ax[1, 1].legend()
    ax[1, 1].set_xlim([-1, 1]);
    ax[1, 1].set_ylim([-1, 1]);

    ax[2, 0].set_title("Error Distribution")
    sns.scatterplot(x='GT', y='Error', data=pd.DataFrame({'GT': ground_truth,
                                                          'Error': np.arcsin(np.sin(2 * raw_error))}),
                    ax=ax[2, 0])  # todo (josh): doesn't this cut off the big errors i.e. raw_error = pi?
    ax[2, 0].set_xlabel(r"$\theta$(rad)");
    ax[2, 0].set_ylabel(r"$\hat{\theta} - \theta$");
    # plt.legend()

    ax[2, 1].set_title("Estimation Distribution")
    ax[2, 1].hist(estim_mean % (np.pi), bins=np.arange(0, np.pi + 0.01, np.pi / 48))
    ax[2, 1].set_xlabel(r"$\hat{\theta}$(rad)");
    ax[2, 1].set_xlim([0, np.pi]);
    ax[2, 1].set_ylabel("Count");
    # ax[2,1].legend()

    # save stuff or show and close
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()

def biasvar_figure(est_summary, stim_test, filename=None):
    """
    ground_truth : 0 to pi
    estim_mean: -pi/2 to pi/2
    raw_error: 0 to pi
    """
    ground_truth    = est_summary['est_target']
    estim_mean      = est_summary['est_mean']
    raw_error       = est_summary['est_error']
    beh_perf        = est_summary['est_perf']


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
            circ_mean = np.arctan2(sinesum, cossum)
            circ_std = np.sqrt(-2 * np.log(meanres - EPSILON))
        errorMean.append(circ_mean)
        errorSTD.append(circ_std)
        binN.append(n)
    errors = np.arctan2(np.sin(2 * raw_error), np.cos(2 * raw_error)) / 2
    # (-180,180), period pi => (-360,360),eriod 2pi=> apply atan2, (-pi,pi), period 2pi=> (-pi/2,pi/2), period pi.
    errorMean = np.array(errorMean) / 2
    errorSTD = np.array(errorSTD) / 2

    fig, ax = plt.subplots(2, 2, figsize=[15, 10])
    plt.subplots_adjust(hspace=0.4)

    ax[0, 0].set_title("Bias (axes unscaled)")
    sns.scatterplot(x='GT', y='Bias',
                    data=pd.DataFrame({'GT': ground_truth * 180 / np.pi, 'Bias': errors * 180 / np.pi}),
                    ax=ax[0, 0],
                    palette='orange')
    sns.scatterplot(x='GT', y='Average Bias',
                    data=pd.DataFrame({'GT': bin_center * 180 / np.pi, 'Average Bias': errorMean * 180 / np.pi}),
                    ax=ax[0, 0], palette='blue')
    ax[0, 0].set_xlabel(r"$\theta$(deg)");
    ax[0, 0].set_ylabel(r"$\hat{\theta} - \theta$ (deg)");
    ax[0, 0].axhline(y=0, color='g', linestyle='--', label="Unbiased")
    ax[0, 0].set_xlim([0, 180]);
    ax[0, 0].set_ylim([-90, 90]);
    ax[0, 0].legend();
    # plt.legend()

    ax[0, 1].set_title("Variance (axes unscaled)")
    sns.scatterplot(x='GT', y='Std',
                    data=pd.DataFrame({'GT': bin_center * 180 / np.pi, 'Std': (errorSTD * 180 / np.pi)}),
                    ax=ax[0, 1])
    ax[0, 1].set_xlabel(r"$\theta$(deg)");
    ax[0, 1].set_ylabel('Variabiliy (deg)');
    # ax[1].set_yscale('log'); ax[1].set_ylim([0, 50]); ax[1].set_yticks([10, 30, 50]);
    # ax[1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[0, 1].set_xlim([0, 180]);
    ax[0, 1].set_ylim([0, 50]);

    ax[1, 0].set_title("Average Bias")
    sns.scatterplot(x='GT', y='Average Bias',
                    data=pd.DataFrame({'GT': bin_center * 180 / np.pi, 'Average Bias': errorMean * 180 / np.pi}),
                    ax=ax[1, 0])
    ax[1, 0].set_xlabel(r"$\theta$(deg)");
    ax[1, 0].set_ylabel(r"$\hat{\theta} - \theta$ (deg)");
    ax[1, 0].axhline(y=0, color='g', linestyle='--', label="Unbiased")
    ax[1, 0].axvline(x=90, color='g', linestyle='--', label="Cardinal")
    ax[1, 0].set_xlim([0, 180]);
    ax[1, 0].set_ylim([-12, 12]);
    ax[1, 0].legend();
    # plt.legend()

    ax[1, 1].set_title("Variance")
    sns.scatterplot(x='GT', y='Std',
                    data=pd.DataFrame({'GT': bin_center * 180 / np.pi, 'Std': (errorSTD * 180 / np.pi)}),
                    ax=ax[1, 1])
    ax[1, 1].set_xlabel(r"$\theta$(deg)");
    ax[1, 1].set_ylabel('Variabiliy (deg)');
    ax[1, 1].axvline(x=90, color='g', linestyle='--', label="Cardinal")
    ax[1, 1].set_yscale('log');
    ax[1, 1].set_ylim([1, 50]);
    ax[1, 1].set_yticks([10, 30, 50]);
    ax[1, 1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[1, 1].set_xlim([0, 180]);

    # save stuff or show and close
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()

    return errorMean, errorSTD , binN

def plot_decision_effects(df,df2, filename = None):
    # plot distributions wrt reference and decision
    a4_dims = (11.7, 8.27)
    fig, ax = plt.subplots(figsize=a4_dims)
    g1 = sns.violinplot(x="Ref", y="est_error", hue="CW",
                        data=df, palette="hls", split=True,
                        scale="count", inner="stick",
                        scale_hue=False, bw=.2)
    # g.legend(['CCW','CW'])
    xlabels = ['{:,.2f}'.format(float(x.get_text())) for x in g1.get_xticklabels()];
    g1.set_xticklabels(xlabels)
    g1.set_title("Estimation mean")
    g1.set_xlabel("Reference (deg)")
    g1.set_ylabel("Estimation error (cos)")

    # save stuff or show and close
    if filename is not None:
        plt.savefig(filename+'dist')
    else:
        plt.show()
    plt.close()


    axes = {}
    fig = plt.figure(constrained_layout=True, figsize=(12, 5))
    gs = fig.add_gridspec(1, 2)
    axes[0] = fig.add_subplot(gs[0, 0])
    g = sns.scatterplot(x='Ref', y='mean', hue="CW", palette="hls", marker='o', s=200, data=df2)
    ticks_loc = g.get_xticks().tolist()
    g.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    xlabels = ['{:,.2f}'.format(x) for x in g.get_xticks()]
    g.set_xticklabels(xlabels)
    g.set_title("Estimation error mean")
    g.set_xlabel("Reference (deg)")
    g.set_ylabel("mean error (deg)")

    axes[1] = fig.add_subplot(gs[0, 1])
    g = sns.scatterplot(x='Ref', y='sd', hue="CW", palette="hls", marker='o', s=200, data=df2)
    ticks_loc = g.get_xticks().tolist()
    g.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    xlabels = ['{:,.2f}'.format(x) for x in g.get_xticks()]
    g.set_xticklabels(xlabels)
    g.set_title("Estimation error std")
    g.set_xlabel("Reference (deg)")
    g.set_ylabel("std of error (deg)")

    # save stuff or show and close
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()

    return None