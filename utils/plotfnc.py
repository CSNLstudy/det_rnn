import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

## todo: put this function here or in the stimulus?

def plot_rnn_output(par,trial_info,pred_output,stim, TEST_TRIAL=None):
    if TEST_TRIAL is None:
        TEST_TRIAL = np.random.randint(stim.batch_size)

    fig, axes = plt.subplots(4, 1, figsize=(8, 12))
    if tf.is_tensor(trial_info['neural_input']):
        a0 = trial_info['neural_input'].cpu().numpy()
    else:
        a0 = trial_info['neural_input']
    if tf.is_tensor(trial_info['desired_output']):
        a1 = trial_info['desired_output'].cpu().numpy()
    else:
        a1 = trial_info['desired_output']
    if tf.is_tensor(pred_output):
        a2 = pred_output.cpu().numpy()
    else:
        a2 = pred_output

    axes[0].imshow(a0[:, TEST_TRIAL, :].T,interpolation='none',aspect='auto');
    axes[0].set_title("Neural Input")
    if par['resp_decoding'] == 'conti':
        axes[1].plot(a1[:, TEST_TRIAL, :].T);
        axes[1].set_ylim([-np.pi, np.pi]);
        axes[1].set_title("Desired Output")
        axes[2].plot(a2[:, TEST_TRIAL, :]);
        axes[2].set_ylim([-np.pi, np.pi]);
        axes[2].set_title("Predicted output")
    elif par['resp_decoding'] == 'disc':
        axes[1].imshow(a1[:, TEST_TRIAL, :].T, interpolation='none', aspect='auto');
        axes[1].set_title("Desired Output")

        axes[2].imshow(a2[:, TEST_TRIAL, :stim.n_rule_output].T,
                       interpolation='none',
                       aspect='auto');
        axes[2].set_title("Predicted output rules")
        axes[3].imshow(a2[:, TEST_TRIAL, stim.n_rule_output:].T,
                       interpolation='none',
                       aspect='auto');
        axes[3].set_title("Predicted output")
    plt.show()

def plot_trial(stim, trial_info, TEST_TRIAL=None):
    if TEST_TRIAL is None:
        TEST_TRIAL = np.random.randint(stim.batch_size)

    axes = {}
    fig = plt.figure(constrained_layout=True, figsize=(10, 8))
    gs = fig.add_gridspec(6, 2)
    axes[0] = fig.add_subplot(gs[0, 0])
    im0 = axes[0].imshow(trial_info['neural_input'][:, TEST_TRIAL, :stim.n_rule_input].T,
                         interpolation='none',
                         aspect='auto');
    axes[0].set_title("Input Rule")
    axes[0].set_xlabel("Time (frames)")
    fig.colorbar(im0, ax=axes[0])

    axes[4] = fig.add_subplot(gs[1, 0])
    im3 = axes[4].imshow(trial_info['mask'][:, TEST_TRIAL, :stim.n_rule_output].T,
                         interpolation='none',
                         aspect='auto');
    axes[4].set_title("Training Mask_rules")
    axes[4].set_xlabel("Time (frames)")
    fig.colorbar(im3, ax=axes[4])

    axes[5] = fig.add_subplot(gs[1, 1])
    im4 = axes[5].imshow(trial_info['mask'][:, TEST_TRIAL, stim.n_rule_output:].T,
                         interpolation='none',
                         aspect='auto');
    axes[5].set_title("Training Mask");
    axes[5].set_xlabel("Time (frames)")
    axes[5].set_ylabel("Neuron")
    fig.colorbar(im4, ax=axes[5])

    axes[2] = fig.add_subplot(gs[0, 1])
    im2 = axes[2].imshow(trial_info['desired_output'][:, TEST_TRIAL, :stim.n_rule_output].T,
                         interpolation='none',
                         aspect='auto');
    axes[2].set_title("Desired Output Rules")
    axes[2].set_xlabel("Time (frames)")
    fig.colorbar(im2, ax=axes[2])


    axes[1] = fig.add_subplot(gs[2:4, :])
    im1 = axes[1].imshow(trial_info['neural_input'][:, TEST_TRIAL, stim.n_rule_input:].T,
                         extent=[0, trial_info['neural_input'].shape[0], stim.pref_dirs[0], stim.pref_dirs[-1]],
                         origin='lower',
                         interpolation='none',
                         aspect='auto');
    axes[1].set_title("Neural Input")
    axes[1].set_xlabel("Time (frames)")
    axes[1].set_ylabel("Neuron (pref. ori. deg)")
    fig.colorbar(im1, ax=axes[1])

    axes[3] = fig.add_subplot(gs[4:6, :])
    im2 = axes[3].imshow(trial_info['desired_output'][:, TEST_TRIAL, stim.n_rule_output:].T,
                         #extent=[0, trial_info['neural_input'].shape[0], stim.pref_dirs[0], stim.pref_dirs[-1]],
                         origin='lower',
                         interpolation='none',
                         aspect='auto');
    axes[3].set_title("Desired Output")
    axes[3].set_xlabel("Time (frames)")
    axes[3].set_ylabel("Neuron (#)")
    fig.colorbar(im2, ax=axes[3])

    #fig.tight_layout(pad=2.0)
    plt.show()

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
