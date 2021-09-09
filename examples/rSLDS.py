import os, pickle, copy, sys

import autograd.numpy as np
import autograd.numpy.random as npr
npr.seed(12345)

import matplotlib.pyplot as plt
from matplotlib import gridspec

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import ssm
from ssm.util import random_rotation, find_permutation

# import detrnn packages
sys.path.append('../')
from det_rnn import *
import det_rnn.train as utils_train
import det_rnn.analysis as utils_analysis
from det_rnn.train.model import Model

from models.base.analysis import behavior_summary, estimation_decision

from utils.plotfnc import *


# continuous state updates
def build_distractor_resistant_SLDS(N, K, D_obs, D_latent, data, stim_train, gain_d=10, alpha=0.1):
    Usize = data.shape[2]
    T = data.shape[0]
    inpt = [data[:, i, :] for i in range(data.shape[1])]

    nrule = stim_train.n_rule_input
    A = stim_train.tuning_input[:, 0, :]
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    snew = np.zeros((N,))
    snew[0:11] = 1
    smat = np.diag(snew)
    A = np.dot(u, np.dot(smat, vh))

    if False:
        plt.imshow(A)
        plt.title('Neural tuning')
        plt.xlabel('neuron#')
        plt.ylabel('neuron#')
        plt.show()

    Anorm = np.eye(D_latent) * (1 - alpha) + alpha * A

    As = [Anorm, Anorm]
    bs = np.zeros((K, D_latent))
    Vs = [np.column_stack((np.zeros((N, nrule)), alpha * A)), np.zeros((N, Usize))]

    # discrete state updates
    R = np.zeros((D_latent,))
    Rs = [R, R]  # continuous state dependent updates
    # r =
    rs = np.zeros((K,))

    w1 = np.zeros(Usize)  # support for state 0 = accumulation state
    w1[0] = 1
    w1[1:4] = -1
    w2 = np.zeros(Usize)  # support for state 1 = memory state (distractor resistance)
    w2[0] = -1
    w2[1:4] = 1
    Ws = np.row_stack((gain_d * w1, gain_d * w2))

    # transition matrix
    tgain = 0.98
    Trans = np.eye(K) * tgain + (np.ones((K, K)) - np.eye(K)) * (1 - tgain) / (K - 1)

    # construct true rslds num_states, obs_dim, input_dim
    true_rslds = ssm.SLDS(D_obs, K, D_latent, M=Usize,
                          transitions="recurrent",
                          dynamics="diagonal_gaussian",
                          emissions="gaussian_orthog",
                          single_subspace=True)

    true_rslds.dynamics.mu_init = np.zeros((K, D_latent))
    true_rslds.dynamics.sigmasq_init = 1e-4 * np.ones((K, D_latent))
    true_rslds.dynamics.As = np.array(As)
    true_rslds.dynamics.bs = np.array(bs)
    true_rslds.dynamics.Vs = np.array(Vs)
    true_rslds.dynamics.sigmasq = 1e-4 * np.ones((K, D_latent))

    true_rslds.transitions.Ws = np.array(Ws)
    true_rslds.transitions.Rs = np.array(Rs)
    true_rslds.transitions.r = np.array(rs)
    # true_rslds.transitions.transition_matrix = Trans # why doesnt this work?

    true_rslds.emissions.inv_etas = np.log(1e-2) * np.ones((1, D_obs))

    return true_rslds


def sample_and_plot_discretegain(data, stim_train, gain_discrete):
    T = data.shape[0]
    B = data.shape[1]
    inpt = [data[:, i, :] for i in range(data.shape[1])]
    # sample network behavior
    true_rslds = build_distractor_resistant_SLDS(N, K, D_obs, D_latent, data, stim_train, gain_d=gain_discrete)

    zs, xs, ys = [], [], []
    for sess in range(B):
        z, x, y = true_rslds.sample(T=T, input=inpt[sess])
        zs.append(z)
        xs.append(x)
        ys.append(y)

    plt.figure(figsize=(15, 10))
    plt.subplot(3, 1, 1)
    plt.imshow(data[:, -1, nrule:].T, origin='upper', interpolation='none', aspect='auto')
    plt.title("Inputs")
    plt.xlabel("Time (frames)")
    plt.ylabel("Neuron (pref. ori. deg)")
    plt.colorbar()

    plt.subplot(3, 1, 2)
    plt.imshow(x.T, origin='upper', interpolation='none', aspect='auto')
    plt.title("Recurrent neurons (continuous states)")
    plt.xlabel("Time (frames)")
    plt.ylabel("Neuron (pref. ori. deg)")
    plt.colorbar()

    plt.subplot(3, 1, 3)
    plt.imshow(z[None, :], origin='upper', interpolation='none', aspect='auto')
    plt.title("Brain state (discrete states)")
    plt.xlabel("Time (frames)")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    return true_rslds, zs, xs, ys


def sample_and_plot_nonexploding(data, gain_di=2, gain_dx=1 / 20, alpha=0.2):
    T = data.shape[0]
    B = data.shape[1]
    inpt = [data[:, i, :] for i in range(data.shape[1])]
    # sample network behavior
    true_rslds = build_nonexploding_SLDS(N, K, D_obs, D_latent, data, stim_train, gain_di=gain_di, gain_dx=gain_dx,
                                         alpha=alpha)

    zs, xs, ys = [], [], []
    for sess in range(B):
        z, x, y = true_rslds.sample(T=T, input=inpt[sess])
        zs.append(z)
        xs.append(x)
        ys.append(y)

    plt.figure(figsize=(15, 10))
    plt.subplot(4, 1, 1)
    plt.imshow(data[:, -1, nrule:].T, origin='upper', interpolation='none', aspect='auto')
    plt.title("Inputs")
    plt.xlabel("Time (frames)")
    plt.ylabel("Neuron (pref. ori. deg)")
    plt.colorbar()

    plt.subplot(4, 1, 2)
    plt.imshow(x.T, origin='upper', interpolation='none', aspect='auto')
    plt.title("Recurrent neurons (continuous states)")
    plt.xlabel("Time (frames)")
    plt.ylabel("Neuron (pref. ori. deg)")
    plt.colorbar()

    plt.subplot(4, 1, 3)
    plt.imshow(np.floor_divide(a, 3), origin='upper', interpolation='none', aspect='auto')
    plt.title("Accumulation/memory state (discrete states)")
    plt.xlabel("Time (frames)")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    plt.subplot(4, 1, 4)
    plt.imshow(np.mod(z[None, :], 3), origin='upper', interpolation='none', aspect='auto')
    plt.title("Recurrent/global inh/global exc state (discrete states)")
    plt.xlabel("Time (frames)")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    return true_rslds, zs, xs, ys


if __name__ == "__main__":
    # Global parameters
    N = 20  # number of neurons
    # T = 500 # determined by the detrnn module
    K = 2  # discrete states: evidence accumulation and "memory"
    D_obs = 2 * N  # observation: gaussian
    D_latent = N  #

    par_train = copy.deepcopy(par)
    par_train['batch_size'] = 5  # smaller batch size
    par_train['n_ori'] = N
    par_train['n_tuned_input'] = N
    par_train['n_tuned_output'] = N
    par_train['n_hidden'] = N
    par_train['design'] = {'iti': (0, 0.2), 'stim': (0.2, 1.2), 'delay': ((1.2, 1.5), (3.2, 3.3)),
                           'decision': (1.5, 3.2), 'estim': (3.3, 3.5)}
    par_train = update_parameters(par_train)
    stim_train = Stimulus(par_train)

    # generate training data
    train_data = stim_train.generate_trial()

    if False:
        A = stim_train.tuning_input[:, 0, :]
        u, s, vh = np.linalg.svd(A, full_matrices=True)
        snew = np.zeros((N,))
        snew[0:11] = 1
        smat = np.diag(snew)
        A = np.dot(u, np.dot(smat, vh))

        plt.imshow(A)
        plt.title('Neural tuning')
        plt.xlabel('neuron#')
        plt.ylabel('neuron#')
        plt.show()

    if False: # plot trial
        random_trial_num = 0
        # random_trial_num = 2
        target_ori = np.arange(0, 180, 180 / par_train['n_ori'])[train_data['stimulus_ori'][random_trial_num]]
        ref_ori = np.arange(0, 180, 180 / par_train['n_tuned_input'])[train_data['ref_neuron'][random_trial_num]]
        print('orientation = ' + str(target_ori) + ', reference = ' + str(ref_ori))
        plot_trial(train_data, stim_train, TEST_TRIAL=random_trial_num)

    nrule = stim_train.n_rule_input
    data = train_data['neural_input']
    Usize = data.shape[2]
    T = data.shape[0]
    inpt = [data[:, i, :] for i in range(data.shape[1])]

    true_rslds, z, x, y = sample_and_plot_discretegain(data, stim_train, gain_discrete=1)

    # Fit an rSLDS with its default initialization, using Laplace-EM with a structured variational posterior
    rslds = ssm.SLDS(D_obs, K, D_latent, M=Usize,
                     transitions="recurrent",
                     dynamics="diagonal_gaussian",
                     emissions="gaussian_orthog",
                     single_subspace=True)
    rslds.initialize(y)
    q_elbos_lem, q_lem = rslds.fit(y,
                                   inputs=inpt,
                                   method="laplace_em",
                                   variational_posterior="structured_meanfield",
                                   initialize=False, num_iters=25, alpha=0.0)
    xhat_lem = q_lem.mean_continuous_states[0]
    rslds.permute(find_permutation(z[0],
                                   rslds.most_likely_states(xhat_lem[0], y[0], input=inpt[0])))
    zhat_lem = rslds.most_likely_states(xhat_lem[0], y[0])

    # store rslds
    rslds_lem = copy.deepcopy(rslds)

    # Plot loglikelihood of data
    plt.figure()
    plt.plot(q_elbos_lem[1:], label="Laplace-EM")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("ELBO")