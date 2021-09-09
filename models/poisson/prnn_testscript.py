import sys, os, time

import copy
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')
from det_rnn import *
import det_rnn.train as utils_train
import det_rnn.analysis as utils_analysis
import tensorflow as tf
from det_rnn.train.model import Model
from utils.plotfnc import *
from models.poisson.poisson_decoder import poisson_decoder
from models.poisson.pRNN import pRNN

# model_dir = "/Users/JRyu/github/det_rnn/experiments/naturalprior/200622/"
# model_dir = "D:/proj/det_rnn/experiments/naturalprior/200622/"
# model_dir = "../experiments//200622/"
# os.makedirs(model_dir, exist_ok=True)

ALLDTYPE = tf.float32
nori = 30

###### Generate stimulus ######

# training set
trials = []
par1 = copy.deepcopy(par)
par1['design'] = {'stim': (0, 0.3), 'delay'   : ((0.3,0.4),(0.5,0.6)),
                  'decision': (0.4, 0.5),'estim'   : (0.6, 0.8), 'iti': (0.8,0.9)}
par1['batch_size']  = nori # make it somewhat continuous
par1['n_ori']       = nori # make it somewhat continuous
par1['kappa']       = 1 # stimulus uncertainty
par1['n_tuned_input'] = nori
par1['n_tuned_output'] = nori
par1['reference']  = [0] # make it somewhat continuous
par1 = update_parameters(par1)
stim1 = Stimulus(par1)
trials += [stim1.generate_trial(balanced=True)]

par2 = copy.deepcopy(par1)
par2['kappa']  = 2 # stimulus uncertainty
stim2 = Stimulus(par2)
trials += [stim2.generate_trial(balanced=True)]

par3 = copy.deepcopy(par1)
par3['kappa']  = 3 # stimulus uncertainty
stim3 = Stimulus(par3)
trials += [stim3.generate_trial(balanced=True)]

par4 = copy.deepcopy(par1)
par4['kappa']  = 10 # stimulus uncertainty
stim4 = Stimulus(par4)
trials += [stim4.generate_trial(balanced=True)]

# show input tuning
plt.figure(figsize=(8,10))
plt.subplot(2,1,1)
for tr in range(4):
    plt.plot(trials[tr]['input_tuning'][int(nori/2),:], label='kappa='+str(trials[tr]['stimulus_kap'][0]))
plt.legend()
plt.title('Input tuning')

for tr in range(4):
    plt.subplot(2,4,5+tr)
    plt.imshow(trials[tr]['input_tuning'])
    plt.title('Tuning matrix kappa='+str(trials[tr]['stimulus_kap'][0]))
plt.show()

########### Poisson decoding ############
if False:
    nsamples = 1000
    gain = 100
    means = []
    vars = []
    decodingW = []
    trueW = []
    for tr in range(4):
        input_tuning = tf.cast(gain * trials[tr]['input_tuning'], dtype=ALLDTYPE)
        tuning = tf.repeat(input_tuning[:, None, :], nsamples, axis=1)  # (S,M,N))

        net = pRNN(nori, dtype=ALLDTYPE, normal_approx = True)
        decoder = poisson_decoder(dtype=ALLDTYPE)
        act = net.poisson_activation(tuning)

        H2    = input_tuning[0,:]
        H3      = tf.math.log(H2)

        if False:
            (S, M, N) = act.shape
            H1 = tf.zeros((N),dtype=ALLDTYPE)
            logp1 = tf.reduce_sum(decoder.logprob_unnorm(act ,H1))
            logp2 = tf.reduce_sum(decoder.logprob_unnorm(act, H2))
            logp3 = tf.reduce_sum(decoder.logprob_unnorm(act, H3))

        cirmean, circvar, H, ml = decoder.decode(act, H0=H3)
        means += [cirmean]
        vars += [circvar]
        decodingW += [H.numpy()]
        trueW += [H3.numpy()]

    plt.figure(figsize=(8,8))
    plt.subplot(2,2,1)
    for tr in range(4):
        plt.plot(np.linspace(0,np.pi,nori),means[tr],
                 label='kappa='+str(trials[tr]['stimulus_kap'][0]))
    plt.title('Decoded circular mean')
    plt.xlabel('Orientation')
    plt.ylabel('Orientation')
    plt.legend()

    plt.subplot(2,2,2)
    for tr in range(4):
        plt.plot(np.linspace(0,np.pi,nori),vars[tr],
                 label='kappa='+str(trials[tr]['stimulus_kap'][0]))
    plt.title('Decoded circular variance')
    plt.xlabel('Orientation')
    plt.ylabel('rad')
    plt.yscale('log')
    plt.legend()

    plt.subplot(2,2,3)
    for tr in range(4):
        plt.plot(np.linspace(0,np.pi,nori),decodingW[tr],
                 label='recovered, kappa='+str(trials[tr]['stimulus_kap'][0]))
        plt.plot(np.linspace(0,np.pi,nori),trueW[tr], ':',
                 label='log tuning, kappa='+str(trials[tr]['stimulus_kap'][0]))
    plt.title('Decoding weights')
    plt.legend()
    plt.show()

# information loss with constant subtraction or rotation
if False:
    means = []
    vars = []
    decodingW = []
    alllabels = ['kappa=1', 'gain increased', 'global inhibition', 'linear transf']

    tr = 1
    gain = 200
    nsamples = 10000
    input_tuning = tf.cast(gain * trials[tr]['input_tuning'], dtype=ALLDTYPE)
    tuning = tf.cast(tf.repeat(input_tuning[:, None, :], nsamples, axis=1), dtype=ALLDTYPE)  # (S,M,N))
    H2 = input_tuning[0, :]
    H3 = tf.math.log(H2)

    gain2 = 300
    input_tuning2 = tf.cast(gain2 * trials[tr]['input_tuning'], dtype=ALLDTYPE)
    tuning2 = tf.repeat(input_tuning2[:, None, :], nsamples, axis=1)

    net         = pRNN(nori, dtype=ALLDTYPE, normal_approx = True)
    decoder     = poisson_decoder(dtype=ALLDTYPE)
    act         = net.poisson_activation(tuning)
    cirmean, circvar, H, ml = decoder.decode(act, H0=H3)
    means += [cirmean];vars += [circvar];decodingW+=[H.numpy()];

    act_gain    = net.poisson_activation(tuning2) # bigger gain
    cirmean, circvar, H, ml = decoder.decode(act_gain, H0=H3)
    means += [cirmean];vars += [circvar];decodingW+=[H.numpy()];

    act_inh     = tf.nn.relu(act - gain/10)
    cirmean, circvar, H, ml = decoder.decode(act_inh, H0=H3)
    means += [cirmean];vars += [circvar];decodingW+=[H.numpy()];

    m1 = tf.random.normal((1, 1, nori, nori*100))
    m2 = tf.random.normal((1, 1, nori * 100,nori))
    # lintrans = m1@m2
    lintrans = trials[2]['input_tuning']
    act_rot  = lintrans[None,None,:,:] @ act[:,:,:,None]
    decH = (lintrans.T @ H3[:, None])
    cirmean, circvar, H, ml = decoder.decode(act_rot[:,:,:,0], H0=None) #decH[:,0])
    means += [cirmean]; vars += [circvar];decodingW+=[H.numpy()];

    plt.figure(figsize=(8,8))
    plt.subplot(2,2,1)
    plt.plot(np.linspace(0,np.pi,nori), act[0,0,:], label = alllabels[0])
    plt.plot(np.linspace(0, np.pi, nori), act_gain[0, 0, :], label = alllabels[1])
    plt.plot(np.linspace(0, np.pi, nori), act_inh[0, 0, :], label = alllabels[2])
    plt.plot(np.linspace(0, np.pi, nori), act_rot[0, 0, :, 0], label = alllabels[3])
    plt.legend()
    plt.title('Single trial Activation')

    plt.subplot(2, 2, 2)
    for tr in range(4):
        plt.plot(np.linspace(0,np.pi,nori),means[tr],label=alllabels[tr],
                 alpha = 0.4)
    plt.title('Decoded circular mean')
    plt.xlabel('Orientation')
    plt.ylabel('Orientation')
    plt.legend()

    plt.subplot(2,2,3)
    for tr in range(4):
        plt.plot(np.linspace(0,np.pi,nori),vars[tr],label=alllabels[tr],
                 alpha=0.4)
    plt.title('Decoded circular variance')
    plt.xlabel('Orientation')
    plt.ylabel('rad')
    plt.yscale('log')
    plt.legend()

    plt.subplot(2,2,4)
    for tr in range(4):
        plt.plot(np.linspace(0,np.pi,nori), decodingW[tr],
                 label=alllabels[tr],alpha=0.4)
    plt.title('Decoding weights')
    plt.legend()
    plt.show()

########### Poisson weighting ############

if True:
    gain = 200
    nsubpop = 1000  # number of repeated neurons with tuning
    tr = 1
    input_tuning = tf.cast(gain * trials[tr]['input_tuning'], dtype=ALLDTYPE)
    tuning  = tf.cast(tf.repeat(input_tuning[:, None, :], nsubpop, axis=1), dtype=ALLDTYPE)  # (S,M,N))
    net     = pRNN(nori, dtype=ALLDTYPE, normal_approx=True)
    decoder = poisson_decoder(dtype=ALLDTYPE)

    dgain = 50
    ref0 = np.pi/2
    decision_tuning         = np.zeros((nori,2))
    for ori in range(nori):
        stim_ori = ori*np.pi/nori
        if np.sin(2 * (ref0 - stim_ori)) > 0:
            decision_tuning[ori,:] = [dgain,0]
        else:
            decision_tuning[ori, :] = [0,dgain]
    decision_tuning = tf.constant(decision_tuning,dtype=ALLDTYPE)
    dec_tuning_samples = tf.repeat(decision_tuning[:, None, :], nsubpop, axis=1)

    ref1 = np.pi*2/3
    decision_tuning_wrong = np.zeros((nori, 2))
    for ori in range(nori):
        stim_ori = ori*np.pi/nori
        if np.sin(2 * (ref1 - stim_ori)) > 0:
            decision_tuning_wrong[ori,:] = [dgain,0]
        else:
            decision_tuning_wrong[ori, :] = [0,dgain]
    decision_tuning_wrong = tf.constant(decision_tuning_wrong,dtype=ALLDTYPE)
    dec_wrongtuning_samples = tf.repeat(decision_tuning_wrong[:, None, :], nsubpop, axis=1)

    act_stim = net.poisson_activation(tuning)
    act_dec = net.poisson_activation(dec_tuning_samples)
    act_comb_corr = input_tuning @ act_stim[:,:,:,None] + decision_tuning @ act_dec[:,:,:,None]
    act_comb_wrong = input_tuning @ act_stim[:, :, :, None] + \
                     decision_tuning_wrong @ act_dec[:, :, :, None]

    cirmean0, circvar0, H0, ml0 = decoder.decode(act_stim, H0=None)
    cirmean1, circvar1, H1, ml1 = decoder.decode(act_comb_corr[:,:,:,0], H0=None)
    cirmean2, circvar2, H2, ml2 = decoder.decode(act_comb_wrong[:, :, :, 0], H0=None)

    plt.figure(figsize=(10,8))
    plt.subplot(3, 2, 1)
    plt.imshow(input_tuning.numpy().T)
    plt.title('stimulus tuning')

    plt.subplot(6, 2, 2)
    plt.imshow(decision_tuning.numpy().T)
    plt.title('decision tuning')

    plt.subplot(6, 2, 4)
    plt.imshow(decision_tuning_wrong.numpy().T)
    plt.title('decision wrong tuning')

    plt.subplot(3,2,3)
    plt.plot(np.linspace(0, np.pi, nori), cirmean0, label='original')
    plt.plot(np.linspace(0,np.pi,nori), cirmean1, label='combination_corr')
    plt.plot(np.linspace(0, np.pi, nori), cirmean2, label='combination_wrong')
    plt.title('Decoded circular mean')
    plt.xlabel('Orientation')
    plt.ylabel('Orientation')
    plt.legend()

    plt.subplot(3,2,4)
    plt.plot(np.linspace(0, np.pi, nori), circvar0, label='original')
    plt.plot(np.linspace(0,np.pi,nori), circvar1, label='combination_corr')
    plt.plot(np.linspace(0, np.pi, nori), circvar2, label='combination_wrong')
    plt.title('Decoded circular variance')
    plt.xlabel('Orientation')
    plt.ylabel('rad')
    plt.yscale('log')
    plt.legend()

    plt.subplot(3,2,5)
    plt.plot(np.linspace(0,np.pi,nori),H0.numpy(), label='original')
    plt.plot(np.linspace(0, np.pi, nori), H1.numpy(), label='combination_corr')
    plt.plot(np.linspace(0, np.pi, nori), H2.numpy(), label='combination_wrong')
    plt.title('Decoding weights')
    plt.legend()
    plt.show()
    plt.legend()

########### Poisson network ############
if False:
    gain = 200
    nsubpop = 1000  # number of repeated neurons with tuning
    tr = 1
    input_tuning = tf.cast(gain * trials[tr]['input_tuning'], dtype=ALLDTYPE)
    tuning = tf.cast(tf.repeat(input_tuning[:, None, :], nsubpop, axis=1), dtype=ALLDTYPE)  # (S,M,N))

    rnnmat = trials[3]['input_tuning'] # sharp tuning
    e, v = tf.linalg.eigh(trials[3]['input_tuning'])

    net = pRNN(nori, rnnmat=rnnmat, dtype=ALLDTYPE, normal_approx=True)
    net_gauss = pRNN(nori, rnnmat=rnnmat, dtype=ALLDTYPE, normal_approx=True, noise_fixed=gain / 10)
    decoder = poisson_decoder(dtype=ALLDTYPE)

    T = 50
    data_pois = net(net.poisson_activation(tuning), T)  # (T, S, M, N, 1)
    data_gauss = net_gauss(net_gauss.poisson_activation(tuning), T)

    plt.figure(figsize=(8, 10))
    idx = int(nori / 2)
    plt.subplot(2, 1, 1)
    plt.imshow(data_pois[:, idx, 0, :, 0].numpy().T)
    plt.title('Poisson-like firing')
    plt.colorbar()

    plt.subplot(2, 1, 2)
    plt.imshow(data_gauss[:, idx, 0, :, 0].numpy().T)
    plt.title('Gaussian-like firing')
    plt.colorbar()

    plt.show()
    print('done.')

