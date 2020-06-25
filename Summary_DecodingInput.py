import pickle
import os
from det_rnn import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from scipy import stats

iModel = 4
BatchSize_svm = 100
silence_timestep_in2in = np.array([0, 0]) # stimon=[150, 300], est=[450, 600]
time_train = np.array([225, 375, 525])

iteration_goal              = 10000
iteration_load              = 10000
BatchSize                   = 100
scale_gamma                 = 0.5
n_hidden                    = 150
connect_p_within            = 0.8
connect_p_adjacent_forward  = 0.7
connect_p_distant_forward   = 0.0
connect_p_adjacent_back     = 0.3
connect_p_distant_back      = 0.0
n_orituned_neurons          = 30
dxtick                      = 1000 # in ms
alpha_input                 = 0.7 # Chaudhuri et al., Neuron, 2015
alpha_hidden                = 0.2
alpha_output                = 0.5 # Chaudhuri et al., Neuron, 2015; Motor (F1) cortex's decay is in between input and hidden
delay_train                 = 1.5

nCrossVal = 10

par['n_tuned_input'] = n_orituned_neurons
par['n_tuned_output'] = n_orituned_neurons
par['n_ori'] = n_orituned_neurons
par['n_hidden'] = n_hidden
par['batch_size'] = BatchSize
par['scale_gamma'] = scale_gamma
par['connect_prob_within_module'] = connect_p_within
par['connect_prob_adjacent_module_forward'] = connect_p_adjacent_forward
par['connect_prob_distant_module_forward'] = connect_p_distant_forward
par['connect_prob_adjacent_module_back'] = connect_p_adjacent_back
par['connect_prob_distant_module_back'] = connect_p_distant_back
par['silence_timestep_in2in'] = silence_timestep_in2in
par['alpha_input'] = alpha_input 	# Chaudhuri et al., Neuron, 2015
par['alpha_hidden'] = alpha_hidden
par['alpha_output'] = alpha_output  # Chaudhuri et al., Neuron, 2015; Motor (F1) cortex has similar decay profile with sensory cortex

par['design'].update({'iti'     : (0, 1.5),
                      'stim'    : (1.5, 3.0),
                      'delay'   : (3.0, 3.0 + delay_train),
                      'estim'   : (3.0 + delay_train, 4.5 + delay_train)})

savedir = os.path.dirname(os.path.realpath(__file__)) + \
           '/savedir/connectp_w' + str(connect_p_within) + '_forward_a' + str(connect_p_adjacent_forward) + 'd' + str(connect_p_distant_forward) + \
           'back_a' + str(connect_p_adjacent_back) + 'd' + str(connect_p_distant_back) + 'scalegamma' + str(scale_gamma) + \
           '/alpha_in' + str(par['alpha_input']) + '_h' + str(par['alpha_hidden']) + '_out' + str(par['alpha_output']) + \
           '/nIter' + str(iteration_goal) + 'BatchSize' + str(BatchSize) + '/Delay' + str(delay_train) + '/iModel' + str(iModel)

svmdir = '/svm/Batchsize' + str(BatchSize_svm) + '/silencing_input' + str(silence_timestep_in2in[0]) + 'to' + str(silence_timestep_in2in[1])

if not os.path.isdir(savedir + svmdir):
    os.makedirs(savedir + svmdir)

modelname = '/Iter' + str(iteration_load) + '.pkl'
fn = savedir + modelname
model = pickle.load(open(fn, 'rb'))
w_rnn2in_sparse_mask = model['parameters']['modular_sparse_mask'] # this sparse mask should be identical with the training and testing

par['batch_size'] = BatchSize_svm
par = update_parameters(par)
par_model = model['parameters']

h = model['h'][-1].numpy().astype('float32')
w_rnn = model['w_rnn'][-1].numpy().astype('float32')
b_rnn = model['b_rnn'][-1].numpy().astype('float32')

iw_rnn = par['EI_mask'] @ (w_rnn2in_sparse_mask * tf.nn.relu(w_rnn))

var_dict = {}
var_dict['h'] = h
var_dict['w_rnn'] = w_rnn
var_dict['b_rnn'] = b_rnn

dxtick = dxtick/10

## load stimulus for simulation result

stimulus = Stimulus(par)
trial_info = stimulus.generate_trial()

in_data = trial_info['neural_input'].astype('float32')
out_target = trial_info['desired_output']
mask_train = trial_info['mask']
batch_size = par['batch_size']
syn_x_init = par['syn_x_init']
syn_u_init = par['syn_u_init']

def rnn_cell(rnn_input, h, syn_x, syn_u, w_rnn):

    rnn_input = tf.concat((rnn_input, tf.zeros((par['batch_size'], par['n_hidden'] - par['n_input']))), axis=1)

    syn_x += (par['alpha_std'] * (1 - syn_x) - par['dt']/1000 * syn_u * syn_x * h)  # what is alpha_std???
    syn_u += (par['alpha_stf'] * (par['U'] - syn_u) + par['dt']/1000 * par['U'] * (1 - syn_u) * h)

    syn_x = tf.minimum(np.float32(1), tf.nn.relu(syn_x))
    syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
    h_post = syn_u * syn_x * h
    # h_post = h

    noise_rnn = np.sqrt(2*par['alpha_mask'])*par['noise_rnn_sd']
    h = tf.nn.relu((1 - par['alpha_mask']) * h
                   + par['alpha_mask'] * (rnn_input
                                            + h_post @ w_rnn
                                            + var_dict['b_rnn'])
                   + tf.random.normal(h.shape, 0, noise_rnn, dtype=tf.float32))

    return h, syn_x, syn_u

def run_model(in_data, var_dict, syn_x_init, syn_u_init):

    self_h = np.zeros((par['n_timesteps'], par['batch_size'], par['n_hidden']), dtype=np.float32)
    self_syn_x = np.zeros((par['n_timesteps'], par['batch_size'], par['n_hidden']), dtype=np.float32)
    self_syn_u = np.zeros((par['n_timesteps'], par['batch_size'], par['n_hidden']), dtype=np.float32)

    h = np.ones((par['batch_size'], 1)) @ var_dict['h']
    syn_x = syn_x_init
    syn_u = syn_u_init
    w_rnn = par['EI_mask'] @ (w_rnn2in_sparse_mask * tf.nn.relu(var_dict['w_rnn']))

    c = 0
    for rnn_input in in_data:

        if c >= silence_timestep_in2in[0] and c <= silence_timestep_in2in[1]:
            iw_rnn = w_rnn * par['input_silencing_mask']
        else:
            iw_rnn = w_rnn

        h, syn_x, syn_u = rnn_cell(rnn_input, h, syn_x, syn_u, iw_rnn)
        #
        self_h[c, :, :] = h
        self_syn_x[c, :, :] = syn_x
        self_syn_u[c, :, :] = syn_u
        c += 1

    return self_h, self_syn_x, self_syn_u, w_rnn

h, syn_x, syn_u, w_rnn \
    = run_model(in_data, var_dict, syn_x_init, syn_u_init)

inputs = h[:, :, :par['n_input']]
hidden = h[:, :, par['n_input'] : (par['n_hidden'] - par['n_output'])]
output = h[:, :, -par['n_output']:]

syn = syn_x * syn_u
syn_inputs = syn[:, :, :par['n_input']]
syn_hidden = syn[:, :, par['n_input'] : (par['n_hidden'] - par['n_output'])]
syn_output = syn[:, :, -par['n_output']:]

## figure conditions

esti_on = par['design']['estim'][0]*np.array(1000/par['dt'])
stim_onoff = par['design']['stim']*np.array(1000/par['dt'])
ivmax = 0.5
ivmin = 0.05
ichance = np.array([1.0, 1.0])/(par['n_tuned_input']*1.0)

## 1. decoding from input firing rate - vector

boundCrossVal = np.linspace(0, par['batch_size'], nCrossVal + 1, dtype='int')
Accuracy_input = np.zeros(par['n_timesteps'])
for itime in range(par['n_timesteps']):
    print('1. decoding from input firing rate 1 - vector ' + str(itime))
    ipredict = np.zeros((par['batch_size']))
    for iCrossVal in range(nCrossVal):
        iInd_test = np.arange(boundCrossVal[iCrossVal], boundCrossVal[iCrossVal+1])
        iInd_train = np.arange(0, par['batch_size'])
        iInd_train = np.delete(iInd_train, iInd_test, axis=0)

        itrain_input = inputs[itime, iInd_train, :]
        itrain_stim = trial_info['stimulus_ori'][iInd_train]
        itest_input = inputs[itime, iInd_test, :]
        itest_stim = trial_info['stimulus_ori'][iInd_test]

        svm_model_linear = SVC(kernel='linear', C=1).fit(itrain_input, itrain_stim)
        svm_predictions = svm_model_linear.predict(itest_input)

        ipredict[iInd_test] = svm_predictions
    Accuracy_input[itime] = np.sum(ipredict == trial_info['stimulus_ori'])/par['batch_size']

itrain_stim = trial_info['stimulus_ori']
Accuracy_acrosstime_vector_input = np.zeros((par['n_timesteps'], 3))
q = 0
for itime_train in time_train:
    itrain_input = inputs[itime_train, :, :]
    svm_model_linear = SVC(kernel='linear', C=1).fit(itrain_input, itrain_stim)
    for itime in range(par['n_timesteps']):
        print('1. decoding from input firing rate ' + str(q+2) + ' - vector ' + str(itime))
        ipredict = np.zeros((par['batch_size']))
        for iCrossVal in range(nCrossVal):
            iInd_test = np.arange(boundCrossVal[iCrossVal], boundCrossVal[iCrossVal+1])
            itest_input = inputs[itime, iInd_test, :]
            itest_stim = trial_info['stimulus_ori'][iInd_test]
            svm_predictions = svm_model_linear.predict(itest_input)

            ipredict[iInd_test] = svm_predictions
        Accuracy_acrosstime_vector_input[itime, q] = np.sum(ipredict == trial_info['stimulus_ori'])/par['batch_size']
    q += 1

plt.rcParams.update({'font.size': 20})
fig = plt.figure(figsize=(10, 7), dpi=80)
plt.clf()
plt.plot(Accuracy_input)
plt.plot(Accuracy_acrosstime_vector_input)
plt.rcParams.update({'font.size': 20})
plt.plot(np.array([0.0, par['n_timesteps']*1.0]), ichance, 'k--')
plt.plot(np.array([0.0, par['n_timesteps']*1.0]), np.array([0, 0]), 'k-')
plt.plot(np.array([1, 1])*esti_on, np.array([0, 1]),'k--')
plt.plot(np.array([1, 1])*stim_onoff[0], np.array([0, 1]),'k--')
plt.plot(np.array([1, 1])*stim_onoff[1], np.array([0, 1]),'k--')
plt.xlabel('time point')
plt.ylabel('classification accuracy')
plt.grid()
plt.savefig(savedir + svmdir + '/1_Accuracy_input' + str(iteration_load) + '.png', bbox_inches='tight')

## 2, decoding from hidden firing rate - vector

Accuracy_hidden = np.zeros(par['n_timesteps'])
for itime in range(par['n_timesteps']):
    print('2. decoding from hidden firing rate 1 - vector ' + str(itime))
    ipredict = np.zeros((par['batch_size']))
    for iCrossVal in range(nCrossVal):
        iInd_test = np.arange(boundCrossVal[iCrossVal], boundCrossVal[iCrossVal+1])
        iInd_train = np.arange(0, par['batch_size'])
        iInd_train = np.delete(iInd_train, iInd_test, axis=0)

        itrain_h = hidden[itime, iInd_train, :]
        itrain_stim = trial_info['stimulus_ori'][iInd_train]
        itest_h = hidden[itime, iInd_test, :]
        itest_stim = trial_info['stimulus_ori'][iInd_test]

        svm_model_linear = SVC(kernel='linear', C=1).fit(itrain_h, itrain_stim)
        svm_predictions = svm_model_linear.predict(itest_h)

        ipredict[iInd_test] = svm_predictions
    Accuracy_hidden[itime] = np.sum(ipredict == trial_info['stimulus_ori'])/par['batch_size']

itrain_stim = trial_info['stimulus_ori']
itime_train = 200
Accuracy_acrosstime_vector_hidden = np.zeros((par['n_timesteps'], 3))
q = 0
for itime_train in time_train:
    itrain_h = hidden[itime_train, :, :]
    svm_model_linear = SVC(kernel='linear', C=1).fit(itrain_h, itrain_stim)
    for itime in range(par['n_timesteps']):
        print('2. decoding from hidden firing rate ' + str(q+2) + ' - vector ' + str(itime))
        ipredict = np.zeros((par['batch_size']))
        for iCrossVal in range(nCrossVal):
            iInd_test = np.arange(boundCrossVal[iCrossVal], boundCrossVal[iCrossVal+1])
            itest_h = hidden[itime, iInd_test, :]
            itest_stim = trial_info['stimulus_ori'][iInd_test]
            svm_predictions = svm_model_linear.predict(itest_h)

            ipredict[iInd_test] = svm_predictions
        Accuracy_acrosstime_vector_hidden[itime, q] = np.sum(ipredict == trial_info['stimulus_ori'])/par['batch_size']
    q += 1

fig = plt.figure(figsize=(10, 7), dpi=80)
plt.clf()
plt.plot(Accuracy_hidden)
plt.plot(Accuracy_acrosstime_vector_hidden)
plt.rcParams.update({'font.size': 20})
plt.plot(np.array([0.0, par['n_timesteps']*1.0]), ichance, 'k--')
plt.plot(np.array([0.0, par['n_timesteps']*1.0]), np.array([0, 0]), 'k-')
plt.plot(np.array([1, 1])*esti_on, np.array([0, 1]),'k--')
plt.plot(np.array([1, 1])*stim_onoff[0], np.array([0, 1]),'k--')
plt.plot(np.array([1, 1])*stim_onoff[1], np.array([0, 1]),'k--')
plt.xlabel('time point')
plt.ylabel('classification accuracy')
plt.grid()
plt.savefig(savedir + svmdir + '/2_Accuracy_hidden' + str(iteration_load) + '.png', bbox_inches='tight')

## 3, decoding from out firing rate - vector

Accuracy_output = np.zeros(par['n_timesteps'])
for itime in range(par['n_timesteps']):
    print('3. decoding from out firing rate 1 - vector ' + str(itime))
    ipredict = np.zeros((par['batch_size']))
    for iCrossVal in range(nCrossVal):
        iInd_test = np.arange(boundCrossVal[iCrossVal], boundCrossVal[iCrossVal+1])
        iInd_train = np.arange(0, par['batch_size'])
        iInd_train = np.delete(iInd_train, iInd_test, axis=0)

        itrain_out = output[itime, iInd_train, :]
        itrain_stim = trial_info['stimulus_ori'][iInd_train]
        itest_out = output[itime, iInd_test, :]
        itest_stim = trial_info['stimulus_ori'][iInd_test]

        svm_model_linear = SVC(kernel='linear', C=1).fit(itrain_out, itrain_stim)
        svm_predictions = svm_model_linear.predict(itest_out)

        ipredict[iInd_test] = svm_predictions
    Accuracy_output[itime] = np.sum(ipredict == trial_info['stimulus_ori'])/par['batch_size']

itrain_stim = trial_info['stimulus_ori']
Accuracy_acrosstime_vector_output = np.zeros((par['n_timesteps'], 3))
q = 0
for itime_train in time_train:
    itrain_out = output[itime_train, :, :]
    svm_model_linear = SVC(kernel='linear', C=1).fit(itrain_out, itrain_stim)
    for itime in range(par['n_timesteps']):
        print('3. decoding from out firing rate ' + str(q+2) + ' - vector ' + str(itime))
        ipredict = np.zeros((par['batch_size']))
        for iCrossVal in range(nCrossVal):
            iInd_test = np.arange(boundCrossVal[iCrossVal], boundCrossVal[iCrossVal+1])
            itest_out = output[itime, iInd_test, :]
            itest_stim = trial_info['stimulus_ori'][iInd_test]
            svm_predictions = svm_model_linear.predict(itest_out)

            ipredict[iInd_test] = svm_predictions
        Accuracy_acrosstime_vector_output[itime, q] = np.sum(ipredict == trial_info['stimulus_ori'])/par['batch_size']
    q += 1

fig = plt.figure(figsize=(10, 7), dpi=80)
plt.clf()
plt.plot(Accuracy_output)
plt.plot(Accuracy_acrosstime_vector_output)
plt.rcParams.update({'font.size': 20})
plt.plot(np.array([0.0, par['n_timesteps']*1.0]), ichance, 'k--')
plt.plot(np.array([0.0, par['n_timesteps']*1.0]), np.array([0, 0]), 'k-')
plt.plot(np.array([1, 1])*esti_on, np.array([0, 1]),'k--')
plt.plot(np.array([1, 1])*stim_onoff[0], np.array([0, 1]),'k--')
plt.plot(np.array([1, 1])*stim_onoff[1], np.array([0, 1]),'k--')
plt.plot(np.array([1, 1])*itime_train, np.array([0, 1]),'r--')
plt.xlabel('time point')
plt.ylabel('classification accuracy')
plt.grid()
plt.savefig(savedir + svmdir + '/3_Accuracy_output' + str(iteration_load) + '.png', bbox_inches='tight')

## 4. decoding from input firing rate - matrix

Accuracy_acrosstime_matrix_input = np.zeros((par['n_timesteps'], par['n_timesteps']))
for itime_train in range(par['n_timesteps']):
    print('4. decoding from input firing rate - matrix ' + str(itime_train))
    itrain_input = inputs[itime_train, :, :]
    svm_model_linear = SVC(kernel='linear', C=1).fit(itrain_input, trial_info['stimulus_ori'])
    for itime_test in range(par['n_timesteps']):
        itest_input = inputs[itime_test, :, :]
        svm_predictions = svm_model_linear.predict(itest_input)
        iaccuracy = np.sum(svm_predictions == trial_info['stimulus_ori'])/par['batch_size']
        Accuracy_acrosstime_matrix_input[itime_train, itime_test] = iaccuracy

fig = plt.figure(figsize=(13, 13), dpi=80)
plt.rcParams.update({'font.size': 20})
plt.imshow(Accuracy_acrosstime_matrix_input, vmin=ivmin, vmax=ivmax)
plt.plot(np.array([1, 1])*esti_on, np.array([0, 1])*par['n_timesteps'], 'k--')
plt.plot(np.array([0, 1])*par['n_timesteps'], np.array([1, 1])*esti_on, 'k--')
plt.plot(np.array([1, 1])*stim_onoff[0], np.array([0, 1])*par['n_timesteps'], 'k--')
plt.plot(np.array([0, 1])*par['n_timesteps'], np.array([1, 1])*stim_onoff[0], 'k--')
plt.plot(np.array([1, 1])*stim_onoff[1], np.array([0, 1])*par['n_timesteps'], 'k--')
plt.plot(np.array([0, 1])*par['n_timesteps'], np.array([1, 1])*stim_onoff[1], 'k--')
plt.xlabel('test time step')
plt.ylabel('train time step')
plt.title('by input firing rate')
plt.xlim([0, par['n_timesteps']])
plt.ylim([0, par['n_timesteps']])
plt.colorbar()
plt.savefig(savedir + svmdir + '/4_Accuracy_input_matrix' + str(iteration_load) + '.png', bbox_inches='tight')

## 5. decoding from hidden firing rate - maxtix

Accuracy_acrosstime_matrix_hidden = np.zeros((par['n_timesteps'], par['n_timesteps']))
for itime_train in range(par['n_timesteps']):
    print('5. decoding from hidden firing rate - maxtix ' + str(itime_train))
    itrain_h = hidden[itime_train, :, :]
    svm_model_linear = SVC(kernel='linear', C=1).fit(itrain_h, trial_info['stimulus_ori'])
    for itime_test in range(par['n_timesteps']):
        itest_h = hidden[itime_test, :, :]
        svm_predictions = svm_model_linear.predict(itest_h)
        iaccuracy = np.sum(svm_predictions == trial_info['stimulus_ori']) / par['batch_size']
        Accuracy_acrosstime_matrix_hidden[itime_train, itime_test] = iaccuracy

fig = plt.figure(figsize=(13, 13), dpi=80)
plt.rcParams.update({'font.size': 20})
plt.imshow(Accuracy_acrosstime_matrix_hidden, vmin=ivmin, vmax=ivmax)
plt.plot(np.array([1, 1])*esti_on, np.array([0, 1])*par['n_timesteps'], 'k--')
plt.plot(np.array([0, 1])*par['n_timesteps'], np.array([1, 1])*esti_on, 'k--')
plt.plot(np.array([1, 1])*stim_onoff[0], np.array([0, 1])*par['n_timesteps'], 'k--')
plt.plot(np.array([0, 1])*par['n_timesteps'], np.array([1, 1])*stim_onoff[0], 'k--')
plt.plot(np.array([1, 1])*stim_onoff[1], np.array([0, 1])*par['n_timesteps'], 'k--')
plt.plot(np.array([0, 1])*par['n_timesteps'], np.array([1, 1])*stim_onoff[1], 'k--')
plt.xlabel('test time step')
plt.ylabel('train time step')
plt.title('by hidden firing rate')
plt.xlim([0, par['n_timesteps']])
plt.ylim([0, par['n_timesteps']])
plt.colorbar()
plt.savefig(savedir + svmdir + '/5_Accuracy_hidden_matrix' + str(iteration_load) + '.png', bbox_inches='tight')

## 6. decoding from output firing rate - maxtix

Accuracy_acrosstime_matrix_output = np.zeros((par['n_timesteps'], par['n_timesteps']))
for itime_train in range(par['n_timesteps']):
    print('6. decoding from output firing rate - maxtix ' + str(itime_train))
    itrain_h = output[itime_train, :, :]
    svm_model_linear = SVC(kernel='linear', C=1).fit(itrain_h, trial_info['stimulus_ori'])
    for itime_test in range(par['n_timesteps']):
        itest_h = output[itime_test, :, :]
        svm_predictions = svm_model_linear.predict(itest_h)
        iaccuracy = np.sum(svm_predictions == trial_info['stimulus_ori']) / par['batch_size']
        Accuracy_acrosstime_matrix_output[itime_train, itime_test] = iaccuracy

fig = plt.figure(figsize=(13, 13), dpi=80)
plt.rcParams.update({'font.size': 20})
plt.imshow(Accuracy_acrosstime_matrix_output, vmin=ivmin, vmax=ivmax)
plt.plot(np.array([1, 1])*esti_on, np.array([0, 1])*par['n_timesteps'], 'k--')
plt.plot(np.array([0, 1])*par['n_timesteps'], np.array([1, 1])*esti_on, 'k--')
plt.plot(np.array([1, 1])*stim_onoff[0], np.array([0, 1])*par['n_timesteps'], 'k--')
plt.plot(np.array([0, 1])*par['n_timesteps'], np.array([1, 1])*stim_onoff[0], 'k--')
plt.plot(np.array([1, 1])*stim_onoff[1], np.array([0, 1])*par['n_timesteps'], 'k--')
plt.plot(np.array([0, 1])*par['n_timesteps'], np.array([1, 1])*stim_onoff[1], 'k--')
plt.xlabel('test time step')
plt.ylabel('train time step')
plt.title('by output firing rate')
plt.xlim([0, par['n_timesteps']])
plt.ylim([0, par['n_timesteps']])
plt.colorbar()
plt.savefig(savedir + svmdir + '/6_Accuracy_output_matrix' + str(iteration_load) + '.png', bbox_inches='tight')

## 7. decoding from input synapse - vector

boundCrossVal = np.linspace(0, par['batch_size'], nCrossVal + 1, dtype='int')
Accuracy_syn_input = np.zeros(par['n_timesteps'])
for itime in range(par['n_timesteps']):
    print('7. decoding from input synapse 1 - vector ' + str(itime))
    ipredict = np.zeros((par['batch_size']))
    for iCrossVal in range(nCrossVal):
        iInd_test = np.arange(boundCrossVal[iCrossVal], boundCrossVal[iCrossVal+1])
        iInd_train = np.arange(0, par['batch_size'])
        iInd_train = np.delete(iInd_train, iInd_test, axis=0)

        itrain_input = syn_inputs[itime, iInd_train, :]
        itrain_stim = trial_info['stimulus_ori'][iInd_train]
        itest_input = syn_inputs[itime, iInd_test, :]
        itest_stim = trial_info['stimulus_ori'][iInd_test]

        svm_model_linear = SVC(kernel='linear', C=1).fit(itrain_input, itrain_stim)
        svm_predictions = svm_model_linear.predict(itest_input)

        ipredict[iInd_test] = svm_predictions
    Accuracy_syn_input[itime] = np.sum(ipredict == trial_info['stimulus_ori'])/par['batch_size']

itime_train = 200
itrain_input = syn_inputs[itime_train, :, :]
itrain_stim = trial_info['stimulus_ori']
svm_model_linear = SVC(kernel='linear', C=1).fit(itrain_input, itrain_stim)
Accuracy_acrosstime_vector_syn_input = np.zeros(par['n_timesteps'])
for itime in range(par['n_timesteps']):
    print('7. decoding from input synapse 2 - vector ' + str(itime))
    ipredict = np.zeros((par['batch_size']))
    for iCrossVal in range(nCrossVal):
        iInd_test = np.arange(boundCrossVal[iCrossVal], boundCrossVal[iCrossVal+1])
        itest_input = syn_inputs[itime, iInd_test, :]
        itest_stim = trial_info['stimulus_ori'][iInd_test]
        svm_predictions = svm_model_linear.predict(itest_input)

        ipredict[iInd_test] = svm_predictions
    Accuracy_acrosstime_vector_syn_input[itime] = np.sum(ipredict == trial_info['stimulus_ori'])/par['batch_size']

plt.rcParams.update({'font.size': 20})
fig = plt.figure(figsize=(10, 7), dpi=80)
plt.clf()
plt.plot(Accuracy_syn_input)
plt.plot(Accuracy_acrosstime_vector_syn_input)
plt.rcParams.update({'font.size': 20})
plt.plot(np.array([0.0, par['n_timesteps']*1.0]), ichance, 'k--')
plt.plot(np.array([0.0, par['n_timesteps']*1.0]), np.array([0, 0]), 'k-')
plt.plot(np.array([1, 1])*esti_on, np.array([0, 1]),'k--')
plt.plot(np.array([1, 1])*stim_onoff[0], np.array([0, 1]),'k--')
plt.plot(np.array([1, 1])*stim_onoff[1], np.array([0, 1]),'k--')
plt.plot(np.array([1, 1])*itime_train, np.array([0, 1]),'r--')
plt.xlabel('time point')
plt.ylabel('classification accuracy')
plt.savefig(savedir + svmdir + '/7_Accuracy_syn_input' + str(iteration_load) + '.png', bbox_inches='tight')

## 8, decoding from hidden synapse - vector

Accuracy_syn_hidden = np.zeros(par['n_timesteps'])
for itime in range(par['n_timesteps']):
    print('8, decoding from hidden synapse 1 - vector ' + str(itime))
    ipredict = np.zeros((par['batch_size']))
    for iCrossVal in range(nCrossVal):
        iInd_test = np.arange(boundCrossVal[iCrossVal], boundCrossVal[iCrossVal+1])
        iInd_train = np.arange(0, par['batch_size'])
        iInd_train = np.delete(iInd_train, iInd_test, axis=0)

        itrain_h = syn_hidden[itime, iInd_train, :]
        itrain_stim = trial_info['stimulus_ori'][iInd_train]
        itest_h = syn_hidden[itime, iInd_test, :]
        itest_stim = trial_info['stimulus_ori'][iInd_test]

        svm_model_linear = SVC(kernel='linear', C=1).fit(itrain_h, itrain_stim)
        svm_predictions = svm_model_linear.predict(itest_h)

        ipredict[iInd_test] = svm_predictions
    Accuracy_syn_hidden[itime] = np.sum(ipredict == trial_info['stimulus_ori'])/par['batch_size']

itime_train = 200
itrain_h = syn_hidden[itime_train, :, :]
itrain_stim = trial_info['stimulus_ori']
svm_model_linear = SVC(kernel='linear', C=1).fit(itrain_h, itrain_stim)
Accuracy_acrosstime_vector_syn_hidden = np.zeros(par['n_timesteps'])
for itime in range(par['n_timesteps']):
    print('8, decoding from hidden synapse 2 - vector ' + str(itime))
    ipredict = np.zeros((par['batch_size']))
    for iCrossVal in range(nCrossVal):
        iInd_test = np.arange(boundCrossVal[iCrossVal], boundCrossVal[iCrossVal+1])
        itest_h = syn_hidden[itime, iInd_test, :]
        itest_stim = trial_info['stimulus_ori'][iInd_test]
        svm_predictions = svm_model_linear.predict(itest_h)

        ipredict[iInd_test] = svm_predictions
    Accuracy_acrosstime_vector_syn_hidden[itime] = np.sum(ipredict == trial_info['stimulus_ori'])/par['batch_size']

fig = plt.figure(figsize=(10, 7), dpi=80)
plt.clf()
plt.plot(Accuracy_syn_hidden)
plt.plot(Accuracy_acrosstime_vector_syn_hidden)
plt.rcParams.update({'font.size': 20})
plt.plot(np.array([0.0, par['n_timesteps']*1.0]), ichance, 'k--')
plt.plot(np.array([0.0, par['n_timesteps']*1.0]), np.array([0, 0]), 'k-')
plt.plot(np.array([1, 1])*esti_on, np.array([0, 1]),'k--')
plt.plot(np.array([1, 1])*stim_onoff[0], np.array([0, 1]),'k--')
plt.plot(np.array([1, 1])*stim_onoff[1], np.array([0, 1]),'k--')
plt.plot(np.array([1, 1])*itime_train, np.array([0, 1]),'r--')
plt.xlabel('time point')
plt.ylabel('classification accuracy')
plt.savefig(savedir + svmdir + '/8_Accuracy_syn_hidden' + str(iteration_load) + '.png', bbox_inches='tight')

## 9, decoding from output synapse - vector

Accuracy_syn_output = np.zeros(par['n_timesteps'])
for itime in range(par['n_timesteps']):
    print('9, decoding from output synapse 1 - vector ' + str(itime))
    ipredict = np.zeros((par['batch_size']))
    for iCrossVal in range(nCrossVal):
        iInd_test = np.arange(boundCrossVal[iCrossVal], boundCrossVal[iCrossVal+1])
        iInd_train = np.arange(0, par['batch_size'])
        iInd_train = np.delete(iInd_train, iInd_test, axis=0)

        itrain_h = syn_output[itime, iInd_train, :]
        itrain_stim = trial_info['stimulus_ori'][iInd_train]
        itest_h = syn_output[itime, iInd_test, :]
        itest_stim = trial_info['stimulus_ori'][iInd_test]

        svm_model_linear = SVC(kernel='linear', C=1).fit(itrain_h, itrain_stim)
        svm_predictions = svm_model_linear.predict(itest_h)

        ipredict[iInd_test] = svm_predictions
    Accuracy_syn_output[itime] = np.sum(ipredict == trial_info['stimulus_ori'])/par['batch_size']

itime_train = 200
itrain_h = syn_output[itime_train, :, :]
itrain_stim = trial_info['stimulus_ori']
svm_model_linear = SVC(kernel='linear', C=1).fit(itrain_h, itrain_stim)
Accuracy_acrosstime_vector_syn_output = np.zeros(par['n_timesteps'])
for itime in range(par['n_timesteps']):
    print('9, decoding from output synapse 2 - vector ' + str(itime))
    ipredict = np.zeros((par['batch_size']))
    for iCrossVal in range(nCrossVal):
        iInd_test = np.arange(boundCrossVal[iCrossVal], boundCrossVal[iCrossVal+1])
        itest_h = syn_output[itime, iInd_test, :]
        itest_stim = trial_info['stimulus_ori'][iInd_test]
        svm_predictions = svm_model_linear.predict(itest_h)

        ipredict[iInd_test] = svm_predictions
    Accuracy_acrosstime_vector_syn_output[itime] = np.sum(ipredict == trial_info['stimulus_ori'])/par['batch_size']

fig = plt.figure(figsize=(10, 7), dpi=80)
plt.clf()
plt.plot(Accuracy_syn_output)
plt.plot(Accuracy_acrosstime_vector_syn_output)
plt.rcParams.update({'font.size': 20})
plt.plot(np.array([0.0, par['n_timesteps']*1.0]), ichance, 'k--')
plt.plot(np.array([0.0, par['n_timesteps']*1.0]), np.array([0, 0]), 'k-')
plt.plot(np.array([1, 1])*esti_on, np.array([0, 1]),'k--')
plt.plot(np.array([1, 1])*stim_onoff[0], np.array([0, 1]),'k--')
plt.plot(np.array([1, 1])*stim_onoff[1], np.array([0, 1]),'k--')
plt.plot(np.array([1, 1])*itime_train, np.array([0, 1]),'r--')
plt.xlabel('time point')
plt.ylabel('classification accuracy')
plt.savefig(savedir + svmdir + '/9_Accuracy_syn_output' + str(iteration_load) + '.png', bbox_inches='tight')

## 10. decoding from input synapse - matrix

Accuracy_acrosstime_matrix_syn_input = np.zeros((par['n_timesteps'], par['n_timesteps']))
for itime_train in range(par['n_timesteps']):
    print('10. decoding from input synapse - matrix' + str(itime_train))
    itrain_input = syn_inputs[itime_train, :, :]
    svm_model_linear = SVC(kernel='linear', C=1).fit(itrain_input, trial_info['stimulus_ori'])
    for itime_test in range(par['n_timesteps']):
        itest_input = syn_inputs[itime_test, :, :]
        svm_predictions = svm_model_linear.predict(itest_input)
        iaccuracy = np.sum(svm_predictions == trial_info['stimulus_ori'])/par['batch_size']
        Accuracy_acrosstime_matrix_syn_input[itime_train, itime_test] = iaccuracy

fig = plt.figure(figsize=(13, 13), dpi=80)
plt.rcParams.update({'font.size': 20})
plt.imshow(Accuracy_acrosstime_matrix_syn_input, vmin=ivmin, vmax=ivmax)
plt.plot(np.array([1, 1])*esti_on, np.array([0, 1])*par['n_timesteps'], 'k--')
plt.plot(np.array([0, 1])*par['n_timesteps'], np.array([1, 1])*esti_on, 'k--')
plt.plot(np.array([1, 1])*stim_onoff[0], np.array([0, 1])*par['n_timesteps'], 'k--')
plt.plot(np.array([0, 1])*par['n_timesteps'], np.array([1, 1])*stim_onoff[0], 'k--')
plt.plot(np.array([1, 1])*stim_onoff[1], np.array([0, 1])*par['n_timesteps'], 'k--')
plt.plot(np.array([0, 1])*par['n_timesteps'], np.array([1, 1])*stim_onoff[1], 'k--')
plt.xlabel('test time step')
plt.ylabel('train time step')
plt.title('by input synapse')
plt.xlim([0, par['n_timesteps']])
plt.ylim([0, par['n_timesteps']])
plt.colorbar()
plt.savefig(savedir + svmdir + '/10_Accuracy_syn_input_matrix' + str(iteration_load) + '.png', bbox_inches='tight')

## 11. decoding from hidden synapse - maxtix

Accuracy_acrosstime_matrix_syn_hidden = np.zeros((par['n_timesteps'], par['n_timesteps']))
for itime_train in range(par['n_timesteps']):
    print('10. decoding from hidden synapse - maxtix ' + str(itime_train))
    itrain_h = syn_hidden[itime_train, :, :]
    svm_model_linear = SVC(kernel='linear', C=1).fit(itrain_h, trial_info['stimulus_ori'])
    for itime_test in range(par['n_timesteps']):
        itest_h = syn_hidden[itime_test, :, :]
        svm_predictions = svm_model_linear.predict(itest_h)
        iaccuracy = np.sum(svm_predictions == trial_info['stimulus_ori']) / par['batch_size']
        Accuracy_acrosstime_matrix_syn_hidden[itime_train, itime_test] = iaccuracy

fig = plt.figure(figsize=(13, 13), dpi=80)
plt.rcParams.update({'font.size': 20})
plt.imshow(Accuracy_acrosstime_matrix_syn_hidden, vmin=ivmin, vmax=ivmax)
plt.plot(np.array([1, 1])*esti_on, np.array([0, 1])*par['n_timesteps'], 'k--')
plt.plot(np.array([0, 1])*par['n_timesteps'], np.array([1, 1])*esti_on, 'k--')
plt.plot(np.array([1, 1])*stim_onoff[0], np.array([0, 1])*par['n_timesteps'], 'k--')
plt.plot(np.array([0, 1])*par['n_timesteps'], np.array([1, 1])*stim_onoff[0], 'k--')
plt.plot(np.array([1, 1])*stim_onoff[1], np.array([0, 1])*par['n_timesteps'], 'k--')
plt.plot(np.array([0, 1])*par['n_timesteps'], np.array([1, 1])*stim_onoff[1], 'k--')
plt.xlabel('test time step')
plt.ylabel('train time step')
plt.title('by hidden synapse')
plt.xlim([0, par['n_timesteps']])
plt.ylim([0, par['n_timesteps']])
plt.colorbar()
plt.savefig(savedir + svmdir + '/11_Accuracy_syn_hidden_matrix' + str(iteration_load) + '.png', bbox_inches='tight')

## 12. decoding from output synapse - maxtix

Accuracy_acrosstime_matrix_syn_output = np.zeros((par['n_timesteps'], par['n_timesteps']))
for itime_train in range(par['n_timesteps']):
    print('10. decoding from output synapse - maxtix ' + str(itime_train))
    itrain_h = syn_output[itime_train, :, :]
    svm_model_linear = SVC(kernel='linear', C=1).fit(itrain_h, trial_info['stimulus_ori'])
    for itime_test in range(par['n_timesteps']):
        itest_h = syn_output[itime_test, :, :]
        svm_predictions = svm_model_linear.predict(itest_h)
        iaccuracy = np.sum(svm_predictions == trial_info['stimulus_ori']) / par['batch_size']
        Accuracy_acrosstime_matrix_syn_output[itime_train, itime_test] = iaccuracy

fig = plt.figure(figsize=(13, 13), dpi=80)
plt.rcParams.update({'font.size': 20})
plt.imshow(Accuracy_acrosstime_matrix_syn_output, vmin=ivmin, vmax=ivmax)
plt.plot(np.array([1, 1])*esti_on, np.array([0, 1])*par['n_timesteps'], 'k--')
plt.plot(np.array([0, 1])*par['n_timesteps'], np.array([1, 1])*esti_on, 'k--')
plt.plot(np.array([1, 1])*stim_onoff[0], np.array([0, 1])*par['n_timesteps'], 'k--')
plt.plot(np.array([0, 1])*par['n_timesteps'], np.array([1, 1])*stim_onoff[0], 'k--')
plt.plot(np.array([1, 1])*stim_onoff[1], np.array([0, 1])*par['n_timesteps'], 'k--')
plt.plot(np.array([0, 1])*par['n_timesteps'], np.array([1, 1])*stim_onoff[1], 'k--')
plt.xlabel('test time step')
plt.ylabel('train time step')
plt.title('by output synapse')
plt.xlim([0, par['n_timesteps']])
plt.ylim([0, par['n_timesteps']])
plt.colorbar()
plt.savefig(savedir + svmdir + '/12_Accuracy_syn_output_matrix' + str(iteration_load) + '.png', bbox_inches='tight')

svm_results = { 'Accuracy_input': Accuracy_input,
                'Accuracy_hidden': Accuracy_hidden,
                'Accuracy_output': Accuracy_output,
                'Accuracy_acrosstime_vector_input': Accuracy_acrosstime_vector_input,
                'Accuracy_acrosstime_vector_hidden': Accuracy_acrosstime_vector_hidden,
                'Accuracy_acrosstime_vector_output': Accuracy_acrosstime_vector_output,
                'Accuracy_acrosstime_matrix_input': Accuracy_acrosstime_matrix_input,
                'Accuracy_acrosstime_matrix_hidden': Accuracy_acrosstime_matrix_hidden,
                'Accuracy_acrosstime_matrix_output': Accuracy_acrosstime_matrix_output,

                'Accuracy_syn_input': Accuracy_syn_input,
                'Accuracy_syn_hidden': Accuracy_syn_hidden,
                'Accuracy_syn_output': Accuracy_syn_output,
                'Accuracy_acrosstime_vector_syn_input': Accuracy_acrosstime_vector_syn_input,
                'Accuracy_acrosstime_vector_syn_hidden': Accuracy_acrosstime_vector_syn_hidden,
                'Accuracy_acrosstime_vector_syn_output': Accuracy_acrosstime_vector_syn_output,
                'Accuracy_acrosstime_matrix_syn_input': Accuracy_acrosstime_matrix_syn_input,
                'Accuracy_acrosstime_matrix_syn_hidden': Accuracy_acrosstime_matrix_syn_hidden,
                'Accuracy_acrosstime_matrix_syn_output': Accuracy_acrosstime_matrix_syn_output
                }
pickle.dump(svm_results, open(savedir + svmdir + '/svm_results.pkl', 'wb'))
# model = pickle.load(open(savedir + svmdir + '/svm_results.pkl', 'rb'))