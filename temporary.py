import pickle
import os
from det_rnn import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from scipy import stats

iModel = 1
iteration_goal = 2000
iteration_load = 2000
n_orituned_neurons = 30
BatchSize_load = 50
BatchSize_test = 1000
dxtick = 1000 # in ms
connect_prob = 0.1
scale_w_rnn2in = 0.07

nCrossVal = 10

par['n_tuned_input'] = n_orituned_neurons
par['n_tuned_output'] = n_orituned_neurons
par['n_ori'] = n_orituned_neurons
par['connect_prob'] = connect_prob
par['scale_w_rnn2in'] = scale_w_rnn2in

# par['design'].update({'iti'     : (0, 0.5),
#                       'stim'    : (0.5,2.0),
#                       'delay'   : (2.0,4.5),
#                       'estim'   : (4.5,6.0)})

delay = par['design']['delay'][1] - par['design']['delay'][0]
savedir = os.path.dirname(os.path.realpath(__file__)) + \
           '/savedir/connect_prob' + str(connect_prob) + 'scale_w_rnn2in' + str(scale_w_rnn2in) + \
           '/nIter' + str(iteration_goal) + 'BatchSize' + str(BatchSize_load) + '/delay' + str(delay) + '/iModel' + str(iModel)

if not os.path.isdir(savedir + '/svm/'):
    os.makedirs(savedir + '/svm/')

modelname = '/Iter' + str(iteration_load) + '.pkl'
fn = savedir + modelname
model = pickle.load(open(fn, 'rb'))
w_rnn2in_sparse_mask = model['parameters']['w_rnn2in_sparse_mask'] # this sparse mask should be identical with the training and testing

par['batch_size'] = BatchSize_test
par = update_parameters(par)
par_model = model['parameters']

h = model['h'][-1].numpy().astype('float32')
w_in = model['w_in'][-1].numpy().astype('float32')
w_rnn = model['w_rnn'][-1].numpy().astype('float32')
b_rnn = model['b_rnn'][-1].numpy().astype('float32')
w_out = model['w_out'][-1].numpy().astype('float32')
w_rnn2in = model['w_rnn2in'][-1].numpy().astype('float32')
b_out = model['b_out'][-1].numpy().astype('float32')

var_dict = {}
var_dict['h'] = h
var_dict['w_in'] = w_in
var_dict['w_rnn'] = w_rnn
var_dict['b_rnn'] = b_rnn
var_dict['w_out'] = w_out
var_dict['b_out'] = b_out
var_dict['w_rnn2in'] = w_rnn2in

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

def rnn_cell(rnn_input, h, syn_x, syn_u, w_rnn, w_in):
    syn_x += (par['alpha_std'] * (1 - syn_x) - par['dt']/1000 * syn_u * syn_x * h)  # what is alpha_std???
    syn_u += (par['alpha_stf'] * (par['U'] - syn_u) + par['dt']/1000 * par['U'] * (1 - syn_u) * h)

    syn_x = tf.minimum(np.float32(1), tf.nn.relu(syn_x))
    syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
    h_post = syn_u * syn_x * h
    # h_post = h

    noise_rnn = np.sqrt(2*par['alpha_neuron'])*par['noise_rnn_sd']
    h = tf.nn.relu((1 - par['alpha_neuron']) * h
         + par['alpha_neuron'] * (h_post @ w_rnn
                                  + rnn_input @ w_in
                                  + var_dict['b_rnn'])
         + tf.random.normal(h.shape, 0, noise_rnn, dtype=tf.float32))
    return h, syn_x, syn_u

def run_model(in_data, var_dict, syn_x_init, syn_u_init):

    self_h = np.zeros((par['n_timesteps'], par['batch_size'], par['n_hidden']), dtype=np.float32)
    self_syn_x = np.zeros((par['n_timesteps'], par['batch_size'], par['n_hidden']), dtype=np.float32)
    self_syn_u = np.zeros((par['n_timesteps'], par['batch_size'], par['n_hidden']), dtype=np.float32)
    self_output = np.zeros((par['n_timesteps'], par['batch_size'], par['n_output']), dtype=np.float32)
    self_input = np.zeros((par['n_timesteps'], par['batch_size'], par['n_input']), dtype=np.float32)

    h = np.ones((par['batch_size'], 1)) @ var_dict['h']
    syn_x = syn_x_init
    syn_u = syn_u_init
    w_rnn = par['EImodular_mask'] @ np.maximum(var_dict['w_rnn'], 0)
    w_in = np.maximum(var_dict['w_in'], 0)

    w_rnn2in = par['EImodular_mask'] @ tf.nn.relu(var_dict['w_rnn2in'] * model['parameters']['w_rnn2in_sparse_mask'])
    h_pre = np.float32(np.random.gamma(0.1, 0.2, size=h.shape))


    c = 0
    for rnn_input in in_data:

        rnn_input = tf.nn.relu(rnn_input + h_pre @ w_rnn2in)
        # rnn_input = tf.nn.relu((1 - par['alpha_input']) * rnn_input + par['alpha_neuron'] * (h_pre @ w_rnn2in))

        h_pre = h

        h, syn_x, syn_u = rnn_cell(rnn_input, h, syn_x, syn_u, w_rnn, w_in)
        #
        self_h[c, :, :] = h
        self_syn_x[c, :, :] = syn_x
        self_syn_u[c, :, :] = syn_u
        self_output[c, :, :] = h @ np.maximum(var_dict['w_out'], 0) + var_dict['b_out']
        self_input[c, :, :] = rnn_input
        c += 1

    return self_h, self_output, self_syn_x, self_syn_u, w_rnn, self_input

h, output, syn_x, syn_u, w_rnn, inputs \
    = run_model(in_data, var_dict, syn_x_init, syn_u_init)

## figure conditions

esti_on = par['design']['estim'][0]*np.array(1000/par['dt'])
stim_onoff = par['design']['stim']*np.array(1000/par['dt'])

## decoding from input units

boundCrossVal = np.linspace(0, par['batch_size'], nCrossVal + 1, dtype='int')
Accuracy_input = np.zeros(par['n_timesteps'])
for itime in range(par['n_timesteps']):
    print(itime)
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

itime_train = 200
itrain_input = inputs[itime_train, :, :]
itrain_stim = trial_info['stimulus_ori']
svm_model_linear = SVC(kernel='linear', C=1).fit(itrain_input, itrain_stim)
Accuracy_acrosstime_vector_input = np.zeros(par['n_timesteps'])
for itime in range(par['n_timesteps']):
    print(itime)
    ipredict = np.zeros((par['batch_size']))
    for iCrossVal in range(nCrossVal):
        iInd_test = np.arange(boundCrossVal[iCrossVal], boundCrossVal[iCrossVal+1])
        itest_input = inputs[itime, iInd_test, :]
        itest_stim = trial_info['stimulus_ori'][iInd_test]
        svm_predictions = svm_model_linear.predict(itest_input)

        ipredict[iInd_test] = svm_predictions
    Accuracy_acrosstime_vector_input[itime] = np.sum(ipredict == trial_info['stimulus_ori'])/par['batch_size']

fig = plt.figure(figsize=(10, 7), dpi=80)
plt.clf()
plt.plot(Accuracy_input)
plt.plot(Accuracy_acrosstime_vector_input)
plt.rcParams.update({'font.size': 20})
plt.plot(np.array([0.0, par['n_timesteps']*1.0]), np.array([1.0, 1.0])/(par['n_tuned_input']*1.0), 'k--')
plt.plot(np.array([0.0, par['n_timesteps']*1.0]), np.array([0, 0]), 'k-')
plt.plot(np.array([1, 1])*esti_on, np.array([0, 1]),'k--')
plt.plot(np.array([1, 1])*stim_onoff[0], np.array([0, 1]),'k--')
plt.plot(np.array([1, 1])*stim_onoff[1], np.array([0, 1]),'k--')
plt.plot(np.array([1, 1])*itime_train, np.array([0, 1]),'r--')
plt.xlabel('time point')
plt.ylabel('classification accuracy')
plt.savefig(savedir + '/svm/Accuracy_input' + str(iteration_load) + '.png', bbox_inches='tight')

## decoding from hidden units

Accuracy_hidden = np.zeros(par['n_timesteps'])
for itime in range(par['n_timesteps']):
    print(itime)
    ipredict = np.zeros((par['batch_size']))
    for iCrossVal in range(nCrossVal):
        iInd_test = np.arange(boundCrossVal[iCrossVal], boundCrossVal[iCrossVal+1])
        iInd_train = np.arange(0, par['batch_size'])
        iInd_train = np.delete(iInd_train, iInd_test, axis=0)

        itrain_h = h[itime, iInd_train, :]
        itrain_stim = trial_info['stimulus_ori'][iInd_train]
        itest_h = h[itime, iInd_test, :]
        itest_stim = trial_info['stimulus_ori'][iInd_test]

        svm_model_linear = SVC(kernel='linear', C=1).fit(itrain_h, itrain_stim)
        svm_predictions = svm_model_linear.predict(itest_h)

        ipredict[iInd_test] = svm_predictions
    Accuracy_hidden[itime] = np.sum(ipredict == trial_info['stimulus_ori'])/par['batch_size']

itime_train = 200
itrain_h = h[itime_train, :, :]
itrain_stim = trial_info['stimulus_ori']
svm_model_linear = SVC(kernel='linear', C=1).fit(itrain_h, itrain_stim)
Accuracy_acrosstime_vector_hidden = np.zeros(par['n_timesteps'])
for itime in range(par['n_timesteps']):
    print(itime)
    ipredict = np.zeros((par['batch_size']))
    for iCrossVal in range(nCrossVal):
        iInd_test = np.arange(boundCrossVal[iCrossVal], boundCrossVal[iCrossVal+1])
        itest_h = h[itime, iInd_test, :]
        itest_stim = trial_info['stimulus_ori'][iInd_test]
        svm_predictions = svm_model_linear.predict(itest_h)

        ipredict[iInd_test] = svm_predictions
    Accuracy_acrosstime_vector_hidden[itime] = np.sum(ipredict == trial_info['stimulus_ori'])/par['batch_size']

fig = plt.figure(figsize=(10, 7), dpi=80)
plt.clf()
plt.plot(Accuracy_hidden)
plt.plot(Accuracy_acrosstime_vector_hidden)
plt.rcParams.update({'font.size': 20})
plt.plot(np.array([0.0, par['n_timesteps']*1.0]), np.array([1.0, 1.0])/(par['n_tuned_input']*1.0), 'k--')
plt.plot(np.array([0.0, par['n_timesteps']*1.0]), np.array([0, 0]), 'k-')
plt.plot(np.array([1, 1])*esti_on, np.array([0, 1]),'k--')
plt.plot(np.array([1, 1])*stim_onoff[0], np.array([0, 1]),'k--')
plt.plot(np.array([1, 1])*stim_onoff[1], np.array([0, 1]),'k--')
plt.plot(np.array([1, 1])*itime_train, np.array([0, 1]),'r--')
plt.xlabel('time point')
plt.ylabel('classification accuracy')
plt.savefig(savedir + '/svm/Accuracy_hidden' + str(iteration_load) + '.png', bbox_inches='tight')

##

Accuracy_acrosstime_matrix_hidden = np.zeros((par['n_timesteps'], par['n_timesteps']))
for itime_train in range(par['n_timesteps']):
    print(itime_train)
    itrain_h = h[itime_train, :, :]
    svm_model_linear = SVC(kernel='linear', C=1).fit(itrain_h, trial_info['stimulus_ori'])
    for itime_test in range(par['n_timesteps']):
        itest_h = h[itime_test, :, :]
        svm_predictions = svm_model_linear.predict(itest_h)
        iaccuracy = np.sum(svm_predictions == trial_info['stimulus_ori']) / par['batch_size']
        Accuracy_acrosstime_matrix_hidden[itime_train, itime_test] = iaccuracy

fig = plt.figure(figsize=(13, 13), dpi=80)
plt.rcParams.update({'font.size': 20})
plt.imshow(Accuracy_acrosstime_matrix_hidden)
plt.plot(np.array([1, 1])*esti_on, np.array([0, 1])*par['n_timesteps'], 'k--')
plt.plot(np.array([0, 1])*par['n_timesteps'], np.array([1, 1])*esti_on, 'k--')
plt.plot(np.array([1, 1])*stim_onoff[0], np.array([0, 1])*par['n_timesteps'], 'k--')
plt.plot(np.array([0, 1])*par['n_timesteps'], np.array([1, 1])*stim_onoff[0], 'k--')
plt.plot(np.array([1, 1])*stim_onoff[1], np.array([0, 1])*par['n_timesteps'], 'k--')
plt.plot(np.array([0, 1])*par['n_timesteps'], np.array([1, 1])*stim_onoff[1], 'k--')
plt.xlabel('time point')
plt.ylabel('time point')
plt.xlim([0, par['n_timesteps']])
plt.ylim([0, par['n_timesteps']])
plt.colorbar()
plt.savefig(savedir + '/svm/Accuracy_hidden_matrix' + str(iteration_load) + '.png', bbox_inches='tight')

Accuracy_acrosstime_matrix_input = np.zeros((par['n_timesteps'], par['n_timesteps']))
for itime_train in range(par['n_timesteps']):
    print(itime_train)
    itrain_input = inputs[itime_train, :, :]
    svm_model_linear = SVC(kernel='linear', C=1).fit(itrain_input, trial_info['stimulus_ori'])
    for itime_test in range(par['n_timesteps']):
        itest_input = inputs[itime_test, :, :]
        svm_predictions = svm_model_linear.predict(itest_input)
        iaccuracy = np.sum(svm_predictions == trial_info['stimulus_ori'])/par['batch_size']
        Accuracy_acrosstime_matrix_input[itime_train, itime_test] = iaccuracy

fig = plt.figure(figsize=(13, 13), dpi=80)
plt.rcParams.update({'font.size': 20})
plt.imshow(Accuracy_acrosstime_matrix_input)
plt.plot(np.array([1, 1])*esti_on, np.array([0, 1])*par['n_timesteps'], 'k--')
plt.plot(np.array([0, 1])*par['n_timesteps'], np.array([1, 1])*esti_on, 'k--')
plt.plot(np.array([1, 1])*stim_onoff[0], np.array([0, 1])*par['n_timesteps'], 'k--')
plt.plot(np.array([0, 1])*par['n_timesteps'], np.array([1, 1])*stim_onoff[0], 'k--')
plt.plot(np.array([1, 1])*stim_onoff[1], np.array([0, 1])*par['n_timesteps'], 'k--')
plt.plot(np.array([0, 1])*par['n_timesteps'], np.array([1, 1])*stim_onoff[1], 'k--')
plt.xlabel('time point')
plt.ylabel('time point')
plt.xlim([0, par['n_timesteps']])
plt.ylim([0, par['n_timesteps']])
plt.colorbar()
plt.savefig(savedir + '/svm/Accuracy_input_matrix' + str(iteration_load) + '.png', bbox_inches='tight')

svm_results = {'Accuracy_input': Accuracy_input,
           'Accuracy_acrosstime_vector_input': Accuracy_acrosstime_vector_input,
           'Accuracy_hidden': Accuracy_hidden,
           'Accuracy_acrosstime_vector_hidden': Accuracy_acrosstime_vector_hidden,
           'Accuracy_acrosstime_matrix_input': Accuracy_acrosstime_matrix_input,
           'Accuracy_acrosstime_matrix_hidden': Accuracy_acrosstime_matrix_hidden
           }
pickle.dump(svm_results, open(savedir + '/svm/svm_results.pkl', 'wb'))