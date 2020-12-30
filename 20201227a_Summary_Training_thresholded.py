import pickle
import os
from det_rnn import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

thresholds = np.arange(90, 100, 2)
nthre = len(thresholds)

iModel = 5

iteration = 10000
BatchSize = 100

savedir     = os.path.dirname(os.path.realpath(__file__)) + '/savedir/iModel' + str(iModel)
modelname   = '/Iter' + str(iteration) + '.pkl'
fn          = savedir + modelname
model       = pickle.load(open(fn, 'rb'))

par['batch_size'] = BatchSize
par = update_parameters(par)

h = model['h'][-1].numpy().astype('float32')
w_rnn0 = model['w_rnn'][-1].numpy().astype('float32')
b_rnn = model['b_rnn'][-1].numpy().astype('float32')
b_out = model['b_out'][-1].numpy().astype('float32')
w_out0 = model['w_out'][-1].numpy().astype('float32')
w_in0 = model['w_in'][-1].numpy().astype('float32')

w_rnn0[w_rnn0<0]    = 0
w_out0[w_out0<0]    = 0
w_in0[w_in0<0]      = 0

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
    syn_x += (par['alpha_std'] * (1 - syn_x) - par['dt'] / 1000 * syn_u * syn_x * h)  # what is alpha_std???
    syn_u += (par['alpha_stf'] * (par['U'] - syn_u) + par['dt'] / 1000 * par['U'] * (1 - syn_u) * h)

    syn_x = tf.minimum(np.float32(1), tf.nn.relu(syn_x))
    syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
    h_post = syn_u * syn_x * h

    noise_rnn = tf.sqrt(2 * par['alpha_neuron']) * par['noise_rnn_sd']
    h = tf.nn.relu((1 - par['alpha_neuron']) * h
                   + par['alpha_neuron'] * (h_post @ w_rnn
                                            + rnn_input @ tf.nn.relu(var_dict['w_in'])
                                            + var_dict['b_rnn'])
                   + tf.random.normal(h.shape, 0, noise_rnn, dtype=tf.float32))
    return h, syn_x, syn_u

def run_model(in_data, var_dict, syn_x_init, syn_u_init):
    self_h = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    self_syn_x = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    self_syn_u = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    self_output = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    h = tf.ones((par['batch_size'], 1)) @ var_dict['h']
    syn_x = syn_x_init
    syn_u = syn_u_init
    w_rnn = par['EI_mask'] @ tf.nn.relu(var_dict['w_rnn'])

    for it in range(par['n_timesteps']):
        rnn_input = in_data[it, :, :]
        h, syn_x, syn_u = rnn_cell(rnn_input, h, syn_x, syn_u, w_rnn)

        self_h = self_h.write(it, h)
        self_syn_x = self_syn_x.write(it, syn_x)
        self_syn_u = self_syn_u.write(it, syn_u)
        self_output = self_output.write(it, h @ tf.nn.relu(var_dict['w_out']) + tf.nn.relu(var_dict['b_out']))

    self_h = self_h.stack()
    self_syn_x = self_syn_x.stack()
    self_syn_u = self_syn_u.stack()
    self_output = self_output.stack()
    return self_h, self_output, self_syn_x, self_syn_u, w_rnn

var_dict            = {}
var_dict['h']       = h
var_dict['b_out']   = b_out
var_dict['b_rnn']   = b_rnn

cThre = 0
nThre = len(thresholds)**3
expectederror = np.zeros((nThre, 4))
for ithreshold_rnn in thresholds:
    for ithreshold_out in thresholds:
        for ithreshold_in in thresholds:
            isavedir = savedir + '/Training/threshold/rnn' \
                       + str(ithreshold_rnn) + 'out' + str(ithreshold_out) + 'in' + str(ithreshold_in) + '/Iter' + str(iteration)

            # if not os.path.isdir(isavedir):
            #     os.makedirs(isavedir)

            iw_rnn  = w_rnn0.copy()
            iw_out  = w_out0.copy()
            iw_in   = w_in0.copy()

            iw_rnn_thre = np.percentile(np.reshape(iw_rnn, -1), ithreshold_rnn)
            iw_out_thre = np.percentile(np.reshape(iw_out, -1), ithreshold_out)
            iw_in_thre = np.percentile(np.reshape(iw_in, -1), ithreshold_in)

            iw_rnn[iw_rnn < iw_rnn_thre] = 0
            iw_out[iw_out < iw_out_thre] = 0
            iw_in[iw_in < iw_in_thre] = 0

            var_dict['w_rnn'] = iw_rnn
            var_dict['w_out'] = iw_out
            var_dict['w_in'] = iw_in

            h, output, syn_x, syn_u, w_rnn \
                = run_model(in_data, var_dict, syn_x_init, syn_u_init)

            starget = np.sum(out_target, axis=2)
            starget = np.expand_dims(starget, axis=2)
            ntarget = out_target / np.repeat(starget, par['n_output'], axis=2)

            noutput = tf.exp(tf.nn.log_softmax(output, axis=2))
            noutput = noutput.numpy()

            batch_expectederror = np.zeros((BatchSize))
            for ibatch in range(BatchSize):
                inoutput = np.mean(noutput[450:, ibatch, 1:], axis=0)/np.sum(np.mean(noutput[450:, ibatch, 1:], axis=0))
                ipreori = np.argmax(inoutput)*180/24
                itargetori = trial_info['stimulus_ori'][ibatch]*180/24
                ierror = ipreori - itargetori
                if ierror > 90:
                    ierror = ierror - 180
                elif ierror < -90:
                    ierror = ierror + 180
                batch_expectederror[ibatch] = abs(ierror)
            expectederror[cThre, :] = [ithreshold_in, ithreshold_rnn, ithreshold_out, np.mean(batch_expectederror)]

            print(str(cThre+1) + '/' + str(nThre))
            cThre += 1

            # ivmin = 0
            # ivmax = 0.1
            # fig = plt.figure(figsize=(10, 8), dpi=80)
            # for i in range(10):
            #
            #     plt.clf()
            #     iT = np.random.randint(batch_size)
            #     plt.subplot(221)
            #     a = output[:,iT,:].numpy()
            #     plt.imshow(a.T,aspect='auto')
            #     plt.colorbar()
            #
            #     plt.subplot(222)
            #     a = noutput[:, iT, :]
            #     plt.imshow(a.T, aspect='auto', vmin=ivmin, vmax=ivmax)
            #     plt.colorbar()
            #
            #     plt.subplot(223)
            #     a = out_target[:, iT, :]
            #     plt.imshow(a.T, aspect='auto')
            #     plt.colorbar()
            #     #
            #     plt.subplot(224)
            #     a = ntarget[:, iT, :]
            #     plt.imshow(a.T, aspect='auto', vmin=ivmin, vmax=ivmax)
            #     plt.colorbar()
            #
            #     plt.savefig(isavedir + '/' + str(i) + '.png', bbox_inches='tight')

isavedir = savedir + '/Training/Iter' + str(iteration) + '/threshold'
if not os.path.isdir(isavedir):
    os.makedirs(isavedir)

for ithreshold_rnn in thresholds:

    ierror = np.zeros((nthre, nthre))
    cout = 0
    for ithreshold_out in thresholds:
        cin = 0
        for ithreshold_in in thresholds:
            iInd = (expectederror[:, 0] == ithreshold_in)*(expectederror[:, 1] == ithreshold_rnn)*(expectederror[:, 2] == ithreshold_out)
            ierror[cout, cin] = expectederror[iInd, 3]
            cin += 1
        cout += 1

    plt.clf()
    plt.imshow(ierror, vmax=45, vmin=0)
    plt.colorbar()
    plt.xticks((np.arange(0, nthre)), thresholds)
    plt.yticks((np.arange(0, nthre)), thresholds)
    plt.ylabel('w_out %')
    plt.xlabel('w_in %')
    plt.title('w_rnn %' + str(ithreshold_rnn))
    plt.savefig(isavedir + '/rnn' + str(ithreshold_rnn) + '.png', bbox_inches='tight')

