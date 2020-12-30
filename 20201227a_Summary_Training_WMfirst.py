import pickle
import os
from det_rnn import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

iModel = 2

iteration = 2000
BatchSize = 200

savedir = os.path.dirname(os.path.realpath(__file__)) + '/savedir/WMfirst/iModel' + str(iModel)

if not os.path.isdir(savedir + '/Training/Iter' + str(iteration)):
    os.makedirs(savedir + '/Training/Iter' + str(iteration))

modelname = '/Iter' + str(iteration) + '.pkl'
fn = savedir + modelname
model = pickle.load(open(fn, 'rb'))

par['design'].update({'iti'     : (0, 1.5),
                      'stim'    : (1.5,3.0),
                      'delay'   : (3.0,3.0),
                      'estim'   : (3.0,4.5)})
par['batch_size'] = BatchSize
par = update_parameters(par)

h = model['h'][-1].numpy().astype('float32')
w_rnn = model['w_rnn'][-1].numpy().astype('float32')
b_rnn = model['b_rnn'][-1].numpy().astype('float32')
b_out = model['b_out'][-1].numpy().astype('float32')
w_out = model['w_out'][-1].numpy().astype('float32')
w_in = model['w_in'][-1].numpy().astype('float32')

var_dict = {}
var_dict['h'] = h
var_dict['w_rnn'] = w_rnn
var_dict['w_out'] = w_out
var_dict['b_out'] = b_out
var_dict['w_in'] = w_in
var_dict['b_rnn'] = b_rnn

## plot loss

fig = plt.figure(figsize=(15, 10), dpi=80)
plt.rcParams.update({'font.size': 25})
plt.plot(model['loss'], color='k', label='total loss')
plt.plot(model['loss_orient'], color='m', label='estimate loss')
plt.plot(np.asarray(model['spike_loss'])*par['spike_cost'], color='b', label='spike loss')
plt.legend()
plt.xlabel('iteration')
plt.ylabel('loss')
plt.ylim(0, min(model['loss']).numpy()*5)
plt.grid()
plt.savefig(savedir + '/Training/Iter' + str(iteration) + '/loss.png', bbox_inches='tight')

## plot error

fig = plt.figure(figsize=(15, 10), dpi=80)
plt.rcParams.update({'font.size': 25})
plt.plot(model['error'], color='k', label='total loss')
plt.legend()
plt.xlabel('iteration')
plt.ylabel('error')
plt.savefig(savedir + '/Training/Iter' + str(iteration) + '/error.png', bbox_inches='tight')

## load stimulus for simulation result

stimulus = Stimulus(par)
trial_info = stimulus.generate_trial()
#
fig, axes = plt.subplots(3,1, figsize=(10,8))
TEST_TRIAL = np.random.randint(stimulus.batch_size)
a0 = axes[0].imshow(trial_info['neural_input'][:,TEST_TRIAL,:].T, aspect='auto'); axes[0].set_title("Neural Input"); fig.colorbar(a0, ax=axes[0])
a1 = axes[1].imshow(trial_info['desired_output'][:,TEST_TRIAL,:].T, aspect='auto'); axes[1].set_title("Desired Output"); fig.colorbar(a1, ax=axes[1])
a2 = axes[2].imshow(trial_info['mask'][:,TEST_TRIAL,:].T, aspect='auto'); axes[2].set_title("Training Mask"); fig.colorbar(a2, ax=axes[2]) # a bug here
fig.tight_layout(pad=2.0)
plt.show()

in_data = trial_info['neural_input'].astype('float32')
out_target = trial_info['desired_output']
mask_train = trial_info['mask']
batch_size = par['batch_size']
syn_x_init = par['syn_x_init']
syn_u_init = par['syn_u_init']
# syn_x_init_in = par['syn_x_init_input']
# syn_u_init_in = par['syn_u_init_input']

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
                                            + tf.nn.relu(var_dict['b_rnn']))
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

h, output, syn_x, syn_u, w_rnn \
    = run_model(in_data, var_dict, syn_x_init, syn_u_init)

starget = tf.reduce_sum(out_target, axis=2)
starget = tf.expand_dims(starget, axis=2)
ntarget = out_target / tf.repeat(starget, par['n_output'], axis=2)
cenoutput = tf.nn.softmax(output, axis=2)

ivmin = 0
ivmax = 0.1

fig = plt.figure(figsize=(10, 8), dpi=80)

for i in range(30):

    plt.clf()
    iT = np.random.randint(batch_size)
    plt.subplot(221)
    a = output[:,iT,:].numpy()
    plt.imshow(a.T,aspect='auto')
    plt.colorbar()

    cenoutput = tf.exp(tf.nn.log_softmax(output, axis=2))
    cenoutput = cenoutput.numpy()

    plt.subplot(222)
    a = cenoutput[:, iT, :]
    plt.imshow(a.T, aspect='auto', vmin=ivmin, vmax=ivmax)
    plt.colorbar()

    plt.subplot(223)
    a = out_target[:, iT, :]
    plt.imshow(a.T, aspect='auto')
    plt.colorbar()
    #
    plt.subplot(224)
    a = ntarget[:, iT, :].numpy()
    plt.imshow(a.T, aspect='auto', vmin=ivmin, vmax=ivmax)
    plt.colorbar()

    plt.savefig(savedir + '/Training/Iter' + str(iteration) + '/' + str(i) + '.png', bbox_inches='tight')
