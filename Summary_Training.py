import pickle
import os
from det_rnn import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
iModel = 12
iteration_goal = 3000
iteration_load = 3000
n_orituned_neurons = 30
BatchSize = 50
OneHotTarget = 0
CrossEntropy = 1
dxtick = 1000 # in ms

alpha_input                 = 0.2 # Chaudhuri et al., Neuron, 2015
alpha_output                = 0.2 # Chaudhuri et al., Neuron, 2015; Motor (F1) cortex's decay is in between input and hidden

delay_initial               = 1.5
delta_delay_update          = 1.0 # if estimation errors of consecutive "N_conseq_epoch_est_error" is lower than "criterion_est_error", delay increases by "delta_delay_update"
criterion_est_error         = 8
N_conseq_epoch_est_error    = 100
goal_delay                  = 1.5
Darwin_Iter                 = 1000
Darwin_EstError             = 40

dxtick                      = 1000

par['batch_size'] = BatchSize
par['scale_gamma'] = scale_gamma

par['n_untuned_input'] = n_untuned_input
par['n_tuned_input'] = n_orituned_neurons
par['n_tuned_output'] = n_orituned_neurons
par['n_ori'] = n_orituned_neurons

savedir = os.path.dirname(os.path.realpath(__file__)) + \
          '/savedir/OneHotTarget' + str(OneHotTarget) + 'CrossEntropy' + str(CrossEntropy) + 'BatchSize' + str(BatchSize) +\
          '/nIter' + str(iteration_goal) + '/iModel' + str(
    iModel)
if not os.path.isdir(savedir + '/estimation/Iter' + str(iteration_load)):
    os.makedirs(savedir + '/estimation/Iter' + str(iteration_load))

modelname = '/Iter' + str(iteration_load) + '.pkl'
fn = savedir + modelname
model = pickle.load(open(fn, 'rb'))

par['batch_size'] = BatchSize
par = update_parameters(par)
par_model = model['parameters']

h = model['h'][-1].numpy().astype('float32')
w_rnn = model['w_rnn'][-1].numpy().astype('float32')
b_rnn = model['b_rnn'][-1].numpy().astype('float32')

in_h = model['in_h'][-1].numpy().astype('float32')
# w_in2in = model['w_in2in'][-1].numpy().astype('float32')
# w_rnn2in = model['w_rnn2in'][-1].numpy().astype('float32')
# b_in = model['b_in'][-1].numpy().astype('float32')

w_in = par['EI_input_mask'] * np.maximum(w_in, 0)
# iw_in2in = par['EI_in2in_mask'] @ np.maximum(w_in2in, 0)
# iw_rnn2in = par['EImodular_mask'] @ np.maximum(w_rnn2in, 0)
iw_rnn = par['EImodular_mask'] @ np.maximum(w_rnn, 0)

var_dict = {}
var_dict['h'] = h
var_dict['w_rnn'] = w_rnn
var_dict['b_rnn'] = b_rnn
var_dict['w_out'] = w_out
var_dict['b_out'] = b_out
var_dict['in_h'] = in_h
# var_dict['w_in2in'] = w_in2in
# var_dict['w_rnn2in'] = w_rnn2in
# var_dict['b_in'] = b_in

dxtick = dxtick/10

## plot weights

MaxVal1 = np.max([np.max(w_rnn), np.max(b_rnn)])*2/3
MaxVal1 = 2
MaxVal2 = 1

fig = plt.figure(figsize=(25, 15), dpi=80)
plt.rcParams.update({'font.size': 25})
#
# iax = plt.subplot(2, 3, 1)
# im = plt.imshow(w_in, vmin=-MaxVal1, vmax=MaxVal1)
# plt.title('w_in')
# plt.ylabel('from inputs')
# plt.xlabel('to RNN')
# cb = plt.colorbar(im, orientation="horizontal", pad=0.2)
#
# iax = plt.subplot(2, 3, 4)
# im = plt.imshow(iw_in2in, vmin=-MaxVal1, vmax=MaxVal1)
# plt.title('w_in2in')
# plt.ylabel('from inputs')
# plt.xlabel('to RNN')

iax = plt.subplot(2, 3, 2)
plt.imshow(iw_rnn, vmin=-MaxVal1, vmax=MaxVal1)
plt.title('w_rnn')
plt.ylabel('from RNN')
plt.xlabel('to RNN')
# #
# iax = plt.subplot(2, 3, 5)
# plt.imshow(iw_rnn2in, vmin=-MaxVal1, vmax=MaxVal1)
# plt.title('w_rnn')
# plt.ylabel('from RNN')
# plt.xlabel('to inputs')

iax = plt.subplot(1, 9, 7)
plt.imshow((np.ones((5,1))@b_rnn[:, np.newaxis].T).T, vmin=0, vmax=MaxVal1)
plt.title('b_rnn')
cb = plt.colorbar()

# iax = plt.subplot(2, 8, 16)
# plt.imshow(w_out, vmin=0, vmax=MaxVal2)
# plt.title('w_out')
# plt.ylabel('from RNN')
# plt.xlabel('to output')
# cb = plt.colorbar()
#
# iax = plt.subplot(16, 8, 16)
# plt.imshow(np.ones((5,1))@b_out[:, np.newaxis].T, vmin=0, vmax=MaxVal2)
# plt.title('b_out')
# cb = plt.colorbar()
plt.savefig(savedir + '/TrainingSummary_weight_nIter' + str(iteration_load) + '.png', bbox_inches='tight')

## plot loss

fig = plt.figure(figsize=(15, 10), dpi=80)
plt.rcParams.update({'font.size': 25})
plt.plot(model['loss'], color='k', label='total loss')
plt.plot(model['loss_orient'], color='m', label='estimate loss')
plt.plot(np.asarray(model['spike_loss'])*par_model['spike_cost'], color='b', label='spike loss')
plt.plot(np.asarray(model['weight_loss'])*par_model['weight_cost'], color='r', label='spike loss')
plt.legend()
plt.xlabel('iteration')
plt.ylabel('loss')
plt.ylim(0, min(model['loss']).numpy()*5)
plt.grid()
plt.savefig(savedir + '/TrainingSummary_loss_Iter' + str(iteration_load) + '.png', bbox_inches='tight')


## load stimulus for simulation result

stimulus = Stimulus(par)
trial_info = stimulus.generate_trial()
#
# fig, axes = plt.subplots(3,1, figsize=(10,8))
# TEST_TRIAL = np.random.randint(stimulus.batch_size)
# a0 = axes[0].imshow(trial_info['neural_input'][:,TEST_TRIAL,:].T, aspect='auto'); axes[0].set_title("Neural Input"); fig.colorbar(a0, ax=axes[0])
# a1 = axes[1].imshow(trial_info['desired_output'][:,TEST_TRIAL,:].T, aspect='auto'); axes[1].set_title("Desired Output"); fig.colorbar(a1, ax=axes[1])
# a2 = axes[2].imshow(trial_info['mask'][:,TEST_TRIAL,:].T, aspect='auto'); axes[2].set_title("Training Mask"); fig.colorbar(a2, ax=axes[2]) # a bug here
# fig.tight_layout(pad=2.0)
# plt.show()


in_data = trial_info['neural_input'].astype('float32')
out_target = trial_info['desired_output']
mask_train = trial_info['mask']
batch_size = par['batch_size']
syn_x_init = par['syn_x_init']
syn_u_init = par['syn_u_init']
# syn_x_init_in = par['syn_x_init_input']
# syn_u_init_in = par['syn_u_init_input']

def rnn_cell(rnn_input, h, syn_x, syn_u, w_rnn):

    rnn_input = tf.concat((rnn_input, tf.zeros((par['batch_size'], par['n_total'] - par['n_input']))), axis=1)

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

    self_h = np.zeros((par['n_timesteps'], par['batch_size'], par['n_total']), dtype=np.float32)
    self_syn_x = np.zeros((par['n_timesteps'], par['batch_size'], par['n_total']), dtype=np.float32)
    self_syn_u = np.zeros((par['n_timesteps'], par['batch_size'], par['n_total']), dtype=np.float32)
    self_output = np.zeros((par['n_timesteps'], par['batch_size'], par['n_output']), dtype=np.float32)

    self_in_h = np.zeros((par['n_timesteps'], par['batch_size'], 2*par['n_input']), dtype=np.float32)
    # self_in_syn_x = np.zeros((par['n_timesteps'], par['batch_size'], 2*par['n_input']), dtype=np.float32)
    # self_in_syn_u = np.zeros((par['n_timesteps'], par['batch_size'], 2*par['n_input']), dtype=np.float32)

    h = np.ones((par['batch_size'], 1)) @ var_dict['h']
    syn_x = syn_x_init
    syn_u = syn_u_init
    w_rnn = par['EImodular_mask'] @ np.maximum(var_dict['w_rnn'], 0)
    w_in = par['EI_input_mask'] * np.maximum(var_dict['w_in'], 0)

    in_h = np.ones((par['batch_size'], 1)) @ var_dict['in_h']
    # in_syn_x = syn_x_init_in
    # in_syn_u = syn_u_init_in
    # w_in2in = par['EI_in2in_mask'] @ np.maximum(var_dict['w_in2in'], 0)
    # w_rnn2in = par['EImodular_mask'] @ np.maximum(var_dict['w_rnn2in'], 0)

    c = 0
    for rnn_input in in_data:

        in_h = rnn_cell_input(rnn_input, in_h)

        h, syn_x, syn_u = rnn_cell(in_h, h, syn_x, syn_u, w_rnn, w_in)
        #
        self_in_h[c, :, :] = in_h
        # self_in_syn_x[c, :, :] = in_syn_x
        # self_in_syn_u[c, :, :] = in_syn_u
        self_h[c, :, :] = h
        self_syn_x[c, :, :] = syn_x
        self_syn_u[c, :, :] = syn_u
        self_output[c, :, :] = h[:, -par['n_output']:]
        c += 1

    return self_h, self_output, self_syn_x, self_syn_u, w_rnn, self_in_h

h, output, syn_x, syn_u, w_rnn, in_h \
    = run_model(in_data, var_dict, syn_x_init, syn_u_init)

##

MAP_dirs = output[par['output_rg'][par['output_rg'] > np.max(par['dead_rg'])].min():, :, 1:]
MAP_dirs = tf.nn.log_softmax(MAP_dirs, axis=2)
MAP_dirs = tf.reduce_sum(MAP_dirs, axis=0)
MAP_dirs = tf.argmax(MAP_dirs, axis=1)
stim_dirs = tf.constant(par['stim_dirs'])
MAP_dirs = tf.gather(stim_dirs, MAP_dirs)
Target_dirs = tf.gather(stim_dirs, trial_info['stimulus_ori'])
est_error = tf.reduce_mean(tf.math.acos(tf.math.cos(2 * (MAP_dirs - Target_dirs) / 180 * np.pi)) / np.pi * 180 / 2)
print('est_error=' + str(est_error.numpy()))

##

starget = np.sum(out_target, axis=2)
starget = np.expand_dims(starget, axis=2)
ntarget = out_target / np.repeat(starget, par['n_output'], axis=2)
ivmin = 0
ivmax = 0.1

if OneHotTarget is 0:
    starget = np.sum(out_target, axis=2)
    starget = np.expand_dims(starget, axis=2)
    ntarget = out_target / np.repeat(starget, par['n_output'], axis=2)
    ivmin = 0
    ivmax = 0.1
else:
    ntarget = out_target == np.max(out_target, axis=2)[:, :, None]
    ivmin = 0
    ivmax = 1

fig = plt.figure(figsize=(10, 8), dpi=80)

for i in range(30):

    plt.clf()
    iT = np.random.randint(batch_size)
    plt.subplot(221)
    a = output[:,iT,:]
    plt.imshow(a.T,aspect='auto')
    plt.colorbar()

    cenoutput = tf.exp(tf.nn.log_softmax(output, axis=2))
    cenoutput = cenoutput.numpy()

    # sout = np.sum(output, axis=2)
    # sout = np.expand_dims(sout, axis=2)
    # noutput = output / np.repeat(sout,par['n_output'],axis=2)
    # cenoutput = tf.nn.softmax(output, axis=2)
    # cenoutput = cenoutput.numpy()

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
    a = ntarget[:, iT, :]
    plt.imshow(a.T, aspect='auto', vmin=ivmin, vmax=ivmax)
    plt.colorbar()

    plt.savefig(savedir + '/estimation/delay_test' + str(delay_test) + '/Iter' + str(iteration_load) + '/' + str(i) + '.png', bbox_inches='tight')
