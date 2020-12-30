import pickle
import os
from det_rnn import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.stats as stat

iModel = 5

iteration = 10000
BatchSize = 200

savedir = os.path.dirname(os.path.realpath(__file__)) + '/savedir/iModel' + str(iModel)

if not os.path.isdir(savedir + '/Training/Iter' + str(iteration)):
    os.makedirs(savedir + '/Training/Iter' + str(iteration))

modelname = '/Iter' + str(iteration) + '.pkl'
fn = savedir + modelname
model = pickle.load(open(fn, 'rb'))

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

## load stimulus for simulation result

stimulus = Stimulus(par)
trial_info = stimulus.generate_trial()
#

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

h, output, syn_x, syn_u, w_rnn \
    = run_model(in_data, var_dict, syn_x_init, syn_u_init)

##

h = h.numpy()
ih = h[:,0,:]
mh = np.mean(h, axis=1)
mh[0:20, :] = np.nan
zh = stat.zscore(mh, axis=0, nan_policy='omit')

plt.clf()
fig, axes = plt.subplots(3,1, figsize=(6,8))
TEST_TRIAL = np.random.randint(stimulus.batch_size)
a0 = axes[0].imshow(trial_info['neural_input'][:,TEST_TRIAL,:].T, aspect='auto'); axes[0].set_title("Neural Input"); fig.colorbar(a0, ax=axes[0])
a1 = axes[1].imshow(trial_info['desired_output'][:,TEST_TRIAL,:].T, aspect='auto'); axes[1].set_title("Desired Output"); fig.colorbar(a1, ax=axes[1])
a2 = axes[2].imshow(ih.T, aspect='auto', interpolation='none'); axes[2].set_title("h"); fig.colorbar(a2, ax=axes[2]) # a bug here
fig.tight_layout(pad=2.0)
plt.show()


plt.clf()
plt.imshow(zh.T, aspect='auto', vmin=-3, vmax=3)
plt.colorbar()

t_resting   = range(0, 150)
t_encoding  = range(150, 300)
t_maintain  = range(300, 450)
t_probe     = range(450, 600)

h_resting   = mh[t_resting, :]
h_encoding  = mh[t_encoding, :]
h_maintain  = mh[t_maintain, :]
h_probe     = mh[t_probe, :]


t_er, p_er = stat.ttest_rel(h_encoding, h_resting, axis=0)
t_mr, p_mr = stat.ttest_rel(h_maintain, h_resting, axis=0)
t_pr, p_pr = stat.ttest_rel(h_probe, h_resting, axis=0)
t_em, p_em = stat.ttest_rel(h_encoding, h_maintain, axis=0)
t_mp, p_mp = stat.ttest_rel(h_maintain, h_probe, axis=0)
t_pe, p_pe = stat.ttest_rel(h_probe, h_encoding, axis=0)

Ind_encoding    = (t_em>0)*(p_em<0.05)*(t_pe<0)*(p_pe<0.05)*(t_er>0)*(p_er<0.05)
Ind_maintain    = (t_em<0)*(p_em<0.05)*(t_mp>0)*(p_mp<0.05)*(t_mr>0)*(p_mr<0.05)
Ind_probe       = (t_mp<0)*(p_mp<0.05)*(t_pe>0)*(p_pe<0.05)*(t_pr>0)*(p_pr<0.05)


fig, axes = plt.subplots(3,1, figsize=(6,8))
a0 = axes[0].plot(mh[:, Ind_encoding]); axes[0].set_title("encode")
a1 = axes[1].plot(mh[:, Ind_maintain]); axes[1].set_title("maintain")
a2 = axes[2].plot(mh[:, Ind_probe]); axes[2].set_title("probe")
fig.tight_layout(pad=2.0)
plt.show()
