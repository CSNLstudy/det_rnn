import pickle
import os
from det_rnn import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time

nModel = np.array([12, 13])
iteration = 3000
stimulus = Stimulus()
OneHotTarget = 0
CrossEntropy = 1

n_orituned_neurons = 30
BatchSize = 50
orientation_cost = 1

par['n_tuned_input'] = n_orituned_neurons
par['n_tuned_output'] = n_orituned_neurons
par['n_ori'] = n_orituned_neurons
par['batch_size'] = BatchSize
par['orientation_cost'] = orientation_cost

# par['design'].update({'iti'     : (0, 0.5),
#                       'stim'    : (0.5,2.0),
#                       'delay'   : (2.0,4.5),
#                       'estim'   : (4.5,6.0)})

# par['design'].update({'iti'     : (0, 0.5),
#                       'stim'    : (0.5,2.0),
#                       'delay'   : (2.0,8.0),
#                       'estim'   : (8.0,9.5)})

par = update_parameters(par)
stimulus = Stimulus(par)

##

# trial_info = stimulus.generate_trial()
# in_data = tf.constant(trial_info['neural_input'].astype('float32')).numpy()
# out_target = tf.constant(trial_info['desired_output']).numpy()
# mask_train = tf.constant(trial_info['mask']).numpy()
# #
# plt.close()
# fig, axes = plt.subplots(4,1, figsize=(13,8))
# TEST_TRIAL = np.random.randint(stimulus.batch_size)
# a0 = axes[0].imshow(in_data[:,TEST_TRIAL,:].T, aspect='auto'); axes[0].set_title("Neural Input"); fig.colorbar(a0, ax=axes[0])
# a1 = axes[1].imshow(out_target[:,TEST_TRIAL,:].T, aspect='auto'); axes[1].set_title("Desired Output"); fig.colorbar(a1, ax=axes[1])
# a2 = axes[2].imshow(mask_train[:,TEST_TRIAL,:].T, aspect='auto'); axes[2].set_title("Training Mask"); fig.colorbar(a2, ax=axes[2]) # a bug here
# a0 = axes[3].imshow(in_data[:,TEST_TRIAL,:par['n_input']].T - in_data[:,TEST_TRIAL,par['n_input']:].T, aspect='auto'); axes[3].set_title("Neural Input Difference"); fig.colorbar(a0, ax=axes[3])
# fig.tight_layout(pad=2.0)
# plt.show()

##

def initialize_parameters(iModel, par):

    ipar = update_parameters(par)

    ivar_dict = {}
    ivar_list = []
    for k, v in ipar.items():
        if k[-1] == '0':
            name = k[:-1]
            ivar_dict[name] = tf.Variable(ipar[k], name)
            ivar_list.append(ivar_dict[name])
    isyn_x_init = tf.constant(ipar['syn_x_init'])
    isyn_u_init = tf.constant(ipar['syn_u_init'])
    isyn_x_init_in = tf.constant(ipar['syn_x_init_input'])
    isyn_u_init_in = tf.constant(ipar['syn_u_init_input'])
    ibatch_size = ipar['batch_size']

    isavedir = os.path.dirname(os.path.realpath(__file__)) + \
               '/savedir/OneHotTarget' + str(OneHotTarget) + 'CrossEntropy' + str(CrossEntropy) + 'BatchSize' + str(BatchSize) +\
               '/nIter' + str(iteration) + '/iModel' + str(iModel)
    if not os.path.isdir(isavedir):
        os.makedirs(isavedir)

    return ipar, ivar_dict, ivar_list, isyn_x_init, isyn_u_init, ibatch_size, isavedir, isyn_x_init_in, isyn_u_init_in

def rnn_cell_input(rnn_input, in_h):
    # in_syn_x += (par['alpha_std_input'] * (1 - in_syn_x) - par['dt']/1000 * in_syn_u * in_syn_x * in_h)  # what is alpha_std???
    # in_syn_u += (par['alpha_stf_input'] * (par['U_input'] - in_syn_u) + par['dt']/1000 * par['U_input'] * (1 - in_syn_u) * in_h)
    #
    # in_syn_x = tf.minimum(np.float32(1), tf.nn.relu(in_syn_x))
    # in_syn_u = tf.minimum(np.float32(1), tf.nn.relu(in_syn_u))
    # in_h_post = in_syn_u * in_syn_x * in_h
    # h_post = h

    noise_rnn = np.sqrt(2*par['alpha_input'])*par['noise_rnn_sd']
    in_h = tf.nn.relu((1 - par['alpha_input']) * in_h
         + par['alpha_input'] * (rnn_input
                                  # + in_h_post @ w_in2in
                                  # + h @ w_rnn2in
                                  # + var_dict['b_in']
                                  )
         + tf.random.normal(in_h.shape, 0, noise_rnn, dtype=tf.float32)
                      )
    return in_h

def rnn_cell(rnn_input, h, syn_x, syn_u, w_rnn, w_in):
    syn_x += (par['alpha_std'] * (1 - syn_x) - par['dt']/1000 * syn_u * syn_x * h)  # what is alpha_std???
    syn_u += (par['alpha_stf'] * (par['U'] - syn_u) + par['dt']/1000 * par['U'] * (1 - syn_u) * h)

    syn_x = tf.minimum(np.float32(1), tf.nn.relu(syn_x))
    syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
    h_post = syn_u * syn_x * h

    noise_rnn = np.sqrt(2*par['alpha_neuron'])*par['noise_rnn_sd']
    h = tf.nn.relu((1 - par['alpha_neuron']) * h
         + par['alpha_neuron'] * (h_post @ w_rnn
                                  + rnn_input @ w_in
                                  + var_dict['b_rnn'])
         + tf.random.normal(h.shape, 0, noise_rnn, dtype=tf.float32))
    return h, syn_x, syn_u

def run_model(in_data, syn_x_init, syn_u_init):
    self_in_h = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    # self_in_syn_x = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    # self_in_syn_u = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    self_h = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    self_syn_x = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    self_syn_u = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    self_output = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    h = np.ones((par['batch_size'], 1)) @ var_dict['h']
    syn_x = syn_x_init
    syn_u = syn_u_init
    w_rnn = par['EImodular_mask'] @ tf.nn.relu(var_dict['w_rnn'])
    w_in = par['EI_input_mask'] * tf.nn.relu(var_dict['w_in'])

    in_h = np.ones((par['batch_size'], 1)) @ var_dict['in_h']
    # in_syn_x = syn_x_init_in
    # in_syn_u = syn_u_init_in
    # w_in2in = par['EI_in2in_mask'] @ tf.nn.relu(var_dict['w_in2in'])
    # w_rnn2in = par['EImodular_mask'] @ tf.nn.relu(var_dict['w_rnn2in'])

    c = 0
    for rnn_input in in_data:

        in_h = rnn_cell_input(rnn_input, in_h)

        h, syn_x, syn_u = rnn_cell(in_h, h, syn_x, syn_u, w_rnn, w_in)

        self_in_h = self_in_h.write(c, in_h)
        # self_in_syn_x = self_in_syn_x.write(c, in_syn_x)
        # self_in_syn_u = self_in_syn_u.write(c, in_syn_u)
        self_h = self_h.write(c, h)
        self_syn_x = self_syn_x.write(c, syn_x)
        self_syn_u = self_syn_u.write(c, syn_u)
        self_output = self_output.write(c, h @ tf.nn.relu(var_dict['w_out']) + tf.nn.relu(var_dict['b_out']))
        c += 1
    #
    self_in_h = self_in_h.stack()
    # self_in_syn_x = self_in_syn_x.stack()
    # self_in_syn_u = self_in_syn_u.stack()
    self_h = self_h.stack()
    self_syn_x = self_syn_x.stack()
    self_syn_u = self_syn_u.stack()
    self_output = self_output.stack()

    return self_h, self_output, self_syn_x, self_syn_u, w_rnn, self_in_h

def calc_loss(syn_x_init, syn_u_init, in_data, out_target, mask_train):

    h, output, _, _, w_rnn, in_h = run_model(in_data, syn_x_init, syn_u_init)

    if OneHotTarget is 0:
        starget = tf.reduce_sum(out_target, axis=2)
        starget = tf.expand_dims(starget, axis=2)
        ntarget = out_target / tf.repeat(starget, par['n_output'], axis=2)
    else:
        ntarget = tf.cast(out_target == tf.reduce_max(out_target, axis=2)[:, :, None], dtype=tf.float32)

    if CrossEntropy is 0:
        noutput = tf.nn.softmax(output, axis=2)
        loss_orient = tf.reduce_sum(mask_train * (ntarget - noutput) ** 2)
        loss_orient_print = tf.reduce_mean(mask_train * (ntarget - noutput) ** 2)
    else:
        noutput = tf.nn.log_softmax(output, axis=2)
        CE = -ntarget*noutput
        loss_orient = tf.reduce_sum(mask_train*CE)
        loss_orient_print = tf.reduce_mean(mask_train*CE)

    n = 2
    spike_loss = tf.reduce_sum(h**2) + tf.reduce_sum(in_h**2)
    weight_loss = tf.reduce_sum(tf.nn.relu(w_rnn) ** n)
    loss = par['orientation_cost'] * loss_orient + par['spike_cost'] * spike_loss + par['weight_cost'] * weight_loss
    return loss, loss_orient, spike_loss, weight_loss, loss_orient_print

def append_model_performance(model_performance, loss, loss_orient, spike_loss, itime, var_dict):
    model_performance['time'].append(itime)
    model_performance['loss'].append(loss)
    model_performance['loss_orient'].append(loss_orient)
    model_performance['spike_loss'].append(spike_loss)
    model_performance['iteration'].append(iteration)

    model_performance['w_in'].append(var_dict['w_in'])
    model_performance['w_rnn'].append(var_dict['w_rnn'])
    model_performance['b_rnn'].append(var_dict['b_rnn'])
    model_performance['w_out'].append(var_dict['w_out'])
    model_performance['b_out'].append(var_dict['b_out'])
    model_performance['h'].append(var_dict['h'])

    model_performance['in_h'].append(var_dict['in_h'])
    # model_performance['w_in2in'].append(var_dict['w_in2in'])
    # model_performance['w_rnn2in'].append(var_dict['w_rnn2in'])
    # model_performance['b_in'].append(var_dict['b_in'])

    return model_performance

def save_results(model_performance, par, iteration):
    results = {'parameters': par, 'iteration': iteration}
    for k, v in model_performance.items():
        results[k] = v
    pickle.dump(results, open(savedir + '/Iter' + str(iteration) + '.pkl', 'wb'))
    print('Model results saved in', savedir, '/Iter', str(iteration), '.pkl')

t0 = time.time()
iModel = 0
for iModel in range(nModel[0], nModel[1]):

    par, var_dict, var_list, syn_x_init, syn_u_init, batch_size, savedir, syn_x_init_in, syn_u_init_in = initialize_parameters(iModel, par)
    opt = tf.optimizers.Adam(learning_rate=par['learning_rate'])
    model_performance = {'loss': [], 'loss_orient': [], 'spike_loss': [], 'iteration': [], 'w_in': [],
                         'w_rnn': [], 'b_rnn': [], 'w_out': [], 'b_out': [], 'h': [], 'time': [],
                         'in_h': [], 'w_in2in': [], 'w_rnn2in': [], 'b_in': []}

    @ tf.function
    def train_onestep(syn_x_init, syn_u_init, in_data, out_target, mask_train):
        with tf.GradientTape() as t:
            loss, loss_orient, spike_loss, weight_loss, loss_orient_print = calc_loss(syn_x_init, syn_u_init, in_data, out_target, mask_train)
        grads = t.gradient(loss, var_list)
        grads_and_vars = list(zip(grads, var_list))
        capped_gvs = []
        for grad, var in grads_and_vars:
            if 'w_rnn' in var.name:
                grad *= par['w_rnn_mask']
            elif 'w_out' in var.name:
                grad *= par['w_out_mask']
            elif 'w_in' in var.name:
                grad *= par['w_in_mask']
            # elif 'w_in2in' in var.name:
            #     grad *= par['w_in2in_mask']
            capped_gvs.append((tf.clip_by_norm(grad, par['clip_max_grad_val']), var))
        opt.apply_gradients(grads_and_vars=capped_gvs)
        return loss, loss_orient, spike_loss, loss_orient_print

    i = 0
    for i in range(0, iteration):

        trial_info = stimulus.generate_trial()
        in_data = tf.constant(trial_info['neural_input'].astype('float32'))
        out_target = tf.constant(trial_info['desired_output'])
        mask_train = tf.constant(trial_info['mask'])

        loss, loss_orient, spike_loss, loss_orient_print = train_onestep(syn_x_init, syn_u_init, in_data, out_target, mask_train)

        itime = np.around((time.time() - t0) / 60, decimals=1)
        print('iModel=', iModel , ', iter=', i+1,
              ', loss=', loss.numpy(), ', loss_orient=', np.round(loss_orient.numpy()*par['orientation_cost']),
              ', spike_loss=', np.around(spike_loss.numpy()*par['spike_cost'], decimals=1),
              ', mean_loss_orient=', np.round(10000*loss_orient_print.numpy()*par['orientation_cost']),
              ', min=', itime)

        model_performance = append_model_performance(model_performance, loss, loss_orient, spike_loss, itime, var_dict)

        if np.mod(i+1, 500) == 0 or (np.mod(i+1, 20) == 0)*(i+1 <= 100) == 1\
                or (np.mod(i+1, 50) == 0)*(i+1 > 100)*(i+1 <= 500) == 1 or i < 10:
            save_results(model_performance, par, i+1)

    save_results(model_performance, par, i+1)