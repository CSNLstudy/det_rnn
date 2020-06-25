import pickle
import os
from det_rnn import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
from shutil import copyfile
import operator
os.environ['KMP_DUPLICATE_LIB_OK']='True'

nModel = np.array([3, 10])
iteration = 100000
stimulus = Stimulus()

n_orituned_neurons          = 25
BatchSize                   = 70
noise_sd                    = 0 # input noise
scale_gamma                 = 0.5
n_hidden                    = 130

connect_p_within            = 0.8
connect_p_adjacent_forward  = 0.7
connect_p_distant_forward   = 0.0
connect_p_adjacent_back     = 0.3
connect_p_distant_back      = 0.0

alpha_input                 = 0.7 	# Chaudhuri et al., Neuron, 2015
alpha_hidden                = 0.2
alpha_output                = 0.4  # Chaudhuri et al., Neuron, 2015; Motor (F1) cortex's decay is in between input and hidden

delta_delay_update          = 0.5 # if estimation errors of consecutive "N_conseq_epoch_est_error" is lower than "criterion_est_error", delay increases by "delta_delay_update"
criterion_est_error         = 10
N_conseq_epoch_est_error    = 10
goal_delay                  = 15
Darwin_Iter                 = 2000
Darwin_EstError             = 30


par['n_hidden'] = n_hidden
par['n_tuned_input'] = n_orituned_neurons
par['n_tuned_output'] = n_orituned_neurons
par['n_ori'] = n_orituned_neurons
par['batch_size'] = BatchSize
par['scale_gamma'] = scale_gamma
par['connect_prob_within_module'] = connect_p_within
par['connect_prob_adjacent_module_forward'] = connect_p_adjacent_forward
par['connect_prob_distant_module_forward'] = connect_p_distant_forward
par['connect_prob_adjacent_module_back'] = connect_p_adjacent_back
par['connect_prob_distant_module_back'] = connect_p_distant_back
par['noise_sd'] = noise_sd
par['alpha_input'] = alpha_input 	# Chaudhuri et al., Neuron, 2015
par['alpha_hidden'] = alpha_hidden
par['alpha_output'] = alpha_output  # Chaudhuri et al., Neuron, 2015; Motor (F1) cortex has similar decay profile with sensory cortex
par['delta_delay_update'] = delta_delay_update
par['criterion_est_error'] = criterion_est_error
par['N_conseq_epoch_est_error'] = N_conseq_epoch_est_error
par['goal_delay'] = goal_delay
par['Darwin_Iter'] = Darwin_Iter
par['Darwin_EstError'] = Darwin_EstError

par = update_parameters(par)
stimulus = Stimulus(par)

# delay = 2.0
# par['design'].update({'iti'     : (0, 1.5),
#                       'stim'    : (1.5, 3.0),
#                       'delay'   : (3.0, 6.0 + delay),
#                       'estim'   : (6.0 + delay, 7.5 + delay)})
# par = update_parameters(par)
# stimulus = Stimulus(par)
# trial_info = stimulus.generate_trial()
# fig, axes = plt.subplots(3,1, figsize=(10,8))
# TEST_TRIAL = np.random.randint(stimulus.batch_size)
# a0 = axes[0].imshow(trial_info['neural_input'][:,TEST_TRIAL,:].T, aspect='auto'); axes[0].set_title("Neural Input"); fig.colorbar(a0, ax=axes[0])
# a1 = axes[1].imshow(trial_info['desired_output'][:,TEST_TRIAL,:].T, aspect='auto'); axes[1].set_title("Desired Output"); fig.colorbar(a1, ax=axes[1])
# a2 = axes[2].imshow(trial_info['mask'][:,TEST_TRIAL,:].T, aspect='auto'); axes[2].set_title("Training Mask"); fig.colorbar(a2, ax=axes[2]) # a bug here
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
            ivar_dict[name] = tf.Variable(ipar[k], name, dtype='float32')
            ivar_list.append(ivar_dict[name])
    isyn_x_init = tf.constant(ipar['syn_x_init'])
    isyn_u_init = tf.constant(ipar['syn_u_init'])
    ibatch_size = ipar['batch_size']

    delay = par['design']['delay'][1] - par['design']['delay'][0]
    isavedir = os.path.dirname(os.path.realpath(__file__)) + \
               '/savedir' \
               '/connectp_w' + str(connect_p_within) + '_forward_a' + str(connect_p_adjacent_forward) + 'd' + str(connect_p_distant_forward) + \
                    'back_a' + str(connect_p_adjacent_back) + 'd' + str(connect_p_distant_back) + 'scalegamma' + str(scale_gamma) + \
               '/alpha_in' + str(par['alpha_input']) + '_h' + str(par['alpha_hidden']) + '_out' + str(par['alpha_output']) + \
               '/delay_Delta' + str(par['delta_delay_update']) + '_ErrorCri' + str(par['criterion_est_error']) + '_Ntrial' + str(par['N_conseq_epoch_est_error']) + \
               '/nIter' + str(iteration) + 'BatchSize' + str(BatchSize) + \
               '/Delay' + str(delay) + \
               '/iModel' + str(iModel)
    if not os.path.isdir(isavedir):
        os.makedirs(isavedir)
        os.makedirs(isavedir + '/code/')
        os.makedirs(isavedir + '/code/det_rnn')
    codenames = ['main_SensoryOutputFeedbackConnected.py', 'Summary_Training.py', 'det_rnn/__init__.py', 'det_rnn/_functions.py',
                 'det_rnn/_model.py', 'det_rnn/_parameters.py', 'det_rnn/_stimulus.py']
    for codename in codenames:
        copyfile(os.path.dirname(os.path.realpath(__file__)) + '/' + codename, isavedir + '/code/' + codename)

    return ipar, ivar_dict, ivar_list, isyn_x_init, isyn_u_init, ibatch_size, isavedir

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

def run_model(in_data, syn_x_init, syn_u_init):

    self_h = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    self_syn_x = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    self_syn_u = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    self_output = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    h = np.ones((par['batch_size'], 1)) @ var_dict['h']
    syn_x = syn_x_init
    syn_u = syn_u_init
    w_rnn = par['EI_mask'] @ (par['modular_sparse_mask_initial'] * tf.nn.relu(var_dict['w_rnn']))

    for c in range(par['n_timesteps']):
        rnn_input = in_data[c,:,:]
        h, syn_x, syn_u = rnn_cell(rnn_input, h, syn_x, syn_u, w_rnn)

        self_h = self_h.write(c, h)
        self_syn_x = self_syn_x.write(c, syn_x)
        self_syn_u = self_syn_u.write(c, syn_u)
        self_output = self_output.write(c, h[:, -par['n_output']:])

    self_h = self_h.stack()
    self_syn_x = self_syn_x.stack()
    self_syn_u = self_syn_u.stack()
    self_output = self_output.stack()

    return self_h, self_output, self_syn_x, self_syn_u, w_rnn

def calc_loss(syn_x_init, syn_u_init, in_data, out_target, mask_train, trial_info):

    h, output, _, _, w_rnn = run_model(in_data, syn_x_init, syn_u_init)

    starget = tf.reduce_sum(out_target, axis=2)
    starget = tf.expand_dims(starget, axis=2)
    ntarget = out_target / tf.repeat(starget, par['n_output'], axis=2)

    noutput = tf.nn.log_softmax(output, axis=2)
    CE = -ntarget*noutput
    loss_orient = tf.reduce_sum(mask_train*CE)
    loss_orient_print = tf.reduce_mean(mask_train*CE)

    n = 2
    spike_loss = tf.reduce_sum(h**2)
    weight_loss = tf.reduce_sum(tf.nn.relu(w_rnn) ** n)
    loss = par['orientation_cost'] * loss_orient + par['spike_cost'] * spike_loss + par['weight_cost'] * weight_loss

    MAP_dirs = output[par['output_rg'][par['output_rg'] > np.max(par['dead_rg'])].min():, :, 1:]
    MAP_dirs = tf.nn.log_softmax(MAP_dirs, axis=2)
    MAP_dirs = tf.reduce_sum(MAP_dirs, axis=0)
    MAP_dirs = tf.argmax(MAP_dirs, axis=1)
    stim_dirs = tf.constant(par['stim_dirs'])
    MAP_dirs = tf.gather(stim_dirs, MAP_dirs)
    Target_dirs = tf.gather(stim_dirs, trial_info['stimulus_ori'])
    est_error = tf.reduce_mean(tf.math.acos(tf.math.cos(2 * (MAP_dirs - Target_dirs) / 180 * np.pi)) / np.pi * 180 / 2)

    return loss, loss_orient, spike_loss, weight_loss, loss_orient_print, est_error

def append_model_performance(model_performance, loss, loss_orient, spike_loss, est_error, delay, itime, var_dict):
    model_performance['time'].append(itime)
    model_performance['loss'].append(loss)
    model_performance['loss_orient'].append(loss_orient)
    model_performance['spike_loss'].append(spike_loss)
    model_performance['est_error'].append(est_error)
    model_performance['delay'].append(delay)
    model_performance['iteration'].append(iteration)

    model_performance['w_rnn'].append(var_dict['w_rnn'])
    model_performance['b_rnn'].append(var_dict['b_rnn'])
    model_performance['h'].append(var_dict['h'])

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

    par, var_dict, var_list, syn_x_init, syn_u_init, batch_size, savedir = initialize_parameters(iModel, par)
    opt = tf.optimizers.Adam(learning_rate=par['learning_rate'])
    model_performance = {'loss': [], 'loss_orient': [], 'spike_loss': [], 'est_error': [], 'delay': [],
                        'iteration': [], 'w_in': [], 'w_rnn': [], 'b_rnn': [], 'h': [], 'time': []}
    par['modular_sparse_mask_initial'] = par['modular_sparse_mask']

    @ tf.function
    def train_onestep(syn_x_init, syn_u_init, in_data, out_target, mask_train, trial_info):
        with tf.GradientTape() as t:
            loss, loss_orient, spike_loss, weight_loss, loss_orient_print, est_error = calc_loss(syn_x_init, syn_u_init, in_data, out_target, mask_train, trial_info)
        vars_and_grads = t.gradient(loss, var_dict)
        capped_gvs = []
        for var, grad in vars_and_grads.items():
            if 'w_rnn' in var:
                grad *= par['w_rnn_mask'] * par['modular_sparse_mask_initial']
            capped_gvs.append((tf.clip_by_norm(grad, par['clip_max_grad_val']), var_dict[var]))
        opt.apply_gradients(grads_and_vars=capped_gvs)
        return loss, loss_orient, spike_loss, loss_orient_print, est_error

    i = 0
    cUpdate = 1
    for i in range(0, iteration):

        trial_info = stimulus.generate_trial()
        in_data = tf.constant(trial_info['neural_input'].astype('float32'))
        out_target = tf.constant(trial_info['desired_output'])
        mask_train = tf.constant(trial_info['mask'])

        loss, loss_orient, spike_loss, loss_orient_print, est_error \
            = train_onestep(syn_x_init, syn_u_init, in_data, out_target, mask_train, trial_info)

        itime = np.around((time.time() - t0) / 60, decimals=1)
        idelay = 1.5 + (cUpdate-1)*par['delta_delay_update']
        print('iModel=', iModel , ', iter=', i+1, ', delay=', idelay,
              ', loss=', loss.numpy(), ', loss_orient=', np.round(loss_orient.numpy()*par['orientation_cost']),
              ', spike_loss=', np.around(spike_loss.numpy()*par['spike_cost'], decimals=1),
              ', mean_loss_orient=', np.round(10000*loss_orient_print.numpy()*par['orientation_cost']),
              ', est_error=', np.around(est_error.numpy(), decimals=2),
              ', min=', itime)

        model_performance = append_model_performance(model_performance, loss, loss_orient, spike_loss, est_error, idelay, itime, var_dict)

        if np.mod(i+1, 500) == 0 or (np.mod(i+1, 20) == 0)*(i+1 <= 100) == 1\
                or (np.mod(i+1, 50) == 0)*(i+1 > 100)*(i+1 <= 500) == 1 or i < 10:
            save_results(model_performance, par, i+1)
            if cUpdate*par['delta_delay_update'] > par['goal_delay']:
                print('##')
                print('iModel=', iModel, ' training has been accomplished')
                print('##')
                break

        if i+1 == par['Darwin_Iter'] and np.mean(model_performance['est_error'][-par['N_conseq_epoch_est_error']:]) > par['Darwin_EstError']:
            print('##')
            print('iModel=', iModel, 'does not have any potential. Current est_error=',
                  str(np.mean(model_performance['est_error'][-par['N_conseq_epoch_est_error']:])), '. So, training is halted.')
            print('##')
            break

        if i > 10:
            if np.mean(model_performance['est_error'][-par['N_conseq_epoch_est_error']:]) < par['criterion_est_error']:
                par['design'].update({'iti'     : (0, 1.5),
                                      'stim'    : (1.5, 3.0),
                                      'delay'   : (3.0, 3.5 + cUpdate*par['delta_delay_update']),
                                      'estim'   : (3.5 + cUpdate*par['delta_delay_update'], 5.0 + cUpdate*par['delta_delay_update'])})
                par = update_parameters(par)
                stimulus = Stimulus(par)
                print('##')
                print(str(cUpdate) + ' Delay Update, from '
                      + str(1.5 + (cUpdate-1)*par['delta_delay_update']) +
                      ' to ' + str(1.5 + cUpdate*par['delta_delay_update']))
                print('##')
                cUpdate += cUpdate

    save_results(model_performance, par, i+1)