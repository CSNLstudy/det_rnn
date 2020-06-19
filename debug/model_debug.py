import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from det_rnn import *

par['design'].update({'iti'     : (0, 5.5),
                      'stim'    : (5.5,7.0),
                      'delay'   : (7.0,23.5),
                      'estim'   : (23.5,28.0)})

par = update_parameters(par)
stimulus = Stimulus()
trial_info = stimulus.generate_trial()
model = Model()

plt.imshow(model.EI_mask)
plt.show()

path = "/Volumes/Data_CSNL/project/RNN_study/20-06-05/HG/output/"
with open(path+'HL_Masse_longtime.pkl','wb') as f:
    pickle.dump(model,f)


fig, axes = plt.subplots(3,1, figsize=(10,8))
TEST_TRIAL = np.random.randint(stimulus.batch_size)
axes[0].imshow(trial_info['neural_input'][:,TEST_TRIAL,:].T, aspect='auto'); axes[0].set_title("Neural Input")
axes[1].imshow(trial_info['desired_output'][:,TEST_TRIAL,:].T, aspect='auto'); axes[1].set_title("Desired Output")
axes[2].imshow(trial_info['mask'][:,TEST_TRIAL,:].T, aspect='auto', vmax=1); axes[2].set_title("Training Mask")
fig.tight_layout(pad=2.0)
plt.show()


##
plt.imshow(tf.nn.relu(model.var_dict['w_rnn']) * model.EI_mask)
plt.show()



######
par = update_parameters(par)
stimulus = Stimulus()
trial_info = stimulus.generate_trial()

model = Model()
trial_info = stimulus.generate_trial()

model.masse

_Y, _H = model.rnn_model(trial_info['neural_input'])
rnn_input = trial_info['neural_input'][0,:,:]

_syn_x = model.syn_x_init
_syn_u = model.syn_u_init
_h = tf.tile(model.var_dict['h'], (_syn_x.shape[0], 1))

_h, _syn_x, _syn_u = model._rnn_cell(_h, rnn_input, _syn_x, _syn_u, 0)

_input_data = tf.unstack(input_data)


input_data = trial_info['neural_input']
_syn_x = model.syn_x_init
_syn_u = model.syn_u_init
_h = tf.tile(model.var_dict['h'], (_syn_x.shape[0], 1))
_input_data = tf.unstack(input_data)
h_stack = []
y_stack = []


for _iter, rnn_input in enumerate(_input_data):
    _h, _syn_x, _syn_u = model._rnn_cell(_h, rnn_input, _syn_x, _syn_u, _iter)
    plt.imshow(_h); plt.show()
    h_stack.append(_h)
    y_stack.append(_h @ model.var_dict['w_out'] + model.var_dict['b_out'])

h_stack = tf.stack(h_stack)
y_stack = tf.stack(y_stack)




######
for iter in range(2000):
    trial_info = stimulus.generate_trial()
    model(iter, trial_info['neural_input'], trial_info['desired_output'], trial_info['mask'])
    if iter % 10 == 0:
        model.print_results(iter)

n_iterations # what is this?

trial_info  = stimulus.generate_trial() # test by retain=9
pred_output = model.rnn_model(trial_info['neural_input'])

fig, axes = plt.subplots(3,1)
TEST_TRIAL = np.random.randint(model.batch_size)
axes[0].imshow(trial_info['neural_input'][:,TEST_TRIAL,:].T, aspect='auto')
axes[1].imshow(trial_info['desired_output'][:,TEST_TRIAL,:].T, aspect='auto')
axes[2].imshow(pred_output[0].numpy()[:,TEST_TRIAL,:].T,  aspect='auto') # TODO(HG): why is it tuple?
plt.show()

sout = np.sum(pred_output[0], axis=2)
sout = np.expand_dims(sout, axis=2)
noutput = pred_output[0] / np.repeat(sout,par['n_output'],axis=2)
cenoutput = tf.nn.softmax(pred_output[0], axis=2)
cenoutput = cenoutput.numpy()


fig, axes = plt.subplots(3,1)
TEST_TRIAL = np.random.randint(model.batch_size)
axes[0].imshow(trial_info['neural_input'][:,TEST_TRIAL,:].T, aspect='auto')
axes[1].imshow(trial_info['desired_output'][:,TEST_TRIAL,:].T, aspect='auto')
axes[2].imshow(cenoutput[:,TEST_TRIAL,:].T,  aspect='auto', vmax=0.15)
plt.show()

plt.imshow((model.var_dict['w_rnn'].numpy() * model.EI_mask).T)
plt.colorbar()
plt.show()




