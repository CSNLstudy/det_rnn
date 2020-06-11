import os
import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt

# necessary
from det_rnn import par, update_parameters
from det_rnn import Stimulus
from det_rnn import Model

from utils.plotfnc import plot_trial, make_var_dict_figure, plot_rnn_output

basedir = "D:\proj\det_rnn"

print(tf.__version__)

#par['input_rule']: {'fixation': (0, 6.0),
#               'response': (4.5, 6.0)},\
par['design'] = {'iti'	: (0, 0.5),
			   	'stim'	: (0.5, 4.0),
			   	'delay'	: (4.0, 4.5),
			   	'estim'	: (4.5, 6.0)}
par['output_range'] = ((4.5, 6.0))
par['input_rule'] = {'fixation': (0, 4.5),
                     'stim'	: (0.5, 4.0),
                     'response': (4.5, 6.0)}
par['output_rule'] ={'fixation': (0, 4.5),
                     'response': (4.5, 6.0)}
par['mask'] ={'iti'	: 1e3, 'stim' : 1e3, 'delay': 1e3, 'estim' : 1e7,
              'rule_iti' : 1e-2, 'rule_stim' : 1e-2, 'rule_delay' : 1e-2, 'rule_estim' : 1e-2}  # strength
par['masse'] = True

par = update_parameters(par)
stimulus = Stimulus(par)   # the argument `par` may be omitted
trial_info = stimulus.generate_trial()

plot_trial(stimulus,trial_info)
print('Input shape = ' + str(trial_info['neural_input'].shape))

# specify model
model = Model(par)

# train model
for iter in range(100):
    trial_info = stimulus.generate_trial()
    model(iter, trial_info['neural_input'], trial_info['desired_output'], trial_info['mask'])
    if iter % 10 == 0:
        model.print_results(iter)
        make_var_dict_figure(model)
        print('next set of iters...')


# save model
filename = os.path.join(basedir,'output','model_200522_masse.pkl')
with open(filename, 'wb') as f:
    pickle.dump(model, f)
    pickle.dump(par, f)
    pickle.dump(stimulus, f)

# plot model
trial_info = stimulus.generate_trial() # need to save stimulus somewhere too?
pred_output = model.rnn_model(trial_info['neural_input'])

TEST_TRIAL = np.random.randint(stimulus.batch_size)
#plot_trial(stimulus,trial_info, TEST_TRIAL=TEST_TRIAL)
print(par['resp_decoding'] )
print(trial_info['desired_output'][:,TEST_TRIAL,:].shape)
print(trial_info['desired_output'][480,TEST_TRIAL,:])
plot_rnn_output(par,trial_info,pred_output,stimulus, TEST_TRIAL=None)

print('hello')