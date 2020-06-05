import pickle
from det_rnn import *

par['design'].update({'iti'     : (0, 5.5),
                      'stim'    : (5.5,7.0),
                      'delay'   : (7.0,23.5),
                      'estim'   : (23.5,28.0)})

par = update_parameters(par)
stimulus = Stimulus()
trial_info = stimulus.generate_trial()

model = Model()
for iter in range(1000):
    trial_info = stimulus.generate_trial()
    model(iter, trial_info['neural_input'], trial_info['desired_output'], trial_info['mask'])
    if iter % 10 == 0:
        model.print_results(iter) 

with open('mse_nomasse.pkl','wb') as f:
    pickle.dump(model,f)



