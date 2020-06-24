import pickle
from det_rnn import *

par = update_parameters(par)
stimulus = Stimulus()
trial_info = stimulus.generate_trial()

model = Model()
for iter in range(2000):
    trial_info = stimulus.generate_trial()
    model(iter, trial_info['neural_input'], trial_info['desired_output'], trial_info['mask'])
    if iter % 10 == 0:
        model.print_results(iter) 

with open('mse_nomasse.pkl','wb') as f:
    pickle.dump(model,f)