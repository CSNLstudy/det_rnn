import pickle
from det_rnn import *

par = update_parameters(par)
stimulus = Stimulus()
trial_info = stimulus.generate_trial()

model = Model()
for iter in range(3000):
    trial_info = stimulus.generate_trial()
    model(trial_info)
    if iter % 10 == 0:
        model.print_results(iter)

with open('/Volumes/Data_CSNL/project/RNN_study/20-06-19/HG/boost_wm/HL_Masse_mask_moderate.pkl','wb') as f:
    pickle.dump(model,f)



##
