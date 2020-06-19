import os, time, pickle
import numpy as np
import tensorflow as tf
from det_rnn import *
import matplotlib.pyplot as plt


base_path  = "/Volumes/Data_CSNL/project/RNN_study/20-06-19/HG/boost_wm/"
model_code = "HL_booster1_resume"
os.makedirs(base_path+model_code, exist_ok=True)

model_list = os.listdir(base_path+"HL_booster1")
model_list = [m for m in model_list if m.endswith(".pkl")]
latest = base_path+"HL_booster1/"+sorted(model_list)[-1]

with open(latest,'rb') as f:
    childrnn = pickle.load(f)

########################################################################
childrnn.model_performance['loss'][-8]

par['design'].update({'iti'     : (0, 1.5),
                      'stim'    : (1.5,3.0),
                      'delay'   : (3.0,9.0),
                      'estim'   : (9.0,10.5)})
par = update_parameters(par)
stimulus = Stimulus()
trial_info = stimulus.generate_trial()

fig, axes = plt.subplots(3,1, figsize=(10,8))
TEST_TRIAL = np.random.randint(stimulus.batch_size)
axes[0].imshow(trial_info['neural_input'][:,TEST_TRIAL,:].T, aspect='auto'); axes[0].set_title("Neural Input")
axes[1].imshow(trial_info['desired_output'][:,TEST_TRIAL,:].T, aspect='auto'); axes[1].set_title("Desired Output")
axes[2].imshow(trial_info['mask'][:,TEST_TRIAL,:].T, aspect='auto'); axes[2].set_title("Training Mask")
fig.tight_layout(pad=2.0)
plt.show()


trial_info['neural_input'].shape
Y, Loss = childrnn._train_oneiter(trial_info['neural_input'],
                                  trial_info['desired_output'],
                                  trial_info['mask'])
childrnn._append_model_performance(trial_info, Y, Loss, par=par)



childrnn(trial_info)

model()
tf.saved_model.save(model, "/Volumes/Data_CSNL/project/RNN_study/20-06-19/HG/boost_wm/HL_Masse_booster/2/")


model_loaded = tf.saved_model.load("/Volumes/Data_CSNL/project/RNN_study/20-06-19/HG/boost_wm/HL_Masse_booster/2/")

Y, Loss = model._train_oneiter(trial_info['neural_input'],
                               trial_info['desired_output'],
                               trial_info['mask'])

model_loaded.summary()
model_loaded.__call__(trial_info)

model.model_performance['iteration']
Loss['perf_loss']


#####
par = update_parameters(par)
stimulus = Stimulus()
trial_info = stimulus.generate_trial()

model = Model()
for iter in range(200):
    trial_info = stimulus.generate_trial()
    model(trial_info)
    if iter % 10 == 0:
        model.print_results(iter)
