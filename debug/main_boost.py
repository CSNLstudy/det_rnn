import os, time, pickle, copy
import numpy as np
import tensorflow as tf
from det_rnn import *

##
perf_crit   = 0.95 # Human adult(around 23) level
recency     = 50   # number of 'recent' epochs to be assayed
boost_step  = 1.5
extend_time = np.arange(boost_step,15.5,step=boost_step)
mileage_lim = len(extend_time)
milestones  = np.zeros((mileage_lim,), dtype=np.int64)
timestones  = np.zeros((mileage_lim,))

base_path  = "/Volumes/Data_CSNL/project/RNN_study/20-06-19/HG/boost_wm/"
model_code = "HL_booster3"
os.makedirs(base_path+model_code, exist_ok=True)

par = update_parameters(par)
stimulus = Stimulus()

## TODO(HG): modify
stim_par = copy.deepcopy(par)
model = Model(par)
mileage = -1
start_time = time.time()

print("RNN Booster started!")
for iter in range(100000):
    trial_info = stimulus.generate_trial()
    model(trial_info)
    if iter % 30 == 0:
        model.print_results(iter)

    indx_recent = np.arange(iter)[-recency:]
    indx_milest = np.arange(iter)[milestones[mileage]:]
    indx_inters = np.intersect1d(indx_recent,indx_milest)
    perf_vec    = np.array(model.model_performance['perf'])[indx_inters]

    if  (np.mean(perf_vec) > perf_crit) & (len(perf_vec) >= recency):
        check_time = time.time()
        mileage += 1
        if mileage >= mileage_lim:
            print("#"*80+"\nTraining criterion finally met!(Time Spent: {:0.2f}s)\t".format(check_time-start_time)+
                  "Now climb down the mountain!\n"+"#"*80)
            break
        milestones[mileage] = iter
        timestones[mileage] = check_time-start_time
        model.milestones = milestones
        model.timestones = timestones

        os.makedirs(base_path + model_code + "/model_level" + str(mileage), exist_ok=True)
        with open(base_path + model_code + "/model_level" + str(mileage) + "/corpse.pkl", 'wb') as f:
            pickle.dump(model, f)
        tf.saved_model.save(model, base_path + model_code + "/model_level" + str(mileage))

        stim_par['design'].update({'iti': (0, 1.5),
                              'stim': (1.5, 3.0),
                              'delay': (3.0, 4.5 + extend_time[mileage]),
                              'estim': (4.5 + extend_time[mileage], 6.0 + extend_time[mileage])})
        # par['mask']['estim'] *= 2.
        stim_par = update_parameters(stim_par)
        stimulus = Stimulus(stim_par)
        # model.spike_cost /= 2.
        model.design_rg_estim = stim_par['design_rg']['estim']
        print("#"*80+"\nCriterion satisfied!(Time Spent: {:0.2f}s)\t".format(check_time-start_time)+
                     "Now extending: {:0.1f}\n".format(extend_time[mileage])+"#"*80)



with open('/Volumes/Data_CSNL/project/RNN_study/20-06-19/HG/boost_wm/boost_vanila4.pkl','wb') as f:
    pickle.dump(model,f)



import matplotlib.pyplot as plt
plt.plot(model.model_performance['loss'])
plt.show()


################################################################
## Resuming: extended time: 0.5s
base_path  = "/Volumes/Data_CSNL/project/RNN_study/20-06-19/HG/boost_wm/"
model_code = "HL_booster1_resume"
os.makedirs(base_path+model_code, exist_ok=True)

model_list = os.listdir(base_path+"HL_booster1")
model_list = [m for m in model_list if m.endswith(".pkl")]
latest = base_path+"HL_booster1/"+sorted(model_list)[-1]

with open(latest,'rb') as f:
    childrnn = pickle.load(f)


perf_crit   = 0.95 # Human adult(around 23) level
recency     = 50   # number of 'recent' epochs to be assayed
boost_step  = 0.5
extend_time = np.arange(0,9.5,step=boost_step)
mileage_lim = len(extend_time)
milestones  = np.zeros((mileage_lim,), dtype=np.int64)
timestones  = np.zeros((mileage_lim,))


mileage = -1
start_time = time.time()
print("RNN Booster started!")
for iter in range(100000):
    trial_info = stimulus.generate_trial()
    childrnn(trial_info, par=par['design'])
    if iter % 30 == 0:
        childrnn.print_results(iter)

    indx_recent = np.arange(iter)[-recency:]
    indx_milest = np.arange(iter)[milestones[mileage]:]
    indx_inters = np.intersect1d(indx_recent,indx_milest)
    perf_vec    = np.array(childrnn.model_performance['perf'])[indx_inters]

    if  (np.mean(perf_vec) > perf_crit) & (len(perf_vec) >= recency):
        check_time = time.time()
        mileage += 1
        if mileage >= mileage_lim:
            print("#"*80+"\nTraining criterion finally met!(Time Spent: {:0.2f}s)\t".format(check_time-start_time)+
                  "Now climb down the mountain!\n"+"#"*80)
            break
        milestones[mileage] = iter
        timestones[mileage] = check_time-start_time
        childrnn.milestones = milestones
        childrnn.timestones = timestones

        with open(base_path + model_code + "/model_level" + str(mileage) + ".pkl", 'wb') as f:
            pickle.dump(childrnn, f)

        par['design'].update({'iti': (0, 1.5),
                              'stim': (1.5, 3.0),
                              'delay': (3.0, 9.0 + extend_time[mileage]),
                              'estim': (9.0 + extend_time[mileage], 10.5 + extend_time[mileage])})

        # par['mask']['estim'] *= 2.
        par = update_parameters(par)
        stimulus = Stimulus(par)
        # model.spike_cost /= 2.
        print("#"*80+"\nCriterion satisfied!(Time Spent: {:0.2f}s)\t".format(check_time-start_time)+
                     "Now extending: {:0.1f}\n".format(extend_time[mileage])+"#"*80)





# TODO(HG): stimulus generation with probability