#%%
from model import Model
from Stimulus_JS import Stimulus, Predefined_Stim
from hyper import make_par

from datetime import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle as pkl
import pandas as pd
#%%
nS = 8
n_batch = 100
n_hidden = 30

save_dir = Path('./saved_model')

STP = True
Version = 'basic'
# Version = 'distractor'
# Version = 'select'
# Version = 'basic_jitter'

gen_stim, gen_output = Predefined_Stim(Version, n_batch, nS)

#%%
input1 = gen_stim()
output1 = gen_output()
n_input = input1.shape[-1]
n_output = output1.shape[-1]
hp = make_par(n_input, n_hidden, n_output, n_batch, exc_inh_prop=0.8)

plotting = False
if plotting: 
    plt.figure(figsize = (8, 4))
    # iBatch = np.random.randint(0, n_batch)
    batches = [0, -1]
    for iBatch in batches:
        plt.figure(figsize = (8, 4))
        plt.subplot(2,1,1)
        plt.imshow(input1[:,iBatch,:].transpose(), aspect = 'auto')
        plt.subplot(2,1,2)
        plt.imshow(output1[:,iBatch,:].transpose(), aspect = 'auto')

# %%
nModel = 10

for iModel in range(nModel):
    print(iModel)

    nIter = 5000
    hp['masse'] = tf.cast(STP, dtype = tf.bool)
    RNN = Model(hp)

    mask = tf.constant((1-output1[:,:,0])+ 0.1, dtype = tf.float32)
    RNN.loss_mask = mask

    losses = []
    for Iter in range(nIter):
        input1 = gen_stim()
        output1 = gen_output()
        input_data = tf.constant(input1, dtype = tf.float32)
        output_data = tf.constant(output1, dtype = tf.float32)
        y, loss = RNN(input_data, output_data, hp)
        losses.append([loss['perf_loss'].numpy(), loss['spike_loss'].numpy()])
        
        if Iter%500 == 0:
            print(loss['perf_loss'].numpy(), loss['spike_loss'].numpy())

    
    ky = datetime.now().strftime("%m%d%H%M%S")
    
    if STP:
        file_path = save_dir/Version/'STP'/ky
    else:
        file_path = save_dir/Version/'noSTP'/ky
    
    tf.saved_model.save(RNN, str(file_path))

    loss_data = pd.DataFrame(losses, columns = ('perf_loss','spike_loss'))
    loss_data.to_csv(file_path/'loss.csv')

# #%%
# input1 = gen_stim()
# output1 = gen_output()
# input_data = tf.constant(input1, dtype = tf.float32)
# output_data = tf.constant(output1, dtype = tf.float32)

# iBatch = np.random.randint(0, n_batch)
# y, h_stack = RNN.rnn_model(input_data, hp)

# plotting = True
# if plotting: 
#     plt.figure(figsize = (8, 4))
#     # iBatch = np.random.randint(0, n_batch)
#     batches = [0, -1]
#     for iBatch in batches:

#         ori = np.argmax(output1[-1,iBatch,:])
#         plt.figure(figsize = (8, 4.5), dpi = 100)
#         plt.subplot(3,1,1)
#         plt.imshow(input1[:,iBatch,:].transpose(), aspect = 'auto')
#         plt.subplot(3,1,2)
#         plt.imshow(tf.nn.softmax(y)[:,iBatch,:].numpy().transpose(), aspect = 'auto')
#         plt.hlines(y = ori, xmin = 0, xmax = output1.shape[0]-1, color = 'red')
#         plt.subplot(3,1,3)
#         plt.imshow(h_stack[:,iBatch,:].numpy().transpose(), aspect = 'auto')

# %%


# import tensorflow as tf

# RNN_loaded = tf.saved_model.load('./saved_model/test/')
# # %%

# input1 = gen_stim()
# output1 = gen_output()
# RNN_loaded.rnn_model(RNN_loaded)