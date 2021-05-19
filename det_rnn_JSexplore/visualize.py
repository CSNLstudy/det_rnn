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

save_dir = Path('./saved_model')

STP = False
# Version = 'basic'
# Version = 'distractor'
Version = 'select'
# Version = 'basic_jitter'


#% Load trained RNN

if STP:
    file_path = save_dir/Version/'STP'
else:
    file_path = save_dir/Version/'noSTP'

folders = sorted(list(file_path.glob('*')))

RNNs = []
LOSSs = []
for folder in folders:
    RNNs.append(tf.saved_model.load(str(folder)))
    loss = pd.read_csv(folder/'loss.csv')
    LOSSs.append(loss)

# Basic settings 

nS = 8
n_batch = 100
n_hidden = 30

gen_stim, gen_output = Predefined_Stim(Version, n_batch, nS)
input1 = gen_stim()
output1 = gen_output()
n_input = input1.shape[-1]
n_output = output1.shape[-1]
hp = make_par(n_input, n_hidden, n_output, n_batch, exc_inh_prop=0.8)
hp['masse'] = tf.cast(STP, dtype = tf.bool)
#%% visualizing training result
plotting = False
if plotting: 
    fig, axs = plt.subplots(2,1, figsize = (8,6), dpi = 150)
    plt.subplots_adjust(hspace = 0.5)
    for loss in LOSSs:
        x = np.arange(1, 5000, 100)
        axs[0].plot(x, loss['perf_loss'][x])
        axs[0].set_ylim((0,0.4))
        axs[0].set_title('Performance loss')
        axs[1].plot(x, loss['spike_loss'][x])
        axs[1].set_ylim((0,0.04))
        axs[1].set_title('Spike loss')
        axs[1].set_xlabel('Iteration')
        axs[0].legend(np.arange(10),  bbox_to_anchor=(1, 0.5))


    for RNN in RNNs:
        fig, axs = plt.subplots(1,3, figsize = (8,2.5))
        axs[0].imshow(RNN.var_dict['w_in'])
        axs[1].imshow(RNN.var_dict['w_rnn'])
        axs[2].imshow(RNN.var_dict['w_out'])


#%% Desired output
plotting = False

input1 = gen_stim()
output1 = gen_output()
input_data = tf.constant(input1, dtype = tf.float32)
output_data = tf.constant(output1, dtype = tf.float32)

if plotting: 
    plt.figure(figsize = (8, 4))
    # iBatch = np.random.randint(0, n_batch)
    batches = [0, -1]
    for iBatch in batches:

        ori = np.argmax(output1[-1,iBatch,:])
        plt.figure(figsize = (8, 4.5), dpi = 100)
        plt.subplots_adjust(hspace = 0.4)
        plt.subplot(3,1,1)
        plt.imshow(input1[:,iBatch,:].transpose(), aspect = 'auto')
        plt.subplot(3,1,2)
        plt.imshow(output1[:,iBatch,:].transpose(), aspect = 'auto')
        plt.hlines(y = ori, xmin = 0, xmax = output1.shape[0]-1, color = 'red')
        
        
#%%
plotting = False
cand = [3,4]
for iR in cand:
    RNN = RNNs[iR]
    
    input1 = gen_stim()
    output1 = gen_output()
    input_data = tf.constant(input1, dtype = tf.float32)
    output_data = tf.constant(output1, dtype = tf.float32)

    iBatch = np.random.randint(0, n_batch)
    y, h_stack = RNN.rnn_model(input_data, hp)

    if plotting: 
        plt.figure(figsize = (8, 4))
        # iBatch = np.random.randint(0, n_batch)
        batches = [0, -1]
        for iBatch in batches:

            ori = np.argmax(output1[-1,iBatch,:])
            plt.figure(figsize = (8, 4.5), dpi = 100)
            plt.subplots_adjust(hspace = 0.4)
            plt.subplot(3,1,1)
            plt.imshow(input1[:,iBatch,:].transpose(), aspect = 'auto')
            plt.title('Network #{}'.format(iR))
            plt.subplot(3,1,2)
            plt.imshow(tf.nn.softmax(y)[:,iBatch,:].numpy().transpose(), aspect = 'auto')
            plt.hlines(y = ori, xmin = 0, xmax = output1.shape[0]-1, color = 'red')
            plt.subplot(3,1,3)
            plt.imshow(h_stack[:,iBatch,:].numpy().transpose(), aspect = 'auto')
            

# %%

def align_resp(RNN, targ_idx = 15, targ_batch = np.arange(100)):
    
    retrieve_from = 10
    input1 = gen_stim()
    output1 = gen_output()
    input_data = tf.constant(input1, dtype = tf.float32)
    output_data = tf.constant(output1, dtype = tf.float32)
    y, h_stack = RNN.rnn_model(input_data, hp)
    sft = np.mean(tf.nn.softmax(y).numpy()[-retrieve_from:,targ_batch,1:], axis = 0)
    inp = input1[targ_idx,targ_batch,3:]
    
    # plt.subplot(2,1,1)    
    # plt.imshow(inp.transpose(), aspect = 'auto')
    # plt.subplot(2,1,2)    
    # plt.imshow(sft.transpose(), aspect = 'auto')
    
    # plt.imshow(input1[:,10, 3:])
    
    # plt.plot(input1[79, targ_batch, 5])
    
    
    inp_max = np.argmax(inp, axis = 1)
    stk = []
    for i,s in zip(inp_max, sft):
        stk.append(np.roll(s, 4-i))
    
    out = np.stack(stk, axis = 0)

    return out

if Version == 'distractor':
    nRep = 100
    targ_idxs = [15, 75]
    RNN = RNNs[9]

    for targ_idx in targ_idxs:
        stk = []
        for i in range(nRep):
            aligned = align_resp(RNN, targ_idx = targ_idx, targ_batch = np.arange(100))
            stk.append(aligned)
        aligned_merge = np.concatenate(stk,axis = 0)

        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(np.mean(aligned_merge, axis = 0), 'o')
        plt.ylim(-0.1, 1.1)
        plt.subplot(2,1,2)
        plt.plot(np.mean(aligned_merge, axis = 0), 'o')


NetID = 5

if Version == 'select':
    nRep = 100
    targ_idxs = [15, 75]
    RNN = RNNs[NetID]

    for iT, targ_idx in enumerate(targ_idxs):
        stk = []
        for i in range(nRep):
            aligned = align_resp(RNN, targ_idx = targ_idx, targ_batch = np.arange(50))
            stk.append(aligned)
        aligned_merge = np.concatenate(stk,axis = 0)

        if iT == 1:
            key = 'ignored'
        else:
            key = 'selected'
            
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(np.mean(aligned_merge, axis = 0), 'o')
        plt.title('target {} ({})'.format(iT+1, key))
        plt.ylim(-0.1, 1.1)
        plt.subplot(2,1,2)
        plt.plot(np.mean(aligned_merge, axis = 0), 'o')
        plt.subplots_adjust(hspace = 0.5)
        plt.title('Network {}'.format(NetID))

    for iT, targ_idx in enumerate(targ_idxs):

        stk = []
        for i in range(nRep):
            aligned = align_resp(RNN, targ_idx = targ_idx, targ_batch = np.arange(51, 100))
            stk.append(aligned)
        aligned_merge = np.concatenate(stk,axis = 0)

        if iT == 1:
            key = 'selected'
        else:
            key = 'ignored'
            
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(np.mean(aligned_merge, axis = 0), 'o')
        plt.title('target {} ({})'.format(iT+1, key))
        plt.ylim(-0.1, 1.1)
        plt.subplot(2,1,2)
        plt.plot(np.mean(aligned_merge, axis = 0), 'o')
        plt.subplots_adjust(hspace = 0.5)
        plt.title('Network {}'.format(NetID))

        
        

#%%# %%





