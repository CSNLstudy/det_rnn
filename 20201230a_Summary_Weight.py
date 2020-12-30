import pickle
import os
from det_rnn import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.stats as stat

iModel = 5

iteration = 10000
BatchSize = 200

savedir = os.path.dirname(os.path.realpath(__file__)) + '/savedir/iModel' + str(iModel)

if not os.path.isdir(savedir + '/Training/Iter' + str(iteration)):
    os.makedirs(savedir + '/Training/Iter' + str(iteration))

modelname = '/Iter' + str(iteration) + '.pkl'
fn = savedir + modelname
model = pickle.load(open(fn, 'rb'))

par['batch_size'] = BatchSize
par = update_parameters(par)

h = model['h'][-1].numpy().astype('float32')
w_rnn = model['w_rnn'][-1].numpy().astype('float32')
b_rnn = model['b_rnn'][-1].numpy().astype('float32')
b_out = model['b_out'][-1].numpy().astype('float32')
w_out = model['w_out'][-1].numpy().astype('float32')
w_in = model['w_in'][-1].numpy().astype('float32')

ithreshold_in   = 98
ithreshold_rnn  = 90
ithreshold_out  = 92

iw_rnn = w_rnn.copy()
iw_out = w_out.copy()
iw_in = w_in.copy()

iw_rnn[iw_rnn<0]    = 0
iw_out[iw_out<0]    = 0
iw_in[iw_in<0]      = 0

iw_in_thre = np.percentile(np.reshape(iw_in, -1), ithreshold_in)
iw_rnn_thre = np.percentile(np.reshape(iw_rnn, -1), ithreshold_rnn)
iw_out_thre = np.percentile(np.reshape(iw_out, -1), ithreshold_out)

iw_rnn[iw_rnn < iw_rnn_thre] = 0
iw_out[iw_out < iw_out_thre] = 0
iw_in[iw_in < iw_in_thre] = 0

iw_rnn = par['EI_mask']@iw_rnn*par['w_rnn_mask']
iw_out = iw_out*par['w_out_mask']
iw_in = iw_in*par['w_in_mask']

iw_rnn[iw_rnn>0] = 1
iw_rnn[iw_rnn<0] = -1
iw_out[iw_out>0] = 1
iw_in[iw_in>0] = 1

plt.clf()
plt.imshow(iw_rnn,vmin=-1,vmax=1)
plt.colorbar()

plt.clf()
x = plt.hist(np.reshape(iw_rnn,-1),100)
plt.ylim((0, 50))

plt.clf()
plt.imshow(iw_out, aspect='auto',vmax=1)

plt.clf()
plt.hist(np.reshape(iw_out, -1), 100)
plt.ylim((0, 50))

plt.clf()
plt.imshow(iw_in, aspect='auto',vmax=1)

plt.clf()
x = plt.hist(np.reshape(iw_in, -1), 100)
plt.ylim((0, 50))
