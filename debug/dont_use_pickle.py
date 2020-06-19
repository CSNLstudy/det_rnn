#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:08:17 2020

@author: hyunwoogu
"""
import pickle
import numpy as np
import tensorflow as tf # tf 2.0.0

params  = {'cost_w': 0.5, # 'weight' multiplied to loss (hyperparameter to change)
           'W0' : np.random.rand(1), # initial to W
           'b0' : np.random.rand(1)} # initial to b

class Model(tf.Module):
    def __init__(self, params=params):
        super(Model, self).__init__()
        self.set_params(params)
        self.init_vars(params)
        self.optimizer = tf.optimizers.Adam(learning_rate=0.1) 

    def set_params(self, params):
        for k, v in params.items():
            setattr(self, k, v)

    def init_vars(self, params):
        _var_dict = {}
        for k, v in params.items():
            if k[-1] == '0':
                name = k[:-1]
                _var_dict[name] = tf.Variable(params[k], name=name, dtype='float32')
        self.var_dict = _var_dict

    def __call__(self, x,y): # x is data
        loss = self.train_oneiter(x,y)
        print("loss:", loss.numpy())

    def run_model(self, x):
        pred = self.var_dict['W'] * x + self.var_dict['b']
        return pred

    @tf.function
    def train_oneiter(self, x,y):
        with tf.GradientTape() as t:
            pred = self.run_model(x)
            loss = self.cost_w*tf.reduce_mean((pred-y)**2)
        vars_and_grads = t.gradient(loss, self.var_dict)
        capped_gvs = [] # gradient clipping: looks an overkill here, but anyways
        for var, grad in vars_and_grads.items():
            capped_gvs.append((tf.clip_by_norm(grad, 1.0), self.var_dict[var]))
        self.optimizer.apply_gradients(grads_and_vars=capped_gvs)
        return loss
    
    
model = Model(params)
for iter in range(100):
    x = np.random.randint(10,size=5).astype(np.float32)
    y = x
    model(x,y) # train!
    
# save
pkl_savepath = "/Users/hyunwoogu/Documents/model_res/res.pkl"
tf_savepath  = "/Users/hyunwoogu/Documents/model_res/test/"

with open(pkl_savepath, 'wb') as f:
    pickle.dump(model, f) # pickle save
tf.saved_model.save(model, tf_savepath) # tf save


# load
with open(pkl_savepath,'rb') as f:
    pkl_model = pickle.load(f)

tf_model = tf.saved_model.load(tf_savepath)


# 
for iter in range(10):
    x = np.random.randint(10,size=5).astype(np.float32)
    y = x
    pkl_model(x,y)

for iter in range(10):
    x = np.random.randint(10,size=5).astype(np.float32)
    y = x
    loss = tf_model.train_oneiter(x,y)
    print(loss.numpy())
    
# Use functions defined in model
pkl_model.run_model(x)
tf_model.run_model(x) # error 

# Change hyperparameters
tf_model.cost_w *= 1000 # error 
for iter in range(10):
    x = np.random.randint(10,size=5).astype(np.float32)
    y = x
    loss = tf_model.train_oneiter(x,y)
    print(loss.numpy())