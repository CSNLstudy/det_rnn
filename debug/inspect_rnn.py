#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 13:33:32 2020

@author: hyunwoogu
"""

%## Inspecting tensorboard
%load_ext tensorboard
from datetime import datetime
from packaging import version

import tensorflow as tf
from tensorflow import keras

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."
import tensorboard
tensorboard.__version__

log_dir = "/Volumes/Data_CSNL/project/RNN_study/20-06-19/HG/boost_wm/HL_booster3/"
%tensorboard --logdir logs


## Define zombify using both .pb & .pkl (zombie's class: _UserObject)


##
signatures = {
    # 'rnn_model': model.rnn_model.get_concrete_function(tf.TensorSpec(shape=[None, None, None], dtype=tf.float32, name="neural_input")),
    'print_results': model.print_results.get_concrete_function( tf.TensorSpec(shape=[1,], dtype=tf.int64, name="iter"))
}

options = tf.saved_model.SaveOptions(function_aliases={
    # '_calc_loss': _calc_loss,
    # '_get_eval': _get_eval,
    # '_rnn_cell': _rnn_cell,
    'rnn_model': model.rnn_model,
    'print_results': model.print_results
})

signatures = {
    'serving_default': model.__init__.get_concrete_function(),
}




par['design'].update({'iti'     : (0, 5.5),
                      'stim'    : (5.5,7.0),
                      'delay'   : (7.0,23.5),
                      'estim'   : (23.5,28.0)})


#%%
import pickle
import numpy as np
import tensorflow as tf

# params  = {'cost_w': 0.5, # 'weight' multiplied to loss
#            'W0' : np.random.rand(1), # initial to W
#            'b0' : np.random.rand(1)} # initial to b 

hp = {'add_par': 5.,'timestep': 2,
      'cost_w': 0.5, # 'weight' multiplied to loss
      'lr' : 0.1,
      'W0' : np.random.rand(1).astype(np.float32), # initial to W
      'b0' : np.random.rand(1).astype(np.float32)} # initial to b

hp_spec = {}
for k, v in hp.items():
    hp_spec[k] = tf.TensorSpec(np.array(v).shape, tf.dtypes.as_dtype(np.array(v).dtype), name=k)
    
x_spec = tf.TensorSpec([None,5])
y_spec = tf.TensorSpec([5])

# class Model(tf.keras.Model):
class Model(tf.Module):
    def __init__(self, hp=hp):
        super(Model, self).__init__()
        self.init_vars(hp)
        self.optimizer = tf.optimizers.Adam(learning_rate=hp['lr']) 

    def init_vars(self, hp=hp):
        _var_dict = {}
        for k, v in hp.items():
            if k[-1] == '0':
                name = k[:-1]
                _var_dict[name] = tf.Variable(hp[k], name=name, dtype='float32')
        self.var_dict = _var_dict

    # @tf.function(input_signature=[x_spec,y_spec,hp_spec])
    @tf.function
    def __call__(self, x,y, hp):
        loss = self.train_oneiter(x,y, hp)
        return loss
    
    # @tf.function(input_signature=[x_spec,hp_spec])
    @tf.function
    def run_model(self, _x, hp):
        # _x = tf.TensorArray(tf.float32, size=0,dynamic_size=True,infer_shape=False)
        # for i in range(hp['timestep']):
        #     _x.write(i, x[i])
        pred = tf.repeat([0.],5)
        for x_data in _x:
            pred = tf.cast(pred,tf.float32)+ \
            tf.cast(self.var_dict['W'],tf.float32) * tf.cast(x_data,tf.float32) + \
            tf.cast(self.var_dict['b'],tf.float32) + tf.cast(hp['add_par'],tf.float32)
        return pred

    # @tf.function(input_signature=[x_spec,y_spec,hp_spec])
    @tf.function
    def train_oneiter(self, x,y, hp):
        with tf.GradientTape() as t:
            pred = tf.cast(self.run_model(x, hp),tf.float32)
            loss = tf.cast(hp['cost_w'],tf.float32)*\
                tf.cast(tf.reduce_mean((pred-y)**2),tf.float32)
        vars_and_grads = t.gradient(loss, self.var_dict)
        capped_gvs = [] # gradient clipping: looks an overkill here, but anyways
        for var, grad in vars_and_grads.items():
            capped_gvs.append((tf.clip_by_norm(grad, 1.0), self.var_dict[var]))
        self.optimizer.apply_gradients(grads_and_vars=capped_gvs)
        return loss



# Tensorize
for k, v in hp.items():
    hp[k] = tf.constant(v)

# train y=x mapping
model = Model(hp)

#
model.run_model.get_concrete_function(
    _x=x_spec,
    hp=hp_spec
)
model.__call__.get_concrete_function(
    x=x_spec,
    y=y_spec,
    hp=hp_spec
)
model.train_oneiter.get_concrete_function(
    x=x_spec,
    y=y_spec,
    hp=hp_spec
)

for iter in range(100):
    x = np.random.randint(10,size=(2,5)).astype(np.float32)
    y = x.sum(axis=0)
    # loss = model([tf.constant(x[i]) for i in range(2)],
    #              tf.constant(y), hp) # train!
    loss = model(tf.constant(x),
             tf.constant(y), hp) # train!
    print("loss:", loss.numpy())


# save
pkl_savepath   = "/Users/hyunwoogu/Documents/model_res/1/res.pkl"
tf_savepath    = "/Users/hyunwoogu/Documents/model_res/2/"
keras_savepath = "/Users/hyunwoogu/Documents/model_res/3/"

with open(pkl_savepath, 'wb') as f:
    pickle.dump(model, f) # pickle save
tf.saved_model.save(model, tf_savepath) # tf save
# model.save(keras_savepath)

# model.build(input_shape=(5,))

# load
with open(pkl_savepath,'rb') as f:
    pkl_model = pickle.load(f)
model = tf.saved_model.load(tf_savepath) # tf load

##
# for iter in range(10):
#     x = np.random.randint(10,size=5).astype(np.float32)
#     y = x
#     loss = pkl_model(x,y,hp)
#     print(loss.numpy())

for iter in range(10):
    x = np.random.randint(10,size=(2,5)).astype(np.float32)
    y = x.sum(axis=0)
    # x = np.random.randint(10,size=5).astype(np.float32)
    # y = x
    loss = pkl_model([tf.constant(x[i]) for i in range(2)],
             tf.constant(y), hp) # train!
    # loss = model(x,y,hp)
    print(loss.numpy())

hp['add_par'] = tf.constant(100000.) 

for iter in range(10):
    x = np.random.randint(10,size=(5,5)).astype(np.float32)
    y = x.sum(axis=0)
    # x = np.random.randint(10,size=5).astype(np.float32)
    # y = x
    loss = model(tf.constant(x, name="x"),
                 tf.constant(y, name="y"), hp) # train!

    # loss = model(x,y,hp)
    print(loss.numpy())

##
pkl_model.run_model(x, train_par)
model.run_model(x, train_par) # error 

train_par['add_par'] = tf.constant(2000.)

train_par = tf.Variable([200.])

#
model.add_ = model.add_ * 300000000


   

#
model.test_dict = {} 
model.test_dict['Singer'] = tf.Variable('Sam_Smith', trainable=False)





