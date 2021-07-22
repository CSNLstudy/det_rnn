import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class pRNN(tf.Module):
    def __init__(self,
                 nori,
                 rnnmat = None,
                 noise_fixed = 0,
                 normal_approx = True,
                 dtype=tf.float32):
        super(pRNN, self).__init__()
        self.dtype = dtype
        self.normal_approx = normal_approx
        self.nori = nori

        self.noise_fixed = noise_fixed

        if rnnmat is None:
            self.rnnmat = tf.Variable(tf.random.normal([self.nori, self.nori]),
                                      name='rnnmat', dtype=self.dtype)
        else:
            self.rnnmat = tf.Variable(rnnmat,
                                      name='rnnmat', dtype=self.dtype)


    def __call__(self,r, T):
        # r: (S, M, N)
        act = [r[:,:,:,None]]
        for t in range(T):
            act += [self.update(act[-1])]

        return tf.stack(act,axis=0) #(T, S, M, N, 1)


    def update(self,r):
        # r: (S, M, N, 1)
        (S,M,N,_)   = r.shape
        In          = self.rnnmat @ r
        mean_sp     = tf.reduce_mean(In,axis=1,keepdims=True)
        globalinh   = mean_sp - tf.reduce_max(mean_sp,axis=2,keepdims=True)  # global inhibition
        mean    = tf.repeat(globalinh, M, axis=1) # average over subpopulations
        newr    = self.poisson_activation(mean)
        return newr


    def poisson_activation(self, fr):
        if self.normal_approx:
            if self.noise_fixed > 0:
                # try only with fixed noise
                r = fr + tf.cast(self.noise_fixed,dtype=fr.dtype) * tf.random.normal(fr.shape, dtype=fr.dtype)
            else:
                r = fr + tf.sqrt(fr) * tf.random.normal(fr.shape, dtype=fr.dtype)
        else:
            r = tf.random.poisson([], fr, dtype=fr.dtype)

        # r = tf.nn.relu(r)

        return r