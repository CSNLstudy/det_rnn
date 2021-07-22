import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.general import get_logger, Progbar, export_plot

class poisson_decoder(tf.Module):
    def __init__(self, dtype=tf.float32):
        super(poisson_decoder, self).__init__()
        self.dtype = dtype

    def decode(self, r, H0=None):
        (S, M, N) = r.shape
        H       = self.fit(r, H0 = H0, tol= 1e-5, lr=0.1)
        lp      = self.logprob_decode(r ,H) # (S,S1,M)
        ml      = tf.math.argmax(lp,axis=1) # orientation "index" (S,M)
        ml_rad  = ml.numpy() * (2*np.pi) / S # S orientations 0-2pi
        sinr    = np.mean(np.sin(ml_rad),axis=1)
        cosr    = np.mean(np.cos(ml_rad),axis=1)
        circvar = 1 - np.sqrt(np.square(sinr) + np.square(cosr))
        cirmean = np.arctan2(sinr,cosr)

        return cirmean, circvar, H, ml

    def fit(self,r, H0 = None, tol= 1e-5, lr=0.1):
        (S, M, N) = r.shape
        if H0 is None:
            Hs = np.zeros((N))
            H = tf.Variable(Hs, dtype=self.dtype)
        else:
            H = tf.Variable(H0, dtype=self.dtype)
        optimizer = tf.optimizers.Adam(lr)

        nlogprobs = []
        gradnorms = []
        niter = 1000
        prog = Progbar(target=niter)
        for iter in range(niter):
            with tf.GradientTape() as tape:
                nlogprob = -1 * tf.reduce_mean(self.logprob_unnorm(r ,H))
            grads = tape.gradient(nlogprob,H)

            gradnorms += [tf.linalg.global_norm([grads])]
            nlogprobs += [nlogprob]

            optimizer.apply_gradients(zip([grads], [H]))
            if iter > 0:
                checktol = tf.abs(nlogprobs[-1] - nlogprobs[-2])
            else:
                checktol = 1e10
            if iter > 0 and checktol < tol:
                break
            elif tf.math.is_nan(nlogprobs[-1]):
                break
            else:
                prog.update(iter, exact=[("nlogprob", nlogprobs[-1].numpy()), ("gradnorm",gradnorms[-1].numpy()),
                                         ("dloss", checktol)])

        return H

    def logprob_unnorm(self, r ,H):
        # r.shape = (S,M,N)
        # H.shape = (N)
        (S, M, N) = r.shape
        logprob = []
        for s in range(S):
            rolled = tf.roll(H,s,axis=0)
            logprob_s = tf.reduce_sum(rolled[None,:] * r[s,:,:],axis=-1)
            logprob_s -= tf.reduce_sum(tf.exp(rolled),axis=-1)
            # logprob_s -= tf.reduce_sum(tf.math.square(H))
            logprob += [logprob_s]

        return tf.stack(logprob,axis=0) # (S,M)


    def logprob_decode(self, r ,H):
        # decode stimulus s with kernel from s2
        (S, M, N) = r.shape
        logprob = []
        for s in range(S):
            logprob_s = []
            for s2 in range(S):
                rolled = tf.roll(H, s2, axis=0)
                logprob_s2 = tf.reduce_sum(rolled[None, :] * r[s, :, :], axis=-1)
                logprob_s2 -= tf.reduce_sum(tf.exp(rolled), axis=-1)
                logprob_s += [tf.stack(logprob_s2,axis=0)]
            logprob += [tf.stack(logprob_s,axis=0)]

        return tf.stack(logprob,axis=0) # output: (S,S1,M)



