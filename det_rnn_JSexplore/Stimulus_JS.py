#%%
import os, shutil, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# %%
nS = 8

def roll_2d(A, r):
    rows, column_indices = np.ogrid[:A.shape[0], :A.shape[1]]
    r[r < 0] += A.shape[1]
    column_indices = column_indices - r[:, np.newaxis]
    result = A[rows, column_indices]
    return result

#%%
class Stimulus:
    def __init__(self, nS = 8, nBatch = 10):
        self.nS = nS
        self.nBatch = nBatch
        self.cue_position = 1

        basis = np.zeros(nS)
        basis[[0, 1, -1, 2, -2]] = np.array([2,1,1, 0.5, 0.5])
        self.basis = basis

        self.steps_input1 = 10
        self.steps_delay1 = 50
        self.steps_input2 = 10
        self.steps_delay2 = 50
        self.steps_pre = 10

        self.steps_retrieve = 10
        self.update_time()

    def update_time(self):
        self.nStep = self.steps_pre + self.steps_input1 + self.steps_delay1  +  self.steps_input2 + self.steps_delay2 + self.steps_retrieve
        self.time_cue1 = self.steps_pre + np.arange(self.steps_input1)
        self.time_cue2  = (self.steps_pre + self.steps_input1 + self.steps_delay1) + np.arange(self.steps_input2)
        self.time_retrieve = (self.nStep - self.steps_retrieve) + np.arange(self.steps_retrieve)

        # self.time_task = np.concatenate([self.time_cue1, self.time_cue2, self.time_retrieve])
        # self.time_fixation = [s for s in np.arange(self.nStep) if np.in1d(s, self.task_time)]


    def ori_stim(self, oris):
        X = np.tile(self.basis, [len(oris),1])
        return roll_2d(X, np.array(oris))


    def ori_stim2(self, ori1, ori2):
        X = np.concatenate(
        [np.zeros(shape = [self.steps_pre, self.nBatch, self.nS]),
        np.tile(self.ori_stim(ori1), [self.steps_input1,1,1]),
        np.zeros(shape = [self.steps_delay1, self.nBatch, self.nS]),
        np.tile(self.ori_stim(ori2), [self.steps_input2,1,1]),
        np.zeros(shape = [self.steps_delay2+self.steps_retrieve, self.nBatch, self.nS])],
        axis = 0)

        return X


    def make_cue(self):
        cue_position = self.cue_position

        cue_vect = np.zeros((self.nStep , self.nBatch, 3))
        if cue_position == 1:
            cue_vect[self.time_cue1, :, 0] = 1
        elif cue_position == 2:
            cue_vect[self.time_cue2, :, 0] = 1

        cue_vect[self.time_retrieve, :, 2] = 1
        a = np.sum(cue_vect, axis = (1,2))==0
        cue_vect[a, :,1] = 1

        return cue_vect


    def generate_stimulus(self, ori1 = None, ori2 = None):
        
        if ori1 == None: # if ori1, ori2 are not given, they are generatred
            self.ori1 = np.random.randint(0, self.nS, self.nBatch)
        else:
            self.ori1 = ori1
            
        if ori2 == None:
            self.ori2 = np.random.randint(0, self.nS, self.nBatch)
        else:
            self.ori2 = ori2
        
        cue_vect = self.make_cue()
        X = self.ori_stim2(self.ori1, self.ori2)
        y = np.concatenate([cue_vect, X], axis = -1)

        return y


    def generate_output(self):
        
        x = np.zeros((self.nBatch, self.nS))
        x[:, np.argmax(self.basis)] = 1

        if self.cue_position==1:
            x2 = roll_2d(x, self.ori1)
        elif self.cue_position==2:
            x2 = roll_2d(x, self.ori2)

        y = np.zeros((self.nStep, self.nBatch,self.nS+1))
        y[:-self.steps_retrieve,:,0] = 1
        y[self.time_retrieve,:,1:] = x2

        return y

#%% Predefined functions

def Predefined_Stim(Version = 'basic', n_batch = 100, nS = 8):

    if Version == 'basic':
        Stim = Stimulus(nS = nS, nBatch = n_batch)
        Stim.steps_input2 = 0
        Stim.update_time()
        gen_stim = Stim.generate_stimulus
        gen_output = Stim.generate_output

    elif Version == 'distractor':
        Stim = Stimulus(nS = nS, nBatch = n_batch)
        Stim.steps_input2 = 10
        Stim.update_time()
        gen_stim = Stim.generate_stimulus
        gen_output = Stim.generate_output

    elif Version == 'select':
        hf = int(n_batch/2)
        Stim1 = Stimulus(nS = nS, nBatch = hf)
        Stim1.cue_position = 1

        Stim2 = Stimulus(nS = nS, nBatch = n_batch-hf)
        Stim2.cue_position = 2

        def gen_stim():
            y = np.concatenate(
                [Stim1.generate_stimulus(),
                Stim2.generate_stimulus()], axis = 1)
            return y

        def gen_output():
            y = np.concatenate(
                [Stim1.generate_output(),
                Stim2.generate_output()], axis = 1)
            return y

    elif Version == 'basic_jitter':
        hf = int(n_batch/2)
        Stim1 = Stimulus(nS = nS, nBatch = hf)
        Stim1.steps_pre = 10
        Stim1.steps_delay1 = 50
        Stim1.steps_input2 = 0
        Stim1.update_time()

        Stim2 = Stimulus(nS = nS, nBatch = n_batch-hf)
        Stim2.steps_pre = 50
        Stim2.steps_delay1 = 10
        Stim2.steps_input2 = 0
        Stim2.update_time()

        def gen_stim():
            y = np.concatenate(
                [Stim1.generate_stimulus(),
                Stim2.generate_stimulus()], axis = 1)
            return y

        def gen_output():
            y = np.concatenate(
                [Stim1.generate_output(),
                Stim2.generate_output()], axis = 1)
            return y

    return gen_stim, gen_output


