import os, sys, yaml

import numpy as np
from scipy.interpolate import interp1d
import tensorflow as tf

from tensorboard.plugins.hparams import api as tf_hpapi

sys.path.append('../')
from base_model import BaseModel
from .rvnet_hyper import hp
import det_rnn.train as utils_train
import det_rnn.analysis as utils_analysis
from utils.general import get_logger, Progbar, export_plot

EPSILON = 1e-7

__all__ = ['RVNET']

"""Variational network with ring attractors (i.e. Ben-Yishai)"""

class RVNET(BaseModel):
    def __init__(self, hp, par_train,
                 hypsearch_dict=None,
                 dtype=tf.float32, logger=None):
        """
        hp: network hyperparameters
        par_train: stimulus parameters used to train the network (since efficient coding is dependent on stimulus distribution)
        """
        super(RVNET, self).__init__(hp, par_train,hypsearch_dict=hypsearch_dict,dtype=dtype, logger=logger)

    # all the networks to train goes here.
    def build(self):
        raise NotImplementedError

    ''' Train operations'''
    # todo: implement train; add schedulers?

    def evaluate(self, trial_info):
        # inputs and outputs from the trial
        neural_input    = trial_info['input_tuning']
        output_tuning   = trial_info['output_tuning']
        labels          = tf.math.argmax(output_tuning,1)

        raise NotImplementedError

        # should return
        # (1) a loss struct; should contain the final aggregated loss (unregularized); loss_struct['loss']
            # regularization loss for submodules are calculated via
        # (2) output struct (logits)
        return 0

    def calc_loss(self, labels, logits):
        raise NotImplementedError

    ''' UTILS '''
    def _initialize_variable(self, hp, par_train):
        raise NotImplementedError