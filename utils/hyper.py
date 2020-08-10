import numpy as np
import itertools


def hyperparam_comb(hp_dict):
    """
    hp_dict = {'sensory_noise_type': ['Normal_fixed', 'Normal_poisson'],
               'sensory_repr'      : ['Efficient', 'Uniform'],
               'rnn_weights'        : ['sym_sensory', 'sym_stimulus']
               }
   """

    hp_names = []
    hp_lists = []
    hp_n     = []
    for (key, value) in hp_dict.items():
        hp_names += [key]
        hp_lists += [value]
        hp_n     += [len(value)]


    # check number of combinations (this can only be used once?)
    hypsearch = itertools.product(*hp_lists)



    return hypsearch, hp_names, hp_lists, hp_n