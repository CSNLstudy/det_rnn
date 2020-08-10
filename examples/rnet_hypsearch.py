import os, time, sys, copy
import numpy as np
import tensorflow as tf
import itertools

sys.path.append('../')
from vnet.rvnet_hyper import rvnet_hp
from vnet.rvnet_model import RVNET
from det_rnn.base import par, update_parameters, Stimulus
from utils.general import get_logger

experimentname  = 'efficient_coding4'
root_dir        = os.getcwd() + os.path.sep + '..'  # this should be /det_rnn folder
basedir         = root_dir + os.path.sep + 'experiments' + os.path.sep + experimentname

if __name__ == '__main__':
    # list of parameters to change
    par_names = ['kappa', 'kappa_dist_shape', 'kappa_dist_scale']
    par_list  = [(2,0,0), ('dist', 2, 1)]

    # dictionary of hp to test
    hp_dict = {'n_sensory'          : [25, 50],
                'sensory_noise_type': ['Normal_learn', 'Normal_fixed', 'Normal_poisson'],
                'sensory_repr'      : ['Learn', 'Efficient', 'Uniform']}

    hp_names = []
    hp_lists = []
    for (key, value) in hp_dict.items():
        hp_names += [key]
        hp_lists += [value]

    # check number of combinations (this can only be used once?)
    hypsearch = itertools.product(*hp_lists)
    print('Number of hyperparam combinations: ' + str(len(par_list) * sum(1 for l in hypsearch)))

    iterN = 0
    print('Training model for each combination of parameters....')
    hypsearch = itertools.product(*hp_lists)
    for params in hypsearch:
        # set hyperparameters
        hp = rvnet_hp(par)
        hypsearch_dict = {}
        hpstr = ''
        for idx in range(len(params)):
            hp[hp_names[idx]] = params[idx]
            hypsearch_dict[hp_names[idx]] = params[idx]
            hpstr += hp_names[idx] + '=' + str(params[idx]) + ', '

        for pars in par_list:
            # set stimulus parameters
            parstr = ''
            for par_idx in range(len(pars)):
                par[par_names[par_idx]] = pars[par_idx]
                hypsearch_dict[par_names[par_idx]] = pars[par_idx]
                parstr += par_names[par_idx] + '=' + str(pars[par_idx]) + ', '
            print(parstr)

            # set directories
            hp['output_path']   = basedir + os.path.sep + 'm' + str(iterN)
            hp['model_output']  = hp['output_path'] + os.path.sep + 'model'
            hp['log_path']      = hp['output_path'] + os.path.sep + 'log.txt'

            if not os.path.exists(hp['output_path']):
                os.makedirs(hp['output_path'])

            logger = get_logger(hp['log_path'])
            logger.info('::::: HP Combination ' + str(iterN) + '::::::')
            logger.info(parstr)
            logger.info(hpstr)

            # training set
            par_train                   = copy.deepcopy(par)
            # use "natural prior"
            par_train['stim_dist']      = 'natural'
            par_train['natural_a']      = 2e-4
            par_train['n_ori']          = 1000  # make it somewhat continuous
            #par_train['n_tuned_input']  = 50
            #par_train['n_tuned_output'] = 24
            par_train                   = update_parameters(par_train)
            stim_train                  = Stimulus(par_train)

            # testing set; for testing the 24 orientations
            par_test                    = copy.deepcopy(par)
            par_test['batch_size']      = 1000 # increase to test variance as well.
            par_test['stim_dist']       = 'uniform'
            par_test['n_ori']           = 50
            #par_test['n_tuned_input']   = 50, need to change these
            #par_train['n_tuned_output'] = 24
            par_test                    = update_parameters(par_test)
            stim_test                   = Stimulus(par_test)  # the argument `par` may be omitted

            ####### train network with the given hyperparameters #########
            model = VNET(hp, par_train, hypsearch_dict = hypsearch_dict, logger = logger)
            model.train(stim_train = stim_train, stim_test= stim_test)

            iterN += 1