import os, sys, copy
import numpy as np

sys.path.append('../')
from models.vnet import rvnet_hp
from models.vnet.rvnet_model import RVNET
from det_rnn.base import par, update_parameters, Stimulus
from utils.general import get_logger
from utils.hyper import hyperparam_comb

experimentname  = 'mechanistic0'
root_dir        = os.getcwd() + os.path.sep + '..'  # this should be /det_rnn folder
basedir         = root_dir + os.path.sep + 'experiments' + os.path.sep + experimentname

if __name__ == '__main__':
    # list of parameters to change

    # training set
    par_train = copy.deepcopy(par)
    # use "natural prior"
    par_train['stim_dist'] = 'natural'
    par_train['n_ori'] = 1000  # make it somewhat continuous
    par_train['n_tuned_input'] = 100
    par_train['n_tuned_output'] = 100
    par_train = update_parameters(par_train)
    stim_train = Stimulus(par_train)

    # testing set; for testing the 50 orientations
    par_test = copy.deepcopy(par)
    par_test['batch_size'] = 50  # increase to test variance as well.
    par_test['stim_dist'] = 'uniform'
    par_test['n_ori'] = 50
    par_test['n_tuned_input'] = 100
    par_test['n_tuned_output'] = 100
    par_test = update_parameters(par_test)
    stim_test = Stimulus(par_test)  # the argument `par` may be omitted


    par_list = {}
    # dictionary of hp to test
    hp_dict = {'sensory_noise_type': ['Normal_fixed', 'Normal_poisson'],
               'sensory_repr'      : ['Efficient', 'Uniform'],
               'rnn_weights'        : ['sym_sensory', 'sym_stimulus']
               }

    # parameters to change #2
    #'sensory_input_ani': 0.01,  # input anisotropy epsilon (Ben-Yishai, 1995)
    #'sensory_input_gain': 2,  # or learn c (Ben-Yishai, 1995)
    #'rnn_weights_shift': 86,  # 'learn' or a number, J0 (Ben-Yishai, 1995)
    #'rnn_weights_scale': 112,  # 'learn' or a number, J2 (Ben-Yishai, 1995)

    hp_dict2 = {'sensory_input_ani': np.logspace(-2,3,10),
               'sensory_input_gain': np.logspace(-1,2,10),
               'rnn_weights_shift': np.logspace(1,3,10),
               'rnn_weights_scale': np.logspace(1,3,10)
               }

    # check number of combinations (this can only be used once?)
    hypsearch, hp_names, hp_lists, hp_n = hyperparam_comb(hp_dict)
    print('Number of hyperparam combinations: ' + str(len(par_list) * sum(1 for l in hypsearch)))

    iterN = 0
    print('Training model for each combination of parameters....')
    hypsearch, hp_name, hp_lists, hp_n = hyperparam_comb(hp_dict)
    for params in hypsearch:
        # set hyperparameters
        hp = rvnet_hp(par_train)

        hypsearch_dict = {}
        hpstr = ''

        for idx in range(len(params)):
            hp[hp_names[idx]] = params[idx]
            hypsearch_dict[hp_names[idx]] = params[idx]
            hpstr += hp_names[idx] + '=' + str(params[idx]) + ', '

            # set directories
            hp['output_path']   = basedir + os.path.sep + 'm' + str(iterN)
            hp['model_output']  = hp['output_path'] + os.path.sep + 'model'
            hp['log_path']      = hp['output_path'] + os.path.sep + 'log.txt'

            if not os.path.exists(hp['output_path']):
                os.makedirs(hp['output_path'])

            logger = get_logger(hp['log_path'])
            logger.info('::::: HP Combination ' + str(iterN) + '::::::')
            logger.info(hpstr)

            ####### run network with the given hyperparameters #########
            model = RVNET(hp, par_train, hypsearch_dict=hypsearch_dict, logger=logger)

            # plot output and bias, on test data.
            model.plot_summary(stim_test, filename=hp['output_path'] + os.path.sep + 'model')

            subparams, subp_names, subp_lists = hyperparam_comb(hp_dict2)
            for sp in subparams:
                for idx_sp in range(len(sp)):
                    hp[subp_names[idx]] = sp[idx]
                    model = RVNET(hp, par_train)
                    model.plot_summary(stim_test, filename=hp['output_path'] + os.path.sep + 'model')

            iterN += 1