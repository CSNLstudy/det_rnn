import numpy as np
import matplotlib.pyplot as plt
from ._parameters import par
import randomgen.generator as random

__all__ = ['Stimulus']

class Stimulus(object):
    """
    This script is dedicated to generating delayed estimation stimuli @ CSNL.
    """
    def __init__(self, par=par):
        self.set_params(par) # Equip the stimulus class with parameters
        self._generate_tuning() # Generate tuning/input config

    def set_params(self, par):
        for k, v in par.items():
            setattr(self, k, v)

    def generate_trial(self, balanced=False):
        # generate all the orientations to output in a trial
        if balanced is True:
            # if testing is true, generate stimulus evenly
            raise NotImplementedError # todo weave stimulus and reference
            repeatN = np.ceil(self.batch_size/(self.n_ori*len(self.reference))) # repeat to somewhat match batch size.
            stimulus_ori = np.repeat(np.arange(self.n_ori), repeatN) # todo: make the testing trial generation more elegant...
            reference_ori = np.repeat(self.reference, repeatN)
        else:
            stimulus_ori = np.random.choice(np.arange(self.n_ori), p=self.stim_p, size=self.batch_size)
            reference_ori = np.random.choice(self.reference, p=self.ref_p, size=self.batch_size)

        if (type(self.kappa) is str) and(self.kappa is 'dist'):
            stimulus_kap = np.random.gamma(shape = self.kappa_dist_shape, scale = self.kappa_dist_scale, size = stimulus_ori.shape)
        else:
            stimulus_kap = self.kappa * np.ones(stimulus_ori.shape)

        # josh: return by calling the generated class
        # => makes trial_info struct compatible with previous versions of the code.
        return Trial(stim=self,
                     stimulus_ori=stimulus_ori,
                     stimulus_kap=stimulus_kap,
                     reference_ori=reference_ori)()

    def _generate_tuning(self):
        _tuning_input = np.zeros((self.n_tuned_input, self.n_receptive_fields, self.n_ori))
        _tuning_output = np.zeros((self.n_tuned_output, self.n_receptive_fields, self.n_ori))
        stim_dirs = np.float32(np.arange(0, 180, 180 / self.n_ori))
        pref_dirs = np.float32(np.arange(0, 180, 180 / (self.n_tuned_input)))
        for n in range(self.n_tuned_input):
            for i in range(self.n_ori):
                d = np.cos((stim_dirs[i] - pref_dirs[n]) / 90 * np.pi)
                _tuning_input[n, 0, i] = self.strength_input * np.exp(self.kappa * d) / np.exp(self.kappa)
                if self.resp_decoding == 'onehot':
                    _tuning_output[n, 0, i] = self.strength_output * (1. * (d == 1.))
                else:
                    _tuning_output[n, 0, i] = self.strength_output * np.exp(self.kappa * d) / np.exp(self.kappa)

        if self.stim_encoding == 'single':
            self.tuning_input = _tuning_input

        elif self.stim_encoding == 'double':
            self.tuning_input = np.tile(_tuning_input, (2, 1))

        self.tuning_output = _tuning_output

        self.stim_dirs = stim_dirs
        self.pref_dirs = pref_dirs

    """
    def _gen_stim(self, stimulus_ori):
        # TODO(HG): need to be changed if n_ori =/= n_tuned
        neural_input = random.standard_normal(size=(stim.n_timesteps, self.batch_size, self.n_input))*self.noise_sd + self.noise_mean
        neural_input[:,:,:self.n_rule_input] += self._gen_input_rule()
        for t in range(self.batch_size):
            neural_input[self.design_rg['stim'],t,self.n_rule_input:] += self.tuning_input[:,0,stimulus_ori[t]].reshape((1,-1))
        if self.n_subblock > 1: # multi-trial settings
            neural_input = neural_input.transpose((1,0,2)).\
                reshape((self.n_subblock,-1,self.n_input)).transpose((1,0,2))
        return neural_input

    def _gen_output(self, stimulus_ori):
        desired_output = np.zeros((stim.n_timesteps,self.batch_size,self.n_output), dtype=np.float32)
        desired_output[:, :, :self.n_rule_output] = self._gen_output_rule()
        for t in range(self.batch_size):
            if self.resp_decoding == 'conti':
                desired_output[self.output_rg, t, self.n_rule_output:] = stimulus_ori[t] * np.pi / np.float32(self.n_tuned_output)
            elif self.resp_decoding in ['disc', 'onehot']:
                desired_output[self.output_rg, t, self.n_rule_output:] = self.tuning_output[:, 0, stimulus_ori[t]].reshape((1, -1))
        if self.n_subblock > 1: # multi-trial settings
            desired_output = desired_output.transpose((1,0,2)).\
                reshape((self.n_subblock,-1,self.n_output)).transpose((1,0,2))
        return desired_output

    def _gen_mask(self):
        mask = np.zeros((stim.n_timesteps, self.batch_size, self.n_output), dtype=np.float32)
        # set "specific" period
        for step in ['iti','stim','delay','estim']:
            mask[self.design_rg[step], :, self.n_rule_output:] = self.mask[step]
            mask[self.design_rg[step], :, :self.n_rule_output] = self.mask['rule_'+step]
        # set "globally dead" period
        mask[self.dead_rg, :, :] = 0
        if self.n_subblock > 1: # multi-trial settings
            mask = mask.transpose((1,0,2)).\
                reshape((self.n_subblock,-1,self.n_output)).transpose((1,0,2))
        return mask

    def _gen_input_rule(self):
        if self.n_rule_input == 0:
            return np.array([]).reshape((stim.n_timesteps,self.batch_size,0))

        else:
            rule_mat = np.zeros([stim.n_timesteps, self.batch_size, self.n_rule_input])
            for i,k in enumerate(self.input_rule_rg):
                rule_mat[self.input_rule_rg[k], :, i] = self.input_rule_strength
            return rule_mat

    def _gen_output_rule(self):
        if self.n_rule_output == 0:
            return np.array([]).reshape((stim.n_timesteps,self.batch_size,0))

        else:
            rule_mat = np.zeros([stim.n_timesteps, self.batch_size, self.n_rule_output])
            for i,k in enumerate(self.output_rule_rg):
                rule_mat[self.output_rule_rg[k], :, i] = self.output_rule_strength
            return rule_mat
    """

class Trial(object):
    """
    make a Trial class in order to change the trial structures more flexible (i.e. balanced data set)
    """
    def __init__(self, stim, stimulus_ori, stimulus_kap, reference_ori):
        # attributes that need to be flexible
        self.stimulus_ori   = stimulus_ori
        self.reference_ori  = reference_ori
        self.stimulus_kap   = stimulus_kap
        self.batch_size     = stimulus_ori.shape[0]
        self.n_subblock     = int(self.batch_size / stim.trial_per_subblock)
        [self.neural_input, self.input_tuning, self.ref_neuron] = self._gen_stim(stim, reference_ori, stimulus_ori,
                                                                         stimulus_kap)
        [self.desired_output, self.output_tuning] = self._gen_output(stim, reference_ori, stimulus_ori, stimulus_kap)
        self.mask           = self._gen_mask(stim)

    def __call__(self):
        return {'neural_input'  : self.neural_input.astype(np.float32),
                'stimulus_ori'  : self.stimulus_ori,
                'reference_ori' : self.reference_ori,
                'ref_neuron'    : self.ref_neuron,
                'desired_decision': self.desired_output['decision'].astype(np.float32),
                'desired_estim' : self.desired_output['estim'].astype(np.float32),
                'mask_decision' : self.mask['decision'].astype(np.float32),
                'mask_estim'    : self.mask['estim'].astype(np.float32),
                'input_tuning'  : self.input_tuning,
                'output_tuning' : self.output_tuning,
                'stimulus_kap'  : self.stimulus_kap}

    def _gen_stim(self, stim, reference_ori, stimulus_ori, stimulus_kap):
        assert stimulus_ori.shape == stimulus_kap.shape

        # initialize neural input with noise, and stimulus tuning without noise
        neural_input = random.standard_normal(size=(stim.n_timesteps, self.batch_size, stim.n_input))*stim.noise_sd + stim.noise_mean
        neural_input[:,:,:stim.n_rule_input] += self._gen_input_rule(stim) # add rules

        # stimulus directions and preferred directions
        # Compute vonMises in the 2*theta space todo: write a function for vonMises?
        stim_dirs = np.tile(np.float32(np.arange(0, 180, 180 / stim.n_ori)).reshape(-1,1), [1, stim.n_tuned_input]) # stim_dirs x n_tuned_input
        pref_dirs_in    = np.tile(np.float32(np.arange(0, 180, 180 / stim.n_tuned_input)).reshape(1,-1), [self.batch_size, 1]) # B x n_tuned_input
        din             = np.cos((stim_dirs[stimulus_ori,:] - pref_dirs_in) / 90 * np.pi)      # B x n_tuned_input
        input_tuning    = np.exp(stimulus_kap.reshape(-1,1)*din)/np.exp(stimulus_kap.reshape(-1,1))  # broadcasted to B x n_tuned_input
        input_tuning    = input_tuning/np.sum(input_tuning,1,keepdims = True)

        ref_din         = np.cos((stim_dirs[(stimulus_ori + reference_ori) % stim.n_ori,:] - pref_dirs_in) / 90 * np.pi)      # B x n_tuned_input
        # B x n_tuned_input
        ref_neuron      = np.argmax(ref_din,axis=1)

        for b in range(self.batch_size):
            neural_input[stim.design_rg['stim'],b,stim.n_rule_input:] += \
                np.tile(input_tuning[b, :], [stim.design_rg['stim'].shape[0], 1])

            # make reference neuron
            neural_input[stim.design_rg['decision'],b, ref_neuron[b]] += stim.strength_ref

        if self.n_subblock > 1: # multi-trial settings #josh: i.e. concatenate runs in the sublock ??? todo: check this...
            neural_input = neural_input.transpose((1,0,2)).reshape((self.n_subblock,-1,stim.n_input)).transpose((1,0,2))

        return neural_input, input_tuning, ref_neuron


    def _gen_output(self, stim, reference_ori, stimulus_ori, stimulus_kap):
        desired_decision = np.zeros((stim.n_timesteps,self.batch_size,stim.n_output_dm), dtype=np.float32)
        desired_estim    = np.zeros((stim.n_timesteps,self.batch_size,stim.n_output_em), dtype=np.float32)
        desired_decision[:, :, :stim.n_rule_output_dm] = self._gen_output_rule(stim, type = 'decision')
        desired_estim[:, :, :stim.n_rule_output_em]    = self._gen_output_rule(stim, type = 'estim')

        stim_dirs_out = np.tile(np.float32(np.arange(0, 180, 180 / stim.n_ori)).reshape(-1, 1),
                                [1, stim.n_tuned_output])  # stim_dirs x n_tuned_output
        pref_dirs_out   = np.tile(np.float32(np.arange(0, 180, 180 / stim.n_tuned_output)).reshape(1,-1), [self.batch_size, 1])  # B x n_tuned_output
        dout            = np.cos((stim_dirs_out[stimulus_ori,:] - pref_dirs_out) / 90 * np.pi)        # B x n_output

        output_tuning   = np.exp(stimulus_kap.reshape(-1,1)*dout)/np.exp(stimulus_kap.reshape(-1,1))  # broadcasted to B x n_tuned_output
        output_tuning   = output_tuning/np.sum(output_tuning,1,keepdims = True)

        for b in range(self.batch_size):
            if reference_ori[b] == 0:
                if np.random.random() > 0:
                    desired_decision[stim.dm_output_rg, b, stim.n_rule_output_dm + 1] += 1
                else:
                    desired_decision[stim.dm_output_rg, b, stim.n_rule_output_dm] += 1
            else:
                desired_decision[stim.dm_output_rg, b, stim.n_rule_output_dm + (0 < reference_ori[b])] += 1

            if stim.resp_decoding == 'conti':
                desired_estim[stim.em_output_rg, b, stim.n_rule_output_em:] = stimulus_ori[b] * np.pi / np.float32(
                    stim.n_tuned_output)
            elif stim.resp_decoding in ['disc', 'onehot']:
                desired_estim[stim.em_output_rg, b, stim.n_rule_output_em:] = \
                    np.tile(output_tuning[b,:],[stim.design_rg['estim'].shape[0],1])

        if self.n_subblock > 1: # multi-trial settings
            desired_decision = desired_decision.transpose((1,0,2)).reshape((self.n_subblock,-1,stim.n_output_dm)).transpose((1,0,2))
            desired_estim    = desired_estim.transpose((1,0,2)).reshape((self.n_subblock,-1,stim.n_output_em)).transpose((1,0,2))


        return {'decision' : desired_decision, 'estim' : desired_estim}, output_tuning

    """
    def _gen_output(self, stim, stimulus_ori):
        desired_output = np.zeros((stim.n_timesteps,self.batch_size,stim.n_output), dtype=np.float32)
        desired_output[:, :, :stim.n_rule_output] = self._gen_output_rule(stim)
        for t in range(self.batch_size):
            if stim.resp_decoding == 'conti':
                desired_output[stim.output_rg, t, stim.n_rule_output:] = stimulus_ori[t] * np.pi / np.float32(stim.n_tuned_output)
            elif stim.resp_decoding in ['disc', 'onehot']:
                desired_output[stim.output_rg, t, stim.n_rule_output:] = stim.tuning_output[:, 0, stimulus_ori[t]].reshape((1, -1))
        if self.n_subblock > 1: # multi-trial settings

        return desired_output
    """

    def _gen_mask(self, stim):
        mask_decision = np.zeros((stim.n_timesteps, self.batch_size, stim.n_output_dm), dtype=np.float32)
        mask_estim    = np.zeros((stim.n_timesteps, self.batch_size, stim.n_output_em), dtype=np.float32)
        # set "specific" period
        for step in ['iti','stim','delay','decision','estim']:
            mask_decision[stim.design_rg[step], :, stim.n_rule_output_dm:] = stim.mask_dm[step]
            mask_decision[stim.design_rg[step], :, :stim.n_rule_output_dm] = stim.mask_dm['rule_'+step]
            mask_estim[stim.design_rg[step], :, stim.n_rule_output_em:] = stim.mask_em[step]
            mask_estim[stim.design_rg[step], :, :stim.n_rule_output_em] = stim.mask_em['rule_' + step]
        # set "globally dead" period
        mask_decision[stim.dead_rg, :, :] = 0
        mask_estim[stim.dead_rg, :, :] = 0
        if self.n_subblock > 1: # multi-trial settings
            mask_decision = mask_decision.transpose((1,0,2)).reshape((self.n_subblock,-1,stim.n_output_dm)).transpose((1,0,2))
            mask_estim = mask_estim.transpose((1,0,2)).reshape((self.n_subblock,-1,stim.n_output_em)).transpose((1,0,2))
        return {'decision' : mask_decision, 'estim' : mask_estim}


    def _gen_input_rule(self, stim):
        if stim.n_rule_input == 0:
            return np.array([]).reshape((stim.n_timesteps,self.batch_size,0))
        else:
            rule_mat = np.zeros([stim.n_timesteps, self.batch_size, stim.n_rule_input])
            for i,k in enumerate(stim.input_rule_rg):
                rule_mat[stim.input_rule_rg[k], :, i] = stim.input_rule_strength
            return rule_mat

    def _gen_output_rule(self, stim, type):
        if type == 'estim':
            if stim.n_rule_output_em == 0:
                return np.array([]).reshape((stim.n_timesteps,self.batch_size,0))
            else:
                rule_mat = np.zeros([stim.n_timesteps, self.batch_size, stim.n_rule_output_em])
                for i,k in enumerate(stim.output_em_rule_rg):
                    rule_mat[stim.output_em_rule_rg[k], :, i] = stim.output_em_rule_strength
        elif type == 'decision':
            if stim.n_rule_output_dm == 0:
                return np.array([]).reshape((stim.n_timesteps, self.batch_size, 0))
            else:
                rule_mat = np.zeros([stim.n_timesteps, self.batch_size, stim.n_rule_output_dm])
                for i, k in enumerate(stim.output_dm_rule_rg):
                    rule_mat[stim.output_dm_rule_rg[k], :, i] = stim.output_dm_rule_strength
        return rule_mat