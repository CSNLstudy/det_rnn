import numpy as np
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

    def generate_trial(self):
        stimulus                     = self._gen_stimseq()
        neural_input1, neural_input2 = self._gen_stims(stimulus)
        desired_output               = self._gen_output(stimulus)
        mask                         = self._gen_mask()
        return {'neural_input1'   : neural_input1.astype(np.float32),
                'neural_input2'   : neural_input2.astype(np.float32),
                'stimulus_ori'    : stimulus['stimulus_ori'],
                'reference_ori'   : stimulus['reference_ori'],
                'desired_decision': desired_output['decision'].astype(np.float32),
                'desired_estim'   : desired_output['estim'].astype(np.float32),
                'mask_decision'   : mask['decision'].astype(np.float32),
                'mask_estim'      : mask['estim'].astype(np.float32)}

    def _gen_stimseq(self):
        stimulus_ori  = np.random.choice(np.arange(self.n_ori), p=self.stim_p, size=self.batch_size)
        reference_ori = np.random.choice(self.reference, p=self.ref_p, size=self.batch_size)
        return {'stimulus_ori': stimulus_ori, 'reference_ori': reference_ori}

    def _gen_stims(self, stimulus):
        neural_input1 = random.standard_normal(size=(self.n_timesteps, self.batch_size, self.n_input))*self.noise_sd*np.sqrt(2*self.tau/self.dt)
        neural_input2 = random.standard_normal(size=(self.n_timesteps, self.batch_size, self.n_input))*self.noise_sd*np.sqrt(2*self.tau/self.dt)

        # neural_input1[:,:,:self.n_rule_input]  = 0   # noise to rule set to be 0
        # neural_input1[:,:,:self.n_rule_input] += self._gen_input_rule()
        # neural_input2[:,:,:self.n_rule_input]  = 0   # noise to rule set to be 0
        # neural_input2[:,:,:self.n_rule_input] += self._gen_input_rule()

        for t in range(self.batch_size):
            neural_input2[self.design_rg['stim'],t,self.n_rule_input:] += self.tuning_input[:,0,stimulus['stimulus_ori'][t]].reshape((1,-1))
            neural_input1[self.design_rg['decision'],t,self.n_rule_input+(stimulus['stimulus_ori'][t]+stimulus['reference_ori'][t])%self.n_ori] += self.strength_ref

        return neural_input1, neural_input2

    def _gen_output(self, stimulus):
        desired_decision = np.zeros((self.n_timesteps,self.batch_size,self.n_output_dm), dtype=np.float32)
        desired_estim    = np.zeros((self.n_timesteps,self.batch_size,self.n_output_em), dtype=np.float32)
        desired_decision[:, :, :self.n_rule_output_dm] = self._gen_output_rule('decision')
        desired_estim[:, :, :self.n_rule_output_em]    = self._gen_output_rule('estim')
        for t in range(self.batch_size):
            desired_decision[self.dm_output_rg, t, self.n_rule_output_dm + (0 < stimulus['reference_ori'][t])] += self.strength_decision
            if self.resp_decoding == 'conti':
                desired_estim[self.em_output_rg, t, self.n_rule_output_em:] = stimulus['stimulus_ori'][t] * np.pi / np.float32(self.n_tuned_output)
            elif self.resp_decoding in ['disc', 'onehot']:
                desired_estim[self.em_output_rg, t, self.n_rule_output_em:] = self.tuning_output[:, 0, stimulus['stimulus_ori'][t]].reshape((1, -1))
        if self.n_subblock > 1: # multi-trial settings
            desired_decision = desired_decision.transpose((1,0,2)).reshape((self.n_subblock,-1,self.n_output_dm)).transpose((1,0,2))
            desired_estim    = desired_estim.transpose((1,0,2)).reshape((self.n_subblock,-1,self.n_output_em)).transpose((1,0,2))
        return {'decision' : desired_decision, 'estim' : desired_estim}

    def _gen_mask(self):
        mask_decision = np.zeros((self.n_timesteps, self.batch_size, self.n_output_dm), dtype=np.float32)
        mask_estim    = np.zeros((self.n_timesteps, self.batch_size, self.n_output_em), dtype=np.float32)
        # set "specific" period
        for step in ['iti','stim','delay','decision','estim']:
            mask_decision[self.design_rg[step], :, self.n_rule_output_dm:] = self.mask_dm[step]
            mask_decision[self.design_rg[step], :, :self.n_rule_output_dm] = self.mask_dm['rule_'+step]
            mask_estim[self.design_rg[step], :, self.n_rule_output_em:] = self.mask_em[step]
            mask_estim[self.design_rg[step], :, :self.n_rule_output_em] = self.mask_em['rule_' + step]
        # set "globally dead" period
        mask_decision[self.dead_rg, :, :] = 0
        mask_estim[self.dead_rg, :, :] = 0
        if self.n_subblock > 1: # multi-trial settings
            mask_decision = mask_decision.transpose((1,0,2)).reshape((self.n_subblock,-1,self.n_output_dm)).transpose((1,0,2))
            mask_estim = mask_estim.transpose((1,0,2)).reshape((self.n_subblock,-1,self.n_output_em)).transpose((1,0,2))
        return {'decision' : mask_decision, 'estim' : mask_estim}

    def _gen_input_rule(self):
        if self.n_rule_input == 0:
            return np.array([]).reshape((self.n_timesteps,self.batch_size,0))
        else:
            rule_mat = np.zeros([self.n_timesteps, self.batch_size, self.n_rule_input])
            for i,k in enumerate(self.input_rule_rg):
                rule_mat[self.input_rule_rg[k], :, i] = self.input_rule_strength
            return rule_mat

    def _gen_output_rule(self, type):
        if type == 'estim':
            if self.n_rule_output_em == 0: return np.array([]).reshape((self.n_timesteps,self.batch_size,0))
            else:
                rule_mat = np.zeros([self.n_timesteps, self.batch_size, self.n_rule_output_em])
                for i,k in enumerate(self.output_em_rule_rg):
                    rule_mat[self.output_em_rule_rg[k], :, i] = self.output_em_rule_strength
        elif type == 'decision':
            if self.n_rule_output_dm == 0: return np.array([]).reshape((self.n_timesteps, self.batch_size, 0))
            else:
                rule_mat = np.zeros([self.n_timesteps, self.batch_size, self.n_rule_output_dm])
                for i, k in enumerate(self.output_dm_rule_rg):
                    rule_mat[self.output_dm_rule_rg[k], :, i] = self.output_dm_rule_strength
        return rule_mat

    def _generate_tuning(self):
        _tuning_input  = np.zeros((self.n_tuned_input,  self.n_receptive_fields, self.n_ori))
        _tuning_output = np.zeros((self.n_tuned_output, self.n_receptive_fields, self.n_ori))
        stim_dirs = np.float32(np.arange(0,180,180/self.n_ori))
        pref_dirs = np.float32(np.arange(0,180,180/(self.n_ori)))
        for n in range(self.n_tuned_input):
            for i in range(self.n_ori):
                d = np.cos((stim_dirs[i] - pref_dirs[n])/90*np.pi)
                _tuning_input[n,0,i]  = self.strength_input*np.exp(self.kappa*d)/np.exp(self.kappa)
                if self.resp_decoding == 'onehot':
                    _tuning_output[n,0,i] = self.strength_output*(1.*(d==1.))
                else:
                    _tuning_output[n,0,i] = self.strength_output*np.exp(self.kappa*d)/np.exp(self.kappa)

        if self.stim_encoding == 'single':
            self.tuning_input  = _tuning_input

        elif self.stim_encoding == 'double':
            self.tuning_input = np.tile(_tuning_input,(2,1))

        self.tuning_output = _tuning_output


