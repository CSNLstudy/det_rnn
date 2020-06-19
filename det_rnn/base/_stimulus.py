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
        stimulus_ori    = self._gen_stimseq()
        neural_input    = self._gen_stim(stimulus_ori)
        desired_output  = self._gen_output(stimulus_ori)
        mask            = self._gen_mask()
        if self.n_subblock > 1: # multi-trial settings
            neural_input = neural_input.transpose((1,0,2)).\
                reshape((self.n_subblock,-1,self.n_input)).transpose((1,0,2))
            desired_output = desired_output.transpose((1,0,2)).\
                reshape((self.n_subblock,-1,self.n_output)).transpose((1,0,2))
            mask = mask.transpose((1,0,2)).\
                reshape((self.n_subblock,-1,self.n_output)).transpose((1,0,2))
        return {'neural_input'  : neural_input.astype(np.float32),
                'stimulus_ori'  : stimulus_ori,
                'desired_output': desired_output.astype(np.float32),
                'mask'          : mask}

    # TODO(HG): simplify here (Make n_ori flexible!!!)
    def _generate_tuning(self):
        _tuning_input  = np.zeros((self.n_tuned_input,  self.n_receptive_fields, self.n_ori))
        _tuning_output = np.zeros((self.n_tuned_output, self.n_receptive_fields, self.n_ori))
        stim_dirs = np.float32(np.arange(0,180,180/self.n_ori))
        pref_dirs = np.float32(np.arange(0,180,180/(self.n_ori)))

        for n in range(self.n_tuned_input):
            for i in range(self.n_ori):
                d = np.cos((stim_dirs[i] - pref_dirs[n])/90*np.pi)
                _tuning_input[n,0,i]  = self.strength_input*np.exp(self.kappa*d)/np.exp(self.kappa)
                _tuning_output[n,0,i] = self.strength_output*np.exp(self.kappa*d)/np.exp(self.kappa)

        if self.stim_encoding == 'single':
            self.tuning_input  = _tuning_input

        elif self.stim_encoding == 'double':
            self.tuning_input = np.tile(_tuning_input,(2,1))

        self.tuning_output = _tuning_output


    def _gen_stimseq(self):
        stimulus_ori = np.random.choice(np.arange(self.n_ori), p=self.stim_p, size=self.batch_size)
        return stimulus_ori

    def _gen_stim(self, stimulus_ori):
        # TODO(HG): need to be changed if n_ori =/= n_tuned
        neural_input = random.standard_normal(size=(self.n_timesteps, self.batch_size, self.n_input))*self.noise_sd + self.noise_mean
        neural_input[:,:,:self.n_rule_input] += self._gen_input_rule()
        for t in range(self.batch_size):
            neural_input[self.design_rg['stim'],t,self.n_rule_input:] += self.tuning_input[:,0,stimulus_ori[t]].reshape((1,-1))
        return neural_input

    def _gen_output(self, stimulus_ori):
        desired_output = np.zeros((self.n_timesteps,self.batch_size,self.n_output), dtype=np.float32)
        desired_output[:, :, :self.n_rule_output] = self._gen_output_rule()
        for t in range(self.batch_size):
            if self.resp_decoding == 'conti':
                desired_output[self.output_rg, t, self.n_rule_output:] = stimulus_ori[t] * np.pi / np.float32(self.n_tuned_output)
            elif self.resp_decoding == 'disc':
                desired_output[self.output_rg, t, self.n_rule_output:] = self.tuning_output[:, 0, stimulus_ori[t]].reshape((1, -1))
        return desired_output

    def _gen_mask(self):
        mask = np.zeros((self.n_timesteps, self.batch_size, self.n_output), dtype=np.float32)
        # set "specific" period
        for step in ['iti','stim','delay','estim']:
            mask[self.design_rg[step], :, self.n_rule_output:] = self.mask[step]
            mask[self.design_rg[step], :, :self.n_rule_output] = self.mask['rule_'+step]
        # set "globally dead" period
        mask[self.dead_rg, :, :] = 0
        return mask

    def _gen_input_rule(self):
        if self.n_rule_input == 0:
            return np.array([]).reshape((self.n_timesteps,self.batch_size,0))

        else:
            rule_mat = np.zeros([self.n_timesteps, self.batch_size, self.n_rule_input])
            for i,k in enumerate(self.input_rule_rg):
                rule_mat[self.input_rule_rg[k], :, i] = self.input_rule_strength
            return rule_mat

    def _gen_output_rule(self):
        if self.n_rule_output == 0:
            return np.array([]).reshape((self.n_timesteps,self.batch_size,0))

        else:
            rule_mat = np.zeros([self.n_timesteps, self.batch_size, self.n_rule_output])
            for i,k in enumerate(self.output_rule_rg):
                rule_mat[self.output_rule_rg[k], :, i] = self.output_rule_strength
            return rule_mat


