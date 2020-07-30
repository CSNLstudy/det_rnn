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
        # self._generate_tuning() # Generate tuning/input config

    def set_params(self, par):
        for k, v in par.items():
            setattr(self, k, v)

    def generate_trial(self, balanced=False):
        # generate all the orientations to output in a trial
        if balanced is True:
            # if testing is true, generate stimulus evenly
            repeatN = np.ceil(self.batch_size/self.n_ori) # repeat to somewhat match batch size.
            stimulus_ori = np.repeat(np.arange(self.n_ori), repeatN) # todo: make the testing trial generation more elegant...
        else:
            stimulus_ori = np.random.choice(np.arange(self.n_ori), p=self.stim_p, size=self.batch_size)

        if (type(self.kappa) is str) and(self.kappa is 'dist'):
            stimulus_kap = np.random.gamma(shape = self.kappa_dist_shape, scale = self.kappa_dist_scale, size = stimulus_ori.shape)
        else:
            stimulus_kap = self.kappa * np.ones(stimulus_ori.shape)

        return Trial(stim=self,stimulus_ori=stimulus_ori,stimulus_kap=stimulus_kap
                     )() # josh: return by calling the generated class
                         # => makes trial_info struct compatible with previous versions of the code.

    """
    def _gen_stim(self, stimulus_ori):
        # TODO(HG): need to be changed if n_ori =/= n_tuned
        neural_input = random.standard_normal(size=(self.n_timesteps, self.batch_size, self.n_input))*self.noise_sd + self.noise_mean
        neural_input[:,:,:self.n_rule_input] += self._gen_input_rule()
        for t in range(self.batch_size):
            neural_input[self.design_rg['stim'],t,self.n_rule_input:] += self.tuning_input[:,0,stimulus_ori[t]].reshape((1,-1))
        if self.n_subblock > 1: # multi-trial settings
            neural_input = neural_input.transpose((1,0,2)).\
                reshape((self.n_subblock,-1,self.n_input)).transpose((1,0,2))
        return neural_input

    def _gen_output(self, stimulus_ori):
        desired_output = np.zeros((self.n_timesteps,self.batch_size,self.n_output), dtype=np.float32)
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
        mask = np.zeros((self.n_timesteps, self.batch_size, self.n_output), dtype=np.float32)
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

    # TODO(HG): simplify here (Make n_ori flexible!!!)
    def _generate_tuning(self):
        # Tuning function is a von Mises distribution
        # Tuning input shape = (n_input, 1, n_orientations)??
        _tuning_input  = np.zeros((self.n_tuned_input,  self.n_receptive_fields, self.n_ori))
        _tuning_output = np.zeros((self.n_tuned_output, self.n_receptive_fields, self.n_ori))
        stim_dirs = np.float32(np.arange(0,180,180/self.n_ori))
        pref_dirs = np.float32(np.arange(0,180,180/self.n_tuned_input)) #josh: important change!

        for n in range(self.n_tuned_input):
            for i in range(self.n_ori):
                d = np.cos((stim_dirs[i] - pref_dirs[n])/90*np.pi)
                _tuning_input[n,0,i]  = self.strength_input*np.exp(self.kappa*d)/np.exp(self.kappa) #np.exp(kappa) is the height at mode
                _tuning_output[n,0,i] = self.strength_output*np.exp(self.kappa*d)/np.exp(self.kappa)

        if self.stim_encoding == 'single':
            self.tuning_input  = _tuning_input

        elif self.stim_encoding == 'double':
            self.tuning_input = np.tile(_tuning_input,(2,1))

        self.tuning_output  = _tuning_output
        self.pref_dirs      = pref_dirs
    """

class Trial(object):
    """
    make a Trial class in order to change the trial structures more flexible (i.e. balanced data set)
    """
    def __init__(self, stim, stimulus_ori, stimulus_kap):
        # attributes that need to be flexible
        self.stimulus_ori   = stimulus_ori
        self.stimulus_kap   = stimulus_kap
        self.batch_size     = stimulus_ori.shape[0]
        self.n_subblock     = int(self.batch_size / stim.trial_per_subblock)
        [self.neural_input, self.desired_output, self.input_tuning, self.output_tuning] = self._gen_stim(stim, stimulus_ori, stimulus_kap)
        self.mask           = self._gen_mask(stim)

    def __call__(self):
        return {'neural_input'  : self.neural_input.astype(np.float32),
                'desired_output': self.desired_output.astype(np.float32),
                'mask'          : self.mask,
                'input_tuning'  : self.input_tuning,
                'output_tuning' : self.output_tuning,
                'stimulus_ori'  : self.stimulus_ori,
                'stimulus_kap'  : self.stimulus_kap}

    def _gen_stim(self, stim, stimulus_ori, stimulus_kap):
        assert stimulus_ori.shape == stimulus_kap.shape

        # initialize neural input with noise, and stimulus tuning without noise
        neural_input = random.standard_normal(size=(stim.n_timesteps, self.batch_size, stim.n_input))*stim.noise_sd + stim.noise_mean
        neural_input[:,:,:stim.n_rule_input] += self._gen_input_rule(stim) # add rules
        stimulus_tunings = np.zeros(shape=(self.batch_size, stim.n_input))

        desired_output = np.zeros((stim.n_timesteps,self.batch_size,stim.n_output), dtype=np.float32) # output has no noise
        desired_output[:, :, :stim.n_rule_output] = self._gen_output_rule(stim)

        # stimulus directions and preferred directions
        # Compute vonMises in the 2*theta space todo: write a functino for vonMises?
        stim_dirs = np.tile(np.float32(np.arange(0, 180, 180 / stim.n_ori)).reshape(-1,1), [1, stim.n_tuned_input]) # stim_dirs x n_tuned_input
        pref_dirs_in    = np.tile(np.float32(np.arange(0, 180, 180 / stim.n_tuned_input)).reshape(1,-1), [self.batch_size, 1]) # B x n_tuned_input
        din             = np.cos(2*(stim_dirs[stimulus_ori,:] - pref_dirs_in) / 90 * np.pi)      # B x n_tuned_input
        input_tuning    = np.exp(stimulus_kap.reshape(-1,1)*din)/np.exp(stimulus_kap.reshape(-1,1))  # broadcasted to B x n_tuned_input
        input_tuning    = input_tuning/np.sum(input_tuning,1,keepdims = True)

        neural_input[stim.design_rg['stim'],:,stim.n_rule_input:] += np.tile(input_tuning[np.newaxis,:,:],
                                                                             [stim.design_rg['stim'].shape[0], 1, 1]) # time x B x n_tuned_input

        stim_dirs_out = np.tile(np.float32(np.arange(0, 180, 180 / stim.n_ori)).reshape(-1, 1),
                                [1, stim.n_tuned_output])  # stim_dirs x n_tuned_output
        pref_dirs_out   = np.tile(np.float32(np.arange(0, 180, 180 / stim.n_tuned_output)).reshape(1,-1), [self.batch_size, 1])  # B x n_tuned_output
        dout            = np.cos(2*(stim_dirs_out[stimulus_ori,:] - pref_dirs_out) / 90 * np.pi)        # B x n_output todo(josh): check that it is in the right domain (2theta)
        output_tuning   = np.exp(stimulus_kap.reshape(-1,1)*dout)/np.exp(stimulus_kap.reshape(-1,1))  # broadcasted to B x n_tuned_output
        output_tuning   = output_tuning/np.sum(output_tuning,1,keepdims = True)

        desired_output[stim.output_rg, :, stim.n_rule_output:] = np.tile(output_tuning[np.newaxis,:,:],
                                                                         [stim.output_rg.shape[0], 1, 1])  # time x B x n_tuned_input

        if self.n_subblock > 1: # multi-trial settings #josh: i.e. concatenate runs in the sublock ??? todo: check this...
            neural_input = neural_input.transpose((1,0,2)).reshape((self.n_subblock,-1,stim.n_input)).transpose((1,0,2))
            desired_output = desired_output.transpose((1,0,2)).reshape((self.n_subblock,-1,stim.n_tuned_output)).transpose((1,0,2))

        return neural_input, desired_output, input_tuning, output_tuning

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
        mask = np.zeros((stim.n_timesteps, self.batch_size, stim.n_output), dtype=np.float32)
        # set "specific" period
        for step in ['iti','stim','delay','estim']:
            mask[stim.design_rg[step], :, stim.n_rule_output:] = stim.mask[step]
            mask[stim.design_rg[step], :, :stim.n_rule_output] = stim.mask['rule_'+step]
        # set "globally dead" period
        mask[stim.dead_rg, :, :] = 0
        if self.n_subblock > 1: # multi-trial settings
            mask = mask.transpose((1,0,2)).\
                reshape((self.n_subblock,-1,stim.n_output)).transpose((1,0,2))
        return mask

    def _gen_input_rule(self, stim):
        if stim.n_rule_input == 0:
            return np.array([]).reshape((stim.n_timesteps,self.batch_size,0))

        else:
            rule_mat = np.zeros([stim.n_timesteps, self.batch_size, stim.n_rule_input])
            for i,k in enumerate(stim.input_rule_rg):
                rule_mat[stim.input_rule_rg[k], :, i] = stim.input_rule_strength
            return rule_mat

    def _gen_output_rule(self, stim):
        if stim.n_rule_output == 0:
            return np.array([]).reshape((stim.n_timesteps,self.batch_size,0))
        else:
            rule_mat = np.zeros([stim.n_timesteps, self.batch_size, stim.n_rule_output])
            for i,k in enumerate(stim.output_rule_rg):
                rule_mat[stim.output_rule_rg[k], :, i] = stim.output_rule_strength
            return rule_mat

    def plot_trial(self, stim, TEST_TRIAL=None):
        if TEST_TRIAL is None:
            TEST_TRIAL = np.random.randint(self.batch_size)

        axes = {}
        fig = plt.figure(constrained_layout=True, figsize=(10, 8))
        gs = fig.add_gridspec(6, 2)
        axes[0] = fig.add_subplot(gs[0, 0])
        im0 = axes[0].imshow(self.neural_input[:, TEST_TRIAL, :stim.n_rule_input].T,
                             interpolation='none',
                             aspect='auto');
        axes[0].set_title("Input Rule")
        axes[0].set_xlabel("Time (frames)")
        fig.colorbar(im0, ax=axes[0])

        axes[4] = fig.add_subplot(gs[1, 0])
        im3 = axes[4].imshow(self.mask[:, TEST_TRIAL, :stim.n_rule_output].T,
                             interpolation='none',
                             aspect='auto');
        axes[4].set_title("Training Mask_rules")
        axes[4].set_xlabel("Time (frames)")
        fig.colorbar(im3, ax=axes[4])

        axes[5] = fig.add_subplot(gs[1, 1])
        im4 = axes[5].imshow(self.mask[:, TEST_TRIAL, stim.n_rule_output:].T,
                             interpolation='none',
                             aspect='auto');
        axes[5].set_title("Training Mask");
        axes[5].set_xlabel("Time (frames)")
        axes[5].set_ylabel("Neuron")
        fig.colorbar(im4, ax=axes[5])

        axes[2] = fig.add_subplot(gs[0, 1])
        im2 = axes[2].imshow(self.desired_output[:, TEST_TRIAL, :stim.n_rule_output].T,
                             interpolation='none',
                             aspect='auto');
        axes[2].set_title("Desired Output Rules")
        axes[2].set_xlabel("Time (frames)")
        fig.colorbar(im2, ax=axes[2])

        axes[1] = fig.add_subplot(gs[2:4, :])
        im1 = axes[1].imshow(self.neural_input[:, TEST_TRIAL, stim.n_rule_input:].T,
                             extent=[0, self.neural_input.shape[0], stim.pref_dirs[0], stim.pref_dirs[-1]],
                             origin='lower',
                             interpolation='none',
                             aspect='auto');
        axes[1].set_title("Neural Input")
        axes[1].set_xlabel("Time (frames)")
        axes[1].set_ylabel("Neuron (pref. ori. deg)")
        fig.colorbar(im1, ax=axes[1])

        axes[3] = fig.add_subplot(gs[4:6, :])
        im2 = axes[3].imshow(self.desired_output[:, TEST_TRIAL, stim.n_rule_output:].T,
                             # extent=[0, trial_info['neural_input'].shape[0], stim.pref_dirs[0], stim.pref_dirs[-1]],
                             origin='lower',
                             interpolation='none',
                             aspect='auto');
        axes[3].set_title("Desired Output")
        axes[3].set_xlabel("Time (frames)")
        axes[3].set_ylabel("Neuron (#)")
        fig.colorbar(im2, ax=axes[3])

        # fig.tight_layout(pad=2.0)
        plt.show()
