# Author: Hyunwoo Gu(@SNU), 21.01.21.

import warnings, pdb
import numpy as np

class LinearFixedPointNetwork:
    """Generate Linear Fixed-Point Network
    
    An analytic solution for working memory network with desired set of fixed points.
    The resulting network accumulates evidence during stimulus presentation, and main-
    tains the line attractor for delay period.


    Parameters
    ----------

    Attributes
    ----------

    Examples
    --------

    """

    def __init__(self, input_mat, par=None, 
        identity_map=None, encoding_dim=None, accumulation_rate=None, 
        eigenvalues=None,  pad_vectors=None, activation_fun='linear', eps=1e-7, verbose=False):

        self.eps   = eps       # epsilon for checking pattern orthonormality
        self.input = input_mat # [N_stimulus_dimension x N_stimulus_category]
        self.input_dim      = self.input.shape[0]
        self.input_category = self.input.shape[1]
        
        if par is None: # use intrinsic par
            self._default_par()
        else:
            self.alpha          = par['alpha_neuron']
            self.noise          = par['noise_rnn_sd']
            self.strength       = par['strength_output']
            self.n_hidden       = par['n_hidden']
            self.input_duration = len(par['design_rg']['stim'])
            self.batch_size     = par['batch_size']

        self.identity_map   = identity_map
        self.encoding_dim   = encoding_dim
        self.accumulation_rate = accumulation_rate
        self.eigenvalues    = eigenvalues
        self.pad_vectors    = pad_vectors
        self.act_fun        = activation_fun  # currently only available for "linear"

        if identity_map is None:
            # Inference using Moore–Penrose pseudoinverse
            # WARNING: May not be exact, and may be vulnerable to input noise
            # TODO(HG): Better way than np.linalg.inv?
            self.identity_map = input_mat.T @ np.linalg.inv(input_mat @ input_mat.T)
        if encoding_dim is None:
            self.encoding_dim = self.n_hidden
        if accumulation_rate is None:
            self.accumulation_rate = self.strength/self.input_duration
        if eigenvalues is None:
            _eigenvalues = np.ones(self.encoding_dim)
            _eigenvalues[self.input_category:] = 0.00001  # arbitrary decaying factor (for stability)
            self.eigenvalues = _eigenvalues
        if pad_vectors is None:
            self.pad_vectors = np.zeros((self.encoding_dim-self.input_category,self.input_dim))

        # TODO(HG)
        # check if eivenvalues are real
        # if complex, phase information should be first computed
        
    def fit(self, patterns=None, Win=None, Wout=None):
        # fit the network to the desired eigenvectors(i.e. patterns)

        if patterns is None: 
            # (TODO) if no patterns are given, orthogonal bases are generated
            pass 
        else:
            # patterns = self._complete_eigenvectors(patterns)
            is_orthonormal = self._check_orthonormality(patterns)

        if is_orthonormal: 
            _Wrec  = patterns @ np.diag(self.eigenvalues) @ patterns.T
            _Wout  = patterns.T
        else: 
            _Wrec  = patterns @ np.diag(self.eigenvalues) @ np.linalg.inv(patterns)
            _Wout  = np.linalg.inv(patterns)

        _Win   = patterns @ np.concatenate((self.identity_map,self.pad_vectors),axis=0) * self.accumulation_rate
        _Wrec  = 1./self.alpha * (_Wrec + (self.alpha-1.) * np.eye(self.encoding_dim))

        self.Wrec = _Wrec
        
        if Win is None: self.Win  = _Win
        else: self.Win  = Win
        
        if Wout is None: self.Wout = _Wout
        else: self.Wout = Wout

    def predict(self, input_data, apply_Wout=True):
        return self._rnn_model(input_data, apply_Wout=apply_Wout)

    def _rnn_cell(self, h, i):
        noise = np.random.normal(0,self.noise,h.shape)
        if self.act_fun == 'linear': 
            h = ((1-self.alpha)*h + self.alpha*np.dot(h, self.Wrec) + np.dot(i, self.Win.T)) + noise
        elif self.act_fun == 'relu':  # not available
            h = np.maximum((1-self.alpha)*h + self.alpha*np.dot(h, self.Wrec) + np.dot(i, self.Win.T) + noise,0)
        return h

    def _rnn_model(self, input_data, apply_Wout=True):
        h = np.zeros((self.batch_size,self.encoding_dim)) # zero initial value
        h_stack = []
        for rnn_input in input_data:
            h = self._rnn_cell(h, rnn_input)
            h_stack.append(h)
        # pdb.set_trace()
        if apply_Wout:
            return (np.stack(h_stack @ self.Wout.T).transpose((2,0,1)))
        else:
            return np.stack(h_stack).transpose((2,0,1))

    # TODOs(HG)
    def _complete_eigenvectors(self, patterns):
        # If insufficient eigenvectors are provided (i.e. N_dimension > N_vectors)
        # Fills the rest via Gram-Schmidt orthogonalization / Householder transformation

        # if patterns.shape[0] > patterns.shape[1]:            

        # if patterns
        #     warnings.warn("Patterns seem highly dependent. Results might not be highly inaccurate", UserWarning)
        pass

    def _check_orthonormality(self, patterns):
        if np.sum(np.abs(patterns @ patterns.T - np.eye(patterns.shape[0]))) < self.eps: return True
        else:
            warnings.warn("Patterns seem not orthonormal. Matrix inverse precision might be compromised.")
            return False

    def _default_par(self):
        self.alpha          = 0.1
        self.noise          = 0.05
        self.strength       = 1.
        self.n_hidden       = 100
        self.input_duration = 150
        self.batch_size     = 128
