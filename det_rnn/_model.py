import tensorflow as tf
from ._parameters import par

__all__ = ['Model']

class Model(tf.Module):
	def __init__(self, par=par):
		super(Model, self).__init__()
		self._set_params(par)
		self._initialize_variable(par)
		self.optimizer = tf.optimizers.Adam(self.learning_rate) 

	@tf.function
	def __call__(self, trial_info, par):
		y, loss = self.train_oneiter(trial_info['neural_input'],
									 trial_info['desired_output'],
									 trial_info['mask'])
		return y, loss

	@tf.function
	def rnn_model(self, input_data):
		# h : internal activity at current time, (batch_size, n_hidden)
		# syn_x_stack, syn_u_stack, h_stack : (n_timesteps, batch_size, n_units)
		# y_stack : (n_timesteps, batch_size, n_output)
		# rnn_input: (batch_size, n_input), _h: (batch_size, n_hidden)
		_syn_x = self.syn_x_init
		_syn_u = self.syn_u_init
		_h = tf.tile(self.var_dict['h'], (_syn_x.shape[0], 1))
		_input_data = tf.unstack(input_data)
		h_stack = []
		y_stack = []
		syn_x_stack = [] #TODO(HG): fix ad-hoc codes
		syn_u_stack = []
		for _iter, rnn_input in enumerate(_input_data):
			_h, _syn_x, _syn_u = self._rnn_cell(_h, rnn_input, _syn_x, _syn_u, _iter)
			h_stack.append(_h)
			y_stack.append(_h @ self.var_dict['w_out'] + self.var_dict['b_out'])
			if self.masse:
				syn_x_stack.append(_syn_x)
				syn_u_stack.append(_syn_u)
		h_stack = tf.stack(h_stack)
		y_stack = tf.stack(y_stack)
		syn_x_stack = tf.stack(syn_x_stack)
		syn_u_stack = tf.stack(syn_u_stack)
		return y_stack, h_stack, syn_x_stack, syn_u_stack

	@tf.function
	def train_oneiter(self, input_data, target_data, mask):
		with tf.GradientTape() as t:
			_Y, _H, _, _ = self.rnn_model(input_data)  # capitalized since they are stacked
			perf_loss   = self._calc_loss(_Y, target_data, mask)
			spike_loss  = tf.reduce_mean(tf.nn.relu(_H)**2)
			weight_loss = tf.reduce_mean(tf.nn.relu(self.var_dict['w_rnn'])**2)
			loss = perf_loss + self.spike_cost*spike_loss + self.weight_cost*weight_loss

		vars_and_grads = t.gradient(loss, self.var_dict)
		capped_gvs = [] # gradient capping and clipping
		for var, grad in vars_and_grads.items():
			if 'w_rnn' in var:
				grad *= self.w_rnn_mask
			elif 'w_out' in var:
				grad *= self.w_out_mask
			elif 'w_in' in var:
				grad *= self.w_in_mask
			capped_gvs.append((tf.clip_by_norm(grad, self.clip_max_grad_val), self.var_dict[var]))
		self.optimizer.apply_gradients(grads_and_vars=capped_gvs)
		return _Y, {'loss':loss, 'perf_loss': perf_loss, 'spike_loss': spike_loss}

	def _set_params(self, par):
		for k, v in par.items():
			if type(v) is dict:
				for v_k, v_v in v.items():     # TODO(HG): should I remove nested dicts?
					v[v_k] = tf.constant(v_v)
				setattr(self, k, v)
			else:
				setattr(self, k, tf.constant(v))

	def _initialize_variable(self,par):
		_var_dict = {}
		for k, v in par.items():
			if k[-1] == '0':
				name = k[:-1]
				_var_dict[name] = tf.Variable(par[k], name=name, dtype='float32')
		self.var_dict = _var_dict

	def _calc_loss(self, y, target, mask):
		if self.loss_fun == 'cosine':
			loss = tf.reduce_mean(-mask*tf.cos(2.*(y-target)))
		elif self.loss_fun == 'mse':
			loss = tf.reduce_mean(mask*(y-target)**2)
		elif self.loss_fun == 'mse_sigmoid':
			_target_sum = tf.reduce_sum(target, axis=2)
			_target_sum = tf.expand_dims(_target_sum, axis=2)
			_target_normalized = target/_target_sum
			_y_normalized = tf.nn.softmax(y, axis=2)
			loss = tf.reduce_mean(mask * (_target_normalized-_y_normalized)**2)
		else: loss = tf.constant(0.) #TODO(HG) what is it?
		return loss

	def _rnn_cell(self, _h, rnn_input, _syn_x, _syn_u, _iter):
		# w_rnn constrained by modular mask
		_w_rnn = tf.nn.relu(self.var_dict['w_rnn']) * self.EI_mask
		if self.masse:
			_syn_x += self.alpha_std * (1 - _syn_x) - self.dt/1000 * _syn_u * _syn_x * _h
			_syn_u += self.alpha_stf * (self.U - _syn_u) + self.dt/1000 * self.U * (1 - _syn_u) * _h
			_syn_x = tf.minimum(tf.constant(1.), tf.nn.relu(_syn_x))
			_syn_u = tf.minimum(tf.constant(1.), tf.nn.relu(_syn_u))
			_h_post = _syn_u * _syn_x * _h
		else:
			_h_post = _h
		_h = tf.nn.relu(_h * (1 - self.alpha_neuron)
						+ self.alpha_neuron * (rnn_input @ tf.nn.relu(self.var_dict['w_in'])
											 + _h_post @ _w_rnn + self.var_dict['b_rnn'])
						+ tf.random.normal(_h.shape, 0, tf.sqrt(2*self.alpha_neuron)*self.noise_rnn_sd, dtype=tf.float32))
		return _h, _syn_x, _syn_u

