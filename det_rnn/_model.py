import os
import numpy as np
import tensorflow as tf
from ._parameters import par

__all__ = ['Model']

class Model(tf.Module):
	def __init__(self, par=par):
		super(Model, self).__init__()
		self.set_params(par)
		self.initialize_variable(par)
		self.optimizer = tf.optimizers.Adam(self.learning_rate) 
		self.model_performance = {'iteration': [], 'perf': [], 'loss': [],
		'perf_loss': [], 'spike_loss': []}

	def __call__(self, trial_info):
		Y, Loss = self._train_oneiter(trial_info['neural_input'],
									  trial_info['desired_output'],
									  trial_info['mask'])
		self._append_model_performance(trial_info, Y, Loss)

	def set_params(self, par):
		for k, v in par.items():
			setattr(self, k, v)
		self.design_rg_estim = par['design_rg']['estim']

	def initialize_variable(self,par):
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
		elif self.loss_fun == 'mse_normalize':
			_y_sum = np.sum(y,axis=-1)[:, np.newaxis]
			loss = tf.reduce_mean(mask * (y/_y_sum-target)**2)
		elif self.loss_fun == 'mse_sigmoid':
			_target_sum = tf.reduce_sum(target, axis=2)
			_target_sum = tf.expand_dims(_target_sum, axis=2)
			_target_normalized = target/_target_sum
			_y_normalized = tf.nn.softmax(y, axis=2)
			loss = tf.reduce_mean(mask * (_target_normalized-_y_normalized)**2)
		elif self.loss_fun == 'centropy':
			loss = tf.reduce_mean(mask*tf.nn.softmax_cross_entropy_with_logits(
				logits=y, labels=target, axis=2))
		return loss

	def _get_eval(self, trial_info, output):
		if self.resp_decoding == 'conti':
			# TODO(HG): implement circular correlation
			perf = np.mean(np.arccos(np.cos((trial_info['desired_output'][self.design_rg_estim, :, :] -
												 output[self.design_rg_estim, :, :]) * 2.)))
		elif self.resp_decoding == 'disc':
			if self.loss_fun == 'mse_sigmoid':
				cenoutput = tf.nn.softmax(output, axis=2).numpy()
				post_prob = cenoutput[:, :, self.n_rule_output:]
				post_prob = post_prob / (
							np.sum(post_prob, axis=2, keepdims=True) + np.finfo(np.float32).eps)  # Dirichlet normaliation
				post_support = np.linspace(0, np.pi, self.n_ori, endpoint=False) + np.pi / self.n_ori / 2
				pseudo_mean = np.arctan2(post_prob @ np.sin(2 * post_support),
										 post_prob @ np.cos(2 * post_support)) / 2
				estim_sinr = (np.sin(2 * pseudo_mean[self.design_rg_estim, :])).mean(axis=0)
				estim_cosr = (np.cos(2 * pseudo_mean[self.design_rg_estim, :])).mean(axis=0)
				estim_mean = np.arctan2(estim_sinr, estim_cosr) / 2
				perf = np.mean(np.cos(2.*(trial_info['stimulus_ori'] * np.pi/self.n_ori - estim_mean)))
			else:
				target_max = np.argmax(trial_info['desired_output'][self.design_rg_estim, :, :], axis=2)
				output_max = np.argmax(output[self.design_rg_estim, :, :], axis=2)
				perf = np.mean(np.float32(target_max == output_max))
		return perf

	def _rnn_cell(self, _h, rnn_input, _syn_x, _syn_u, _iter):
		# w_rnn constrained by modular mask
		_w_rnn = tf.nn.relu(self.var_dict['w_rnn']) * self.EI_mask
		if self.masse:
			_syn_x += self.alpha_std * (1 - _syn_x) - self.dt/1000 * _syn_u * _syn_x * _h
			_syn_u += self.alpha_stf * (self.U - _syn_u) + self.dt/1000 * self.U * (1 - _syn_u) * _h
			_syn_x = tf.minimum(np.float32(1), tf.nn.relu(_syn_x))
			_syn_u = tf.minimum(np.float32(1), tf.nn.relu(_syn_u))
			_h_post = _syn_u * _syn_x * _h
		else:
			_h_post = _h
		_h = tf.nn.relu(_h * (1 - self.alpha_neuron)
							+ self.alpha_neuron * (rnn_input @ tf.nn.relu(self.var_dict['w_in'])
													+ _h_post @ _w_rnn + self.var_dict['b_rnn'])
							+ tf.random.normal(_h.shape, 0, np.sqrt(2*self.alpha_neuron)*self.noise_rnn_sd, dtype=tf.float32))
		return _h, _syn_x, _syn_u

	def rnn_model(self, input_data):
		# h : internal activity at current time, (batch_size, n_hidden)
		# h_stack : (n_timesteps, batch_size, n_units)
		# y_stack : (n_timesteps, batch_size, n_output)
		# rnn_input: (batch_size, n_input), _h: (batch_size, n_hidden)
		_syn_x = self.syn_x_init
		_syn_u = self.syn_u_init
		_h = tf.tile(self.var_dict['h'], (_syn_x.shape[0], 1))
		_input_data = tf.unstack(input_data)
		h_stack = []
		y_stack = []


		for _iter, rnn_input in enumerate(_input_data):
			_h, _syn_x, _syn_u = self._rnn_cell(_h, rnn_input, _syn_x, _syn_u, _iter)
			h_stack.append(_h)
			y_stack.append(_h @ self.var_dict['w_out'] + self.var_dict['b_out'])

		h_stack = tf.stack(h_stack)
		y_stack = tf.stack(y_stack)
		return y_stack, h_stack

	@tf.function
	def _train_oneiter(self, input_data, target_data, mask):
		with tf.GradientTape() as t:
			n = 2 if self.spike_regularization == 'L2' else 1
			_Y, _H = self.rnn_model(input_data)  # capitalized since they are stacked
			perf_loss   = self._calc_loss(_Y, target_data, mask)
			spike_loss  = tf.reduce_mean(tf.nn.relu(_H) ** n)
			weight_loss = tf.reduce_mean(tf.nn.relu(self.var_dict['w_rnn'])**n)
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

	def _append_model_performance(self, trial_info, Y, Loss):
		# compute
		estim_perf = self._get_eval(trial_info, Y.numpy())
		self.model_performance['loss'].append(Loss['loss'].numpy())
		self.model_performance['perf_loss'].append(Loss['perf_loss'].numpy())
		self.model_performance['spike_loss'].append(Loss['spike_loss'].numpy())        
		self.model_performance['perf'].append(estim_perf)

	def print_results(self, iteration):
		print_res = 'Iter. {:4d}'.format(iteration)
		print_res += ' | Performance {:0.4f}'.format(self.model_performance['perf'][iteration]) +\
					 ' | Loss {:0.4f}'.format(self.model_performance['loss'][iteration])
		print_res += ' | Spike loss {:0.4f}'.format(self.model_performance['spike_loss'][iteration])
		print(print_res)
