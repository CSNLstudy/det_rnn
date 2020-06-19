import numpy as np
import tensorflow as tf
from ._parameters import par

__all__ = ['Model']

# DM-only model
class Model(object):
	def __init__(self, par=par):
		super(Model, self).__init__()
		self.set_params(par)
		self.initialize_variable(par)
		self.optimizer = tf.optimizers.Adam(self.learning_rate) 
		self.model_performance = {'iteration': [], 'eval': [], 'loss': [], 
		'perf_loss': [], 'spike_loss': []}

	def __call__(self, iteration, input_data, target_data, _pass):
		Y, Loss = self._train_oneiter(input_data, target_data, _pass)
		self._append_model_performance(iteration, target_data, Y, Loss)

	# TODO(HG): convert into @property?
	def set_params(self, par):
		for k, v in par.items():
			setattr(self, k, v)

	def initialize_variable(self,par):
		_var_dict = {}
		for k, v in par.items():
			if k[-1] == '0':
				name = k[:-1]
				_var_dict[name] = tf.Variable(par[k], name=name, dtype='float32')
		self.var_dict = _var_dict

	def _get_loss(self, y, target, mask):
		if self.loss_fun == 'cosine':
			loss = tf.reduce_mean(-mask*tf.cos(2.*(y-target)))
		elif self.loss_fun == 'mse':
			loss = tf.reduce_mean(mask*(y-target)**2)
		elif self.loss_fun == 'mse_normalize':
			_y_sum = np.sum(y,axis=-1)[:, np.newaxis]
			loss = tf.reduce_mean(mask * (y/_y_sum-target)**2)
		elif self.loss_fun == 'mse_sigmoid':
			loss = tf.reduce_mean(mask * ((tf.sigmoid(y)-0.5)*2.-target)**2)
		elif self.loss_fun == 'centropy':
			loss = tf.reduce_mean(mask*tf.nn.softmax_cross_entropy_with_logits(
				logits=y, labels=target, axis=2))
		return loss

	def _get_eval(self, target, output):
		# TODO(HG): implement circular correlation
		if self.resp_decoding == 'conti':
			accuracy = np.mean(np.arccos(np.cos((target[self.design_rg['estim'], :, :] -
												 output[self.design_rg['estim'], :, :]) * 2.)))
		elif self.resp_decoding == 'disc':
			target_max = np.argmax(target[self.design_rg['estim'], :, :], axis=2)
			output_max = np.argmax(output[self.design_rg['estim'], :, :], axis=2)
			accuracy = np.mean(np.float32(target_max == output_max))
		return accuracy

	def _rnn_cell(self, _h, rnn_input, _syn_x, _syn_u, _iter):
		# w_rnn constrained by modular mask
		_w_rnn = tf.nn.relu(self.var_dict['w_rnn']) * self.EImodular_mask

		if self.masse:
			_syn_x += (self.alpha_std * (1 - _syn_x) - self.dt * _syn_u * _syn_x * _h) * self.dynamic_synapse
			_syn_u += (self.alpha_stf * (self.U - _syn_u) + self.dt * self.U * (1 - _syn_u) * _h) * self.dynamic_synapse
			_syn_x = tf.minimum(np.float32(1), tf.nn.relu(_syn_x))
			_syn_u = tf.minimum(np.float32(1), tf.nn.relu(_syn_u))
			_h_post = _syn_u * _syn_x * _h
		else:
			_h_post = _h
		_h = tf.nn.relu(_h * (1 - self.alpha_neuron)
							+ self.alpha_neuron * (rnn_input @ tf.nn.relu(self.var_dict['w_in'])
													+ _h_post @ _w_rnn + self.var_dict['b_rnn'])
							+ tf.random.normal(_h.shape, 0, self.noise_rnn_sd, dtype=tf.float32))

		return _h, _syn_x, _syn_u

	def rnn_model(self, input_data):
		# variable names ~_1 or ~_2 are all converted into _~

		_syn_x = self.syn_x_init
		_syn_u = self.syn_u_init

		# h : internal activity at current time, (batch_size, n_units)
		_h = tf.tile(self.var_dict['h'], (_syn_x.shape[0], 1))

		# h_stack : internal activities stacked through time, (time_size, batch_size, n_units)
		# y_stack : output values for Estim (time_size, batch_size, n_output)
		h_stack = []; y_stack = []

		_input_data = tf.unstack(input_data) # this line is important

		for _iter, rnn_input in enumerate(_input_data):
			# rnn_input: (128, 2501), _h: (128, 200)
			_h, _, _ = self._rnn_cell(_h, rnn_input, _syn_x, _syn_u, _iter)
			h_stack.append(_h)
			y_stack.append(_h @ self.var_dict['w_out'] + self.var_dict['b_out'])

		# h_stack.shape = (150, 128, 200)
		# y_stack.shape = (150, 128, 3)
		h_stack = tf.stack(h_stack)
		y_stack = tf.stack(y_stack)
		return y_stack, h_stack

	@tf.function
	def _train_oneiter(self, input_data, target_data, _pass):
		with tf.GradientTape() as t:
			n = 2 if self.spike_regularization == 'L2' else 1
			_Y, _H = self.rnn_model(input_data)  # capitalized since they are stacked
			perf_loss   = self._get_loss(_Y, target_data, _pass)
			spike_loss  = tf.reduce_mean(tf.nn.relu(_H) ** n)
			weight_loss = tf.reduce_mean(tf.nn.relu(self.var_dict['w_rnn'])**n)
			loss = perf_loss + self.spike_cost*spike_loss + self.weight_cost*weight_loss 

		vars_and_grads = t.gradient(loss, self.var_dict)

		# gradient capping and clipping
		capped_gvs = []
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

	def _append_model_performance(self, iteration, target, Y, Loss):
		# compute 
		estim_eval = self._get_eval(target, Y.numpy())
		self.model_performance['iteration'].append(iteration)
		self.model_performance['loss'].append(Loss['loss'].numpy())
		self.model_performance['perf_loss'].append(Loss['perf_loss'].numpy())
		self.model_performance['spike_loss'].append(Loss['spike_loss'].numpy())        
		self.model_performance['eval'].append(estim_eval)

	# TODO(HG): evaluation name
	def print_results(self, iteration):
		print_res = 'Iter. {:4d}'.format(iteration)
		print_res += ' | Evaluaiton {:0.4f}'.format(self.model_performance['eval'][iteration]) +\
					 ' | Loss {:0.4f}'.format(self.model_performance['loss'][iteration])
		print_res += ' | Spike loss {:0.4f}'.format(self.model_performance['spike_loss'][iteration])
		print(print_res)
