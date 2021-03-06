import tensorflow as tf
from .hyper import *

__all__ = ['Model']

class Model(tf.Module):
	def __init__(self, hp=hp):
		super(Model, self).__init__()
		self._initialize_variable(hp)
		self.optimizer = tf.optimizers.Adam(hp['learning_rate'])

	@tf.function
	def __call__(self, trial_info, hp):
		y, loss = self._train_oneiter(trial_info['neural_input'],
									  trial_info['desired_decision'], trial_info['desired_estim'],
									  trial_info['mask_decision'], trial_info['mask_estim'], hp)
		return y, loss

	@tf.function
	def rnn_model(self, input_data, hp):
		_syn_x = hp['syn_x_init']
		_syn_u = hp['syn_u_init']
		_h = tf.cast(tf.tile(self.var_dict['h'], (_syn_x.shape[0], 1)), tf.float32)
		h_stack = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False)
		y_dm_stack  = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False)
		y_em_stack  = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False)
		syn_x_stack = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False)
		syn_u_stack = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False)
		i = 0
		for rnn_input in input_data:
			_h, _syn_x, _syn_u = self._rnn_cell(_h, rnn_input, _syn_x, _syn_u, hp)
			h_stack = h_stack.write(i, tf.cast(_h, tf.float32))
			y_dm_matmul = tf.cast(_h, tf.float32) @ self.var_dict['w_out_dm']
			y_dm_stack = y_dm_stack.write(i, y_dm_matmul + self.var_dict['b_out_dm'])
			y_em_matmul = tf.cast(_h, tf.float32) @ self.var_dict['w_out_em']
			y_em_stack = y_em_stack.write(i, y_em_matmul + self.var_dict['b_out_em'])
			syn_x_stack = syn_x_stack.write(i, tf.cast(_syn_x, tf.float32))
			syn_u_stack = syn_u_stack.write(i, tf.cast(_syn_u, tf.float32))
			i += 1
		h_stack = h_stack.stack()
		y_dm_stack = y_dm_stack.stack()
		y_em_stack = y_em_stack.stack()
		syn_x_stack = syn_x_stack.stack()
		syn_u_stack = syn_u_stack.stack()
		return y_dm_stack, y_em_stack, h_stack, syn_x_stack, syn_u_stack

	def _train_oneiter(self, input_data, target_data_dm, target_data_em, mask_dm, mask_em, hp):
		with tf.GradientTape() as t:
			_Ydm, _Yem, _H, _, _ = self.rnn_model(input_data, hp)  # capitalized since they are stacked
			perf_loss_dm = self._calc_loss(tf.cast(_Ydm,tf.float32), tf.cast(target_data_dm,tf.float32),tf.cast(mask_dm,tf.float32), hp)
			perf_loss_em = self._calc_loss(tf.cast(_Yem,tf.float32), tf.cast(target_data_em,tf.float32),tf.cast(mask_em,tf.float32), hp)
			spike_loss  = tf.reduce_mean(tf.nn.relu(tf.cast(_H,tf.float32))**2)
			weight_loss = tf.reduce_mean(tf.nn.relu(self.var_dict['w_rnn'])**2)
			loss = hp['lam_decision'] * perf_loss_dm + hp['lam_estim'] * perf_loss_em \
				   + tf.cast(hp['spike_cost'],tf.float32)*spike_loss + tf.cast(hp['weight_cost'],tf.float32)*weight_loss

		vars_and_grads = t.gradient(loss, self.var_dict)
		capped_gvs = [] # gradient capping and clipping
		for var, grad in vars_and_grads.items():
			if 'w_rnn' in var:
				grad *= hp['w_rnn_mask']
			elif 'w_out_dm' in var:
				grad *= hp['w_out_dm_mask']
			elif 'w_out_em' in var:
				grad *= hp['w_out_em_mask']
			elif 'w_in' in var:
				grad *= hp['w_in_mask']

			if grad is None:
				capped_gvs.append((grad, self.var_dict[var]))
			else:
				capped_gvs.append((tf.clip_by_norm(grad, hp['clip_max_grad_val']), self.var_dict[var]))
		self.optimizer.apply_gradients(grads_and_vars=capped_gvs)
		return {'dm':_Ydm, 'em':_Yem}, {'loss':loss, 'perf_loss_dm': perf_loss_dm, 'perf_loss_em': perf_loss_em, 'spike_loss': spike_loss}

	def _initialize_variable(self,hp):
		_var_dict = {}
		for k, v in hp.items():
			if k[-1] == '0':
				name = k[:-1]
				_var_dict[name] = tf.Variable(hp[k], name=name, dtype='float32')
		self.var_dict = _var_dict

	def _calc_loss(self, y, target, mask, hp):
		if hp['task_type'] == 1:
			target = target / tf.reduce_sum(target, axis=2, keepdims=True)
		if hp['loss_fun'] == 0:
			_y_normalized = tf.nn.softmax(y)
			loss = tf.reduce_mean(mask * (target - _y_normalized) ** 2)
		elif hp['loss_fun'] == 1:
			_y_logsft = tf.nn.log_softmax(y)
			loss = tf.reduce_mean(mask* (-target * _y_logsft))
		else:
			loss = 0.
		return loss

	def _rnn_cell(self, _h, rnn_input, _syn_x, _syn_u, hp):
		_w_rnn = tf.nn.relu(self.var_dict['w_rnn']) * tf.cast(hp['EI_mask'], tf.float32)
		if hp['masse']:
			_syn_x += tf.cast(hp['alpha_std'] * (1. - _syn_x) - hp['dt']/1000 * _syn_u * _syn_x * _h, tf.float32)
			_syn_u += tf.cast(hp['alpha_stf'] * (hp['U'] - _syn_u) + hp['dt']/1000 * hp['U'] * (1. - _syn_u) * _h, tf.float32)
			_syn_x = tf.cast(tf.minimum(tf.constant(1.), tf.nn.relu(_syn_x)), tf.float32)
			_syn_u = tf.cast(tf.minimum(tf.constant(1.), tf.nn.relu(_syn_u)), tf.float32)
			_h_post = _syn_u * _syn_x * _h
		else:
			_h_post = _h
		_h = tf.nn.relu(tf.cast(_h, tf.float32) * (1. - hp['alpha_neuron'])
						+ hp['alpha_neuron'] * (tf.cast(rnn_input, tf.float32) @ tf.nn.relu(self.var_dict['w_in'])
											 + tf.cast(_h_post, tf.float32) @ _w_rnn + self.var_dict['b_rnn'])
						+ tf.random.normal(_h.shape, 0, tf.sqrt(2*hp['alpha_neuron'])*hp['noise_rnn_sd'], dtype=tf.float32))
		return _h, _syn_x, _syn_u