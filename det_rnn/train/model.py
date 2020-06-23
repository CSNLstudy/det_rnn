import tensorflow as tf
from .hyper import *
from det_rnn.base.functions import alternating

EPSILON = 1e-7

__all__ = ['Model']

class Model(tf.Module):
	def __init__(self, hp=hp):
		super(Model, self).__init__()
		self._initialize_variable(hp)
		self.n_rule_output = hp['n_rule_output'] #
		self.optimizer = tf.optimizers.Adam(hp['learning_rate']) #josh: do we want optimizer inside the model?

	@tf.function
	def __call__(self, trial_info, hp):
		y, loss = self._train_oneiter(trial_info['neural_input'],
									  trial_info['desired_output'],
									  trial_info['mask'], hp)
		return y, loss

	@tf.function
	def rnn_model(self, input_data, hp):
		# make batchsize flexible
		T, B, Nneurons = input_data.shape
		nhidden = self.var_dict['h'].shape[1]
		_syn_x = tf.ones((B, nhidden))
		_syn_u = tf.tile(
			tf.expand_dims(alternating((0.15, 0.45), nhidden),axis=0),
			(B, 1))
		_h = tf.cast(tf.tile(self.var_dict['h'], (B, 1)), tf.float32)
		h_stack = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False)
		y_stack = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False)
		syn_x_stack = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False)
		syn_u_stack = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False)
		i = 0
		for rnn_input in input_data: # loop over time
			_h, _syn_x, _syn_u = self._rnn_cell(_h, rnn_input, _syn_x, _syn_u, hp)
			h_stack = h_stack.write(i, tf.cast(_h, tf.float32))
			y_stack = y_stack.write(i, tf.cast(_h, tf.float32) @ self.var_dict['w_out'] + self.var_dict['b_out'])
			if hp['masse']:
				syn_x_stack = syn_x_stack.write(i, tf.cast(_syn_x, tf.float32))
				syn_u_stack = syn_u_stack.write(i, tf.cast(_syn_u, tf.float32))
			i += 1
		h_stack = h_stack.stack()
		y_stack = y_stack.stack()
		syn_x_stack = syn_x_stack.stack()
		syn_u_stack = syn_u_stack.stack()
		return y_stack, h_stack, syn_x_stack, syn_u_stack

	def return_losses(self, input_data, target_data, mask, hp):
		# check weights
		#print('Number of sparse weights: ' +
		#	  str(tf.math.reduce_sum(tf.cast(self.var_dict['w_rnn'] < 0,tf.float32)).numpy()))
		# do the negative weights change in case of relu?
		_Y, _H, _, _ = self.rnn_model(input_data, hp)  # capitalized since they are stacked
		perf_loss = self._calc_loss(tf.cast(_Y, tf.float32), tf.cast(target_data, tf.float32),
									tf.cast(mask, tf.float32), hp)
		spike_loss = tf.reduce_mean(tf.nn.relu(tf.cast(_H, tf.float32)) ** 2)
		weight_loss = tf.reduce_mean(tf.nn.relu(self.var_dict['w_rnn']) ** 2)
		loss = perf_loss + tf.cast(hp['spike_cost'], tf.float32) * spike_loss + tf.cast(hp['weight_cost'],
																						tf.float32) * weight_loss

		return _Y, {'loss':loss, 'perf_loss': perf_loss, 'spike_loss': spike_loss}

	def _train_oneiter(self, input_data, target_data, mask, hp):
		with tf.GradientTape() as t:
			_Y, _H, _, _ = self.rnn_model(input_data, hp)  # capitalized since they are stacked
			perf_loss   = self._calc_loss(tf.cast(_Y,tf.float32), tf.cast(target_data,tf.float32), tf.cast(mask,tf.float32), hp)
			spike_loss  = tf.reduce_mean(tf.nn.relu(tf.cast(_H,tf.float32))**2)
			weight_loss = tf.reduce_mean(tf.nn.relu(self.var_dict['w_rnn'])**2)
			loss = perf_loss + tf.cast(hp['spike_cost'],tf.float32)*spike_loss + tf.cast(hp['weight_cost'],tf.float32)*weight_loss

		vars_and_grads = t.gradient(loss, self.var_dict) # josh: with GradientTape?
		capped_gvs = [] # gradient capping and clipping
		for var, grad in vars_and_grads.items():
			if 'w_rnn' in var:
				grad *= hp['w_rnn_mask']
			elif 'w_out' in var:
				grad *= hp['w_out_mask']
			elif 'w_in' in var:
				grad *= hp['w_in_mask']
			capped_gvs.append((tf.clip_by_norm(grad, hp['clip_max_grad_val']), self.var_dict[var]))
		self.optimizer.apply_gradients(grads_and_vars=capped_gvs)
		return _Y, {'loss':loss, 'perf_loss': perf_loss, 'spike_loss': spike_loss}

	def _initialize_variable(self,hp):
		_var_dict = {}
		for k, v in hp.items():
			if k[-1] == '0': #make this more robust. how to add weights in tf?
				name = k[:-1]
				_var_dict[name] = tf.Variable(hp[k], name=name, dtype='float32')
		self.var_dict = _var_dict

	def _calc_loss(self, y, target, mask, hp):
		target_ori = target[:,:,self.n_rule_output:]
		y_ori = y[:,:,self.n_rule_output:]
		_target_normalized = target_ori / (tf.reduce_sum(target_ori, axis=2, keepdims=True) + EPSILON) #add epsilon?
		mask_ori = mask[:,:,self.n_rule_output:]
		if hp['loss_fun'] == 0:
			_y_normalized = tf.nn.softmax(y_ori + EPSILON)
			loss = tf.reduce_mean(mask_ori * (_target_normalized - _y_normalized) ** 2)
		elif hp['loss_fun'] == 1:
			_y_logsft = tf.nn.log_softmax(y_ori + EPSILON)
			loss = tf.reduce_mean(mask_ori * (-_target_normalized * _y_logsft))
		else:
			loss = 0.

		# add rule loss:
		rulelosses = (target[:,:,:self.n_rule_output] - y[:,:,:self.n_rule_output])**2 # use mse loss for rules
		loss += tf.reduce_mean(mask[:,:,:self.n_rule_output] * rulelosses)

		#debug losse
		"""
		import matplotlib.pyplot as plt
		trial_n = 0
				
		# cross-entropy loss
		fig, ax = plt.subplots(8, figsize=[15, 8])
		ax[0].set_title("Target")
		im0 = ax[0].imshow(target[:, trial_n, :].numpy().T)
		fig.colorbar(im0, ax=ax[0])
		
		ax[1].set_title("Normalized target")
		im1= ax[1].imshow(_target_normalized[:, trial_n, :].numpy().T)
		fig.colorbar(im1, ax=ax[1])
		
		ax[2].set_title("Predicted output")
		im2 = ax[2].imshow(y[:, trial_n, :].numpy().T)
		fig.colorbar(im2, ax=ax[2])
		
		ax[3].set_title("Predicted output sftmax")
		im3 = ax[3].imshow(_y_normalized[:, trial_n, :].numpy().T)
		fig.colorbar(im3, ax=ax[3])
		
		ax[4].set_title("Predicted output logsftmax")
		im4 = ax[4].imshow(_y_logsft[:, trial_n, :].numpy().T)
		fig.colorbar(im4, ax=ax[4])
		
		ax[5].set_title("loss (target_normalized* y_logsft)")
		im5 = ax[5].imshow(((-_target_normalized * _y_logsft))[:, trial_n, :].numpy().T)
		fig.colorbar(im5, ax=ax[5])

		ax[6].set_title("masked loss")
		im6 = ax[6].imshow((mask_ori*(-_target_normalized * _y_logsft))[:, trial_n, :].numpy().T)
		fig.colorbar(im6, ax=ax[6])

		ax[7].set_title("mask")
		im7 = ax[7].imshow(mask[:, trial_n, :].numpy().T)
		fig.colorbar(im7, ax=ax[7])
		plt.show()
		"""
		return loss

	def _rnn_cell(self, _h, rnn_input, _syn_x, _syn_u, hp):
		_w_rnn = tf.nn.relu(self.var_dict['w_rnn']) * tf.cast(hp['EI_mask'], tf.float32) #josh: tf.nn.relu or tf.abs

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
