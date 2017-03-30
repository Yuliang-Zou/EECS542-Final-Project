# Define the vgg16 style model
# Author: Yuliang Zou
#         ylzou@umich.edu
# Date:   2017-03-24

try:
	import ipdb
except:
	import pdb as ipdb

import tensorflow as tf
import numpy as np
from util import bilinear_upsample_weights


"""Define a base class, containing some useful layer functions"""
class Network(object):
	def __init__(self, inputs):
		self.inputs = []
		self.layers = {}
		self.outputs = {}

	"""Extract parameters from ckpt file to npy file"""
	def extract(self, data_path, session, saver):
		raise NotImplementedError('Must be subclassed.')

	"""Load pre-trained model from numpy data_dict"""
	def load(self, data_dict, session, ignore_missing=True):
		for key in data_dict:
			with tf.variable_scope(key, reuse=True):
				for subkey in data_dict[key]:
					try:
						var = tf.get_variable(subkey)
						session.run(var.assign(data_dict[key][subkey]))
						print "Assign pretrain model " + subkey + " to " + key
					except ValueError:
						print "Ignore " + key
						if not ignore_missing:
							raise

	"""Get outputs given key names"""
	def get_output(self, key):
		if key not in self.outputs:
			raise KeyError
		return self.outputs[key]

	"""Get parameters given key names"""
	def get_param(self, key):
		if key not in self.layers:
			raise KeyError
		return self.layers[key]['weights'], self.layers[key]['biases']

	"""Add conv part of vgg16"""
	def add_conv(self, inputs):
		# Conv1
		with tf.variable_scope('conv1_1') as scope:
			w_conv1_1 = tf.get_variable('weights', [3, 3, 3, 64], 
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv1_1 = tf.get_variable('biases', [64], 
				initializer=tf.constant_initializer(0))
			z_conv1_1 = tf.nn.conv2d(inputs, w_conv1_1, strides=[1, 1, 1, 1], 
				padding='SAME') + b_conv1_1
			a_conv1_1 = tf.nn.relu(z_conv1_1)

		with tf.variable_scope('conv1_2') as scope:
			w_conv1_2 = tf.get_variable('weights', [3, 3, 64, 64], 
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv1_2 = tf.get_variable('biases', [64], 
				initializer=tf.constant_initializer(0))
			z_conv1_2 = tf.nn.conv2d(a_conv1_1, w_conv1_2, strides=[1, 1, 1, 1], 
				padding='SAME') + b_conv1_2
			a_conv1_2 = tf.nn.relu(z_conv1_2)
		
		pool1 = tf.nn.max_pool(a_conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1],
			padding='SAME', name='pool1')

		# Conv2
		with tf.variable_scope('conv2_1') as scope:
			w_conv2_1 = tf.get_variable('weights', [3, 3, 64, 128], 
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv2_1 = tf.get_variable('biases', [128], 
				initializer=tf.constant_initializer(0))
			z_conv2_1 = tf.nn.conv2d(pool1, w_conv2_1, strides=[1, 1, 1, 1], 
				padding='SAME') + b_conv2_1
			a_conv2_1 = tf.nn.relu(z_conv2_1)

		with tf.variable_scope('conv2_2') as scope:
			w_conv2_2 = tf.get_variable('weights', [3, 3, 128, 128], 
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv2_2 = tf.get_variable('biases', [128], 
				initializer=tf.constant_initializer(0))
			z_conv2_2 = tf.nn.conv2d(a_conv2_1, w_conv2_2, strides=[1, 1, 1, 1], 
				padding='SAME') + b_conv2_2
			a_conv2_2 = tf.nn.relu(z_conv2_2)

		pool2 = tf.nn.max_pool(a_conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1],
			padding='SAME', name='pool2')

		# Conv3
		with tf.variable_scope('conv3_1') as scope:
			w_conv3_1 = tf.get_variable('weights', [3, 3, 128, 256],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv3_1 = tf.get_variable('biases', [256],
				initializer=tf.constant_initializer(0))
			z_conv3_1 = tf.nn.conv2d(pool2, w_conv3_1, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv3_1
			a_conv3_1 = tf.nn.relu(z_conv3_1)

		with tf.variable_scope('conv3_2') as scope:
			w_conv3_2 = tf.get_variable('weights', [3, 3, 256, 256],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv3_2 = tf.get_variable('biases', [256],
				initializer=tf.constant_initializer(0))
			z_conv3_2 = tf.nn.conv2d(a_conv3_1, w_conv3_2, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv3_2
			a_conv3_2 = tf.nn.relu(z_conv3_2)

		with tf.variable_scope('conv3_3') as scope:
			w_conv3_3 = tf.get_variable('weights', [3, 3, 256, 256],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv3_3 = tf.get_variable('biases', [256],
				initializer=tf.constant_initializer(0))
			z_conv3_3 = tf.nn.conv2d(a_conv3_2, w_conv3_3, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv3_3
			a_conv3_3 = tf.nn.relu(z_conv3_3)

		pool3 = tf.nn.max_pool(a_conv3_3, ksize=[1,2,2,1], strides=[1,2,2,1],
			padding='SAME', name='pool3')

		# Conv4
		with tf.variable_scope('conv4_1') as scope:
			w_conv4_1 = tf.get_variable('weights', [3, 3, 256, 512],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv4_1 = tf.get_variable('biases', [512],
				initializer=tf.constant_initializer(0))
			z_conv4_1 = tf.nn.conv2d(pool3, w_conv4_1, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv4_1
			a_conv4_1 = tf.nn.relu(z_conv4_1)

		with tf.variable_scope('conv4_2') as scope:
			w_conv4_2 = tf.get_variable('weights', [3, 3, 512, 512],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv4_2 = tf.get_variable('biases', [512],
				initializer=tf.constant_initializer(0))
			z_conv4_2 = tf.nn.conv2d(a_conv4_1, w_conv4_2, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv4_2
			a_conv4_2 = tf.nn.relu(z_conv4_2)

		with tf.variable_scope('conv4_3') as scope:
			w_conv4_3 = tf.get_variable('weights', [3, 3, 512, 512],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv4_3 = tf.get_variable('biases', [512],
				initializer=tf.constant_initializer(0))
			z_conv4_3 = tf.nn.conv2d(a_conv4_2, w_conv4_3, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv4_3
			a_conv4_3 = tf.nn.relu(z_conv4_3)

		pool4 = tf.nn.max_pool(a_conv4_3, ksize=[1,2,2,1], strides=[1,2,2,1],
			padding='SAME', name='pool4')

		# Conv5
		with tf.variable_scope('conv5_1') as scope:
			w_conv5_1 = tf.get_variable('weights', [3, 3, 512, 512],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv5_1 = tf.get_variable('biases', [512],
				initializer=tf.constant_initializer(0))
			z_conv5_1 = tf.nn.conv2d(pool4, w_conv5_1, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv5_1
			a_conv5_1 = tf.nn.relu(z_conv5_1)

		with tf.variable_scope('conv5_2') as scope:
			w_conv5_2 = tf.get_variable('weights', [3, 3, 512, 512],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv5_2 = tf.get_variable('biases', [512],
				initializer=tf.constant_initializer(0))
			z_conv5_2 = tf.nn.conv2d(a_conv5_1, w_conv5_2, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv5_2
			a_conv5_2 = tf.nn.relu(z_conv5_2)

		with tf.variable_scope('conv5_3') as scope:
			w_conv5_3 = tf.get_variable('weights', [3, 3, 512, 512],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv5_3 = tf.get_variable('biases', [512],
				initializer=tf.constant_initializer(0))
			z_conv5_3 = tf.nn.conv2d(a_conv5_2, w_conv5_3, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv5_3
			a_conv5_3 = tf.nn.relu(z_conv5_3)

		# pool5 = tf.nn.max_pool(a_conv5_3, ksize=[1,2,2,1], strides=[1,2,2,1],
			# padding='SAME', name='pool5')

		# Only use conv1-conv5 pre-trained part
		"""
		# Transform fully-connected layers to convolutional layers
		with tf.variable_scope('conv6') as scope:
			w_conv6 = tf.get_variable('weights', [7, 7, 512, 4096],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv6 = tf.get_variable('biases', [4096],
				initializer=tf.constant_initializer(0))
			z_conv6 = tf.nn.conv2d(pool5, w_conv6, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv6
			a_conv6 = tf.nn.relu(z_conv6)
			d_conv6 = tf.nn.dropout(a_conv6, keep_prob)

		with tf.variable_scope('conv7') as scope:
			w_conv7 = tf.get_variable('weights', [1, 1, 4096, 4096],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv7 = tf.get_variable('biases', [4096],
				initializer=tf.constant_initializer(0))
			z_conv7 = tf.nn.conv2d(d_conv6, w_conv7, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv7
			a_conv7 = tf.nn.relu(z_conv7)
			d_conv7 = tf.nn.dropout(a_conv7, keep_prob)

		# Replace the original classifier layer
		with tf.variable_scope('conv8') as scope:
			w_conv8 = tf.get_variable('weights', [1, 1, 4096, num_classes],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv8 = tf.get_variable('biases', [num_classes],
				initializer=tf.constant_initializer(0))
			z_conv8 = tf.nn.conv2d(d_conv7, w_conv8, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv8
		"""

		# Add to store dicts
		self.outputs['conv1_1'] = a_conv1_1
		self.outputs['conv1_2'] = a_conv1_2
		self.outputs['pool1']   = pool1
		self.outputs['conv2_1'] = a_conv2_1
		self.outputs['conv2_2'] = a_conv2_2
		self.outputs['pool2']   = pool2
		self.outputs['conv3_1'] = a_conv3_1
		self.outputs['conv3_2'] = a_conv3_2
		self.outputs['conv3_3'] = a_conv3_3
		self.outputs['pool3']   = pool3
		self.outputs['conv4_1'] = a_conv4_1
		self.outputs['conv4_2'] = a_conv4_2
		self.outputs['conv4_3'] = a_conv4_3
		self.outputs['pool4']   = pool4
		self.outputs['conv5_1'] = a_conv5_1
		self.outputs['conv5_2'] = a_conv5_2
		self.outputs['conv5_3'] = a_conv5_3
		# self.outputs['pool5']   = pool5
		# self.outputs['conv6']   = d_conv6
		# self.outputs['conv7']   = d_conv7
		# self.outputs['conv8']   = z_conv8

		self.layers['conv1_1'] = {'weights':w_conv1_1, 'biases':b_conv1_1}
		self.layers['conv1_2'] = {'weights':w_conv1_2, 'biases':b_conv1_2}
		self.layers['conv2_1'] = {'weights':w_conv2_1, 'biases':b_conv2_1}
		self.layers['conv2_2'] = {'weights':w_conv2_2, 'biases':b_conv2_2}
		self.layers['conv3_1'] = {'weights':w_conv3_1, 'biases':b_conv3_1}
		self.layers['conv3_2'] = {'weights':w_conv3_2, 'biases':b_conv3_2}
		self.layers['conv3_3'] = {'weights':w_conv3_3, 'biases':b_conv3_3}
		self.layers['conv4_1'] = {'weights':w_conv4_1, 'biases':b_conv4_1}
		self.layers['conv4_2'] = {'weights':w_conv4_2, 'biases':b_conv4_2}
		self.layers['conv4_3'] = {'weights':w_conv4_3, 'biases':b_conv4_3}
		self.layers['conv5_1'] = {'weights':w_conv5_1, 'biases':b_conv5_1}
		self.layers['conv5_2'] = {'weights':w_conv5_2, 'biases':b_conv5_2}
		self.layers['conv5_3'] = {'weights':w_conv5_3, 'biases':b_conv5_3}
		# self.layers['conv6']   = {'weights':w_conv6, 'biases':b_conv6}
		# self.layers['conv7']   = {'weights':w_conv7, 'biases':b_conv7}
		# self.layers['conv8']   = {'weights':w_conv8, 'biases':b_conv8}


"""Baseline model: shortcut from conv5 to conv1"""
class BaseNet(Network):
	def __init__(self, config):
		self.batch_num = config['batch_num']
		self.weight_decay = config['weight_decay']
		self.base_lr = config['base_lr']
		self.momentum = config['momentum']

		self.img  = tf.placeholder(tf.float32, 
			[self.batch_num, None, None, 3])
		self.seg  = tf.placeholder(tf.int32, 
			[self.batch_num, None, None, 1])

		self.layers = {}
		self.outputs = {}
		self.set_up()

	def set_up(self):
		self.add_conv(self.img)
		self.add_deconv()
		self.add_predict()
		self.add_loss_op()
		self.add_weight_decay()
		self.add_train_op()

	"""Extract parameters from ckpt file to npy file"""
	def extract(self, data_path, session, saver):
		saver.restore(session, data_path)
		scopes = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1',
		'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1',
		'conv5_2', 'conv5_3']
		data_dict = {}
		for scope in scopes:
			[w, b] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
			data_dict[scope] = {'weights':w.eval(), 'biases':b.eval()}
		file_name = data_path[0:-5]
		np.save(file_name, data_dict)
		ipdb.set_trace()
		return file_name + '.npy'


	"""Add the deconv(upsampling) layer to get dense prediction"""
	def add_deconv(self):
		conv5_3 = self.get_output('conv5_3')
		conv4_3 = self.get_output('conv4_3')
		conv3_3 = self.get_output('conv3_3')
		conv2_2 = self.get_output('conv2_2')
		conv1_2 = self.get_output('conv1_2')

		shape1 = tf.shape(conv1_2)[1]
		shape2 = tf.shape(conv1_2)[2]

		# ======== Preprocessing to align feature dim ======== #
		with tf.variable_scope('align5') as scope:
			w_conv5 = tf.get_variable('weights', [1, 1, 512, 64],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv5 = tf.get_variable('biases', [64],
				initializer=tf.constant_initializer(0))
			z_conv5 = tf.nn.conv2d(conv5_3, w_conv5, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv5

		with tf.variable_scope('align4') as scope:
			w_conv4 = tf.get_variable('weights', [1, 1, 512, 64],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv4 = tf.get_variable('biases', [64],
				initializer=tf.constant_initializer(0))
			z_conv4 = tf.nn.conv2d(conv4_3, w_conv4, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv4

		with tf.variable_scope('align3') as scope:
			w_conv3 = tf.get_variable('weights', [1, 1, 256, 64],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv3 = tf.get_variable('biases', [64],
				initializer=tf.constant_initializer(0))
			z_conv3 = tf.nn.conv2d(conv3_3, w_conv3, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv3

		with tf.variable_scope('align2') as scope:
			w_conv2 = tf.get_variable('weights', [1, 1, 128, 64],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv2 = tf.get_variable('biases', [64],
				initializer=tf.constant_initializer(0))
			z_conv2 = tf.nn.conv2d(conv2_2, w_conv2, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv2

		# ======== Upsample and skip connection ======== #
		with tf.variable_scope('deconv5') as scope:
			# Using fiexed bilinearing upsampling filter
			w_deconv5 = tf.get_variable('weights', trainable=False, 
				initializer=bilinear_upsample_weights(32, 64))

			z_deconv5 = tf.nn.conv2d_transpose(z_conv5, w_deconv5, 
				[self.batch_num, shape1, shape2, 64],
				strides=[1,16,16,1], padding='SAME', name='z')

		with tf.variable_scope('deconv4') as scope:
			# Using fiexed bilinearing upsampling filter
			w_deconv4 = tf.get_variable('weights', trainable=False, 
				initializer=bilinear_upsample_weights(16, 64))

			z_deconv4 = tf.nn.conv2d_transpose(z_conv4, w_deconv4, 
				[self.batch_num, shape1, shape2, 64],
				strides=[1,8,8,1], padding='SAME', name='z')

		with tf.variable_scope('deconv3') as scope:
			# Using fiexed bilinearing upsampling filter
			w_deconv3 = tf.get_variable('weights', trainable=False, 
				initializer=bilinear_upsample_weights(8, 64))

			z_deconv3 = tf.nn.conv2d_transpose(z_conv3, w_deconv3, 
				[self.batch_num, shape1, shape2, 64],
				strides=[1,4,4,1], padding='SAME', name='z')

		with tf.variable_scope('deconv2') as scope:
			# Using fiexed bilinearing upsampling filter
			w_deconv2 = tf.get_variable('weights', trainable=False, 
				initializer=bilinear_upsample_weights(4, 64))

			z_deconv2 = tf.nn.conv2d_transpose(z_conv2, w_deconv2, 
				[self.batch_num, shape1, shape2, 64],
				strides=[1,2,2,1], padding='SAME', name='z')

		# Old version API
		# merge = tf.concat(3, [conv1_2, z_deconv2, z_deconv3, z_deconv4, z_deconv5])
		merge = conv1_2 + z_deconv2 + z_deconv3 + z_deconv4 + z_deconv5

		# Add to store dicts
		self.outputs['align5'] = z_conv5
		self.outputs['align4'] = z_conv4
		self.outputs['align3'] = z_conv3
		self.outputs['align2'] = z_conv2
		self.outputs['deconv5'] = z_deconv5
		self.outputs['deconv4'] = z_deconv4
		self.outputs['deconv3'] = z_deconv3
		self.outputs['deconv2'] = z_deconv2
		self.outputs['merge'] = merge

		self.layers['align5'] = {'weights':w_conv5, 'biases':b_conv5}
		self.layers['align4'] = {'weights':w_conv4, 'biases':b_conv4}
		self.layers['align3'] = {'weights':w_conv3, 'biases':b_conv3}
		self.layers['align2'] = {'weights':w_conv2, 'biases':b_conv2}
		# Fixed parameters, don't need to track
		# self.layers['deconv5']  = {'weights':w_deconv5, 'biases':None}
		# self.layers['deconv4']  = {'weights':w_deconv4, 'biases':None}
		# self.layers['deconv3']  = {'weights':w_deconv3, 'biases':None}
		# self.layers['deconv2']  = {'weights':w_deconv2, 'biases':None}

	"""Add final conv layer to predict"""
	def add_predict(self):
		merge = self.get_output('merge')

		with tf.variable_scope('predict') as scope:
			w = tf.get_variable('weights', [3, 3, 64, 2],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b = tf.get_variable('biases', [2],
				initializer=tf.constant_initializer(0))
			z = tf.nn.conv2d(merge, w, strides= [1, 1, 1, 1],
				padding='SAME') + b

		# Add to store dicts
		self.outputs['predict'] = z
		self.layers['predict']  = {'weights':w, 'biases':b}

	"""Add pixelwise softmax loss"""
	def add_loss_op(self):
		pred = self.get_output('predict')
		pred_reshape = tf.reshape(pred, [-1, 2])
		gt_reshape = tf.reshape(self.seg, [-1])

		shape1 = tf.shape(pred)[1]
		shape2 = tf.shape(pred)[2]

		Y_pos = tf.to_float(tf.reduce_sum(gt_reshape))
		beta = 1 - Y_pos / tf.to_float(self.batch_num * shape1 * shape2)

		loss_reshape = tf.nn.sparse_softmax_cross_entropy_with_logits(pred_reshape, gt_reshape)
		# loss = tf.reshape(loss_reshape, [self.batch_num, shape1, shape2, 1])
		idx_pos = tf.to_int32(tf.where(tf.equal(gt_reshape, 1)))
		loss_pos = beta * tf.gather(loss_reshape, idx_pos)
		idx_neg = tf.to_int32(tf.where(tf.equal(gt_reshape, 0)))
		loss_neg = (1 - beta) * tf.gather(loss_reshape, idx_neg)

		loss_avg = tf.reduce_mean(loss_pos) + tf.reduce_mean(loss_neg)

		self.loss = loss_avg

	"""Add weight decay"""
	def add_weight_decay(self):
		for key in self.layers:
			w = self.layers[key]['weights']
			self.loss += self.weight_decay * tf.nn.l2_loss(w)

	"""Set up training optimization"""
	def add_train_op(self):
		self.train_op = tf.train.MomentumOptimizer(self.base_lr, 
			self.momentum).minimize(self.loss)
		# self.train_op = tf.train.AdamOptimizer(self.base_lr).minimize(self.loss)


"""Baseline model + ConvLSTM"""
class ConvBaseNet(BaseNet):
	def __init__(self, config):
		# Different behaviors during trainig/test
		self.mode = config['mode']

		BaseNet.__init__(self, config)
		
	"""Override"""
	def set_up(self):
		self.add_conv(self.img)
		self.add_deconv()
		# Add ConvLSTM
		self.add_conv_lstm()
		self.add_predict()
		self.add_loss_op()
		self.add_weight_decay()
		self.add_train_op()

	"""Add ConvLSTM to model temporal relationship"""
	def add_conv_lstm(self):
		input = self.get_output('merge')

		num_features = 64
		forget_bias = 1.0
		filter_size = 3
		activation = tf.nn.tanh

		with tf.variable_scope('conv_lstm') as scope:
			w = tf.get_variable('weights', [filter_size, filter_size, 
				2 * num_features, 4 * num_features], 
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b = tf.get_variable('biases', [4*num_features], 
				initializer=tf.constant_initializer(0))

		# Add to store dict
		self.layers['conv_lstm'] = {'weights':w, 'biases':b}

		shape1 = tf.shape(input)[1]
		shape2 = tf.shape(input)[2]

		# Training
		if self.mode == 'TRAIN':
			# Initialize with zero states
			cell = tf.zeros([1,shape1,shape2,num_features])
			hidd = tf.zeros([1,shape1,shape2,num_features])
			output = []

			for i in range(self.batch_num):
				concat = tf.nn.conv2d(tf.concat(3, [input[i:i+1,:,:,:], hidd]),
					w, strides=[1,1,1,1], padding='SAME') + b
				i, j, f, o = tf.split(3, 4, concat)

				new_cell = (cell * tf.nn.sigmoid(f + forget_bias) + 
					tf.nn.sigmoid(i) * activation(j))
				new_hidd = activation(new_cell) + tf.nn.sigmoid(o)
				output.append(new_hidd)

				cell = new_cell
				hidd = new_hidd

				if i == 0:
					tf.get_variable_scope('conv_lstm').reuse_variables()

			conv_lstm = tf.concat(0, output)
			self.outputs['conv_lstm'] = conv_lstm

		# Test
		elif self.mode == 'TEST':
			# Add placeholder
			self.cell = tf.placeholder(tf.float32, 
				[1, None, None, num_features])
			self.hidd = tf.placeholder(tf.float32,
				[1, None, None, num_features])

			concat = tf.nn.conv2d(tf.concat(3, [input, self.hidd]), w,
				strides=[1,1,1,1], padding='SAME') + b
			i, j, f, o = tf.split(3, 4, concat)

			new_cell = (self.cell * tf.nn.sigmoid(f + forget_bias) + 
				tf.nn.sigmoid(i) * activation(j))
			new_hidd = activation(new_cell) + tf.nn.sigmoid(o)

			self.outputs['conv_lstm'] = new_hidd
			self.outputs['cell'] = new_cell
		else:
			raise KeyError


	"""Override: Add final conv layer to predict"""
	def add_predict(self):
		conv_lstm = self.get_output('conv_lstm')

		with tf.variable_scope('predict') as scope:
			w = tf.get_variable('weights', [3, 3, 64, 2],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b = tf.get_variable('biases', [2],
				initializer=tf.constant_initializer(0))
			z = tf.nn.conv2d(conv_lstm, w, strides= [1, 1, 1, 1],
				padding='SAME') + b

		# Add to store dicts
		self.outputs['predict'] = z
		self.layers['predict']  = {'weights':w, 'biases':b}


"""Analogy-making model"""
class AnalogyNet(Network):
	def __init__(self, config):
		self.batch_num = config['batch_num']
		self.weight_decay = config['weight_decay']
		self.base_lr = config['base_lr']
		self.momentum = config['momentum']

		# batch_num * (A, B) pairs
		self.img  = tf.placeholder(tf.float32, 
			[self.batch_num, 2, None, None, 3])
		self.seg  = tf.placeholder(tf.int32, 
			[self.batch_num, 2, None, None, 1])

		self.layers = {}
		self.outputs = {}
		self.set_up()

	def set_up(self):
		self.add_img_enc()
		self.add_seg_enc()
		self.add_dec()
		self.add_loss_op()
		self.add_train_op()

	"""Override: Load pre-trained model from numpy data_dict"""
	def load(self, data_dict, session, name='img_enc_', ignore_missing=True):
		for key in data_dict:
			with tf.variable_scope(name + key, reuse=True):
				for subkey in data_dict[key]:
					try:
						# A bit different here
						var = self.layers[name + key][subkey]
						session.run(var.assign(data_dict[key][subkey]))
						print "Assign pretrain model " + subkey + " to " + name + key
					except:
						print "Ignore " + name + key
						if not ignore_missing:
							raise

	def _add_vgg_conv(self, inputs, name=''):
		if name == 'seg_enc_':
			input_dim = 1
		else:
			input_dim = 3

		# Conv1
		with tf.variable_scope(name + 'conv_1_1') as scope:
			w_conv1_1 = tf.get_variable('weights', [3, 3, input_dim, 64], 
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv1_1 = tf.get_variable('biases', [64], 
				initializer=tf.constant_initializer(0))
			z_conv1_1 = tf.nn.conv2d(inputs, w_conv1_1, strides=[1, 1, 1, 1], 
				padding='SAME') + b_conv1_1
			a_conv1_1 = tf.nn.relu(z_conv1_1)

		with tf.variable_scope(name + 'conv_1_2') as scope:
			w_conv1_2 = tf.get_variable('weights', [3, 3, 64, 64], 
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv1_2 = tf.get_variable('biases', [64], 
				initializer=tf.constant_initializer(0))
			z_conv1_2 = tf.nn.conv2d(a_conv1_1, w_conv1_2, strides=[1, 1, 1, 1], 
				padding='SAME') + b_conv1_2
			a_conv1_2 = tf.nn.relu(z_conv1_2)
		
		pool1 = tf.nn.max_pool(a_conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1],
			padding='SAME', name=name+'pool1')

		# Conv2
		with tf.variable_scope(name + 'conv_2_1') as scope:
			w_conv2_1 = tf.get_variable('weights', [3, 3, 64, 128], 
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv2_1 = tf.get_variable('biases', [128], 
				initializer=tf.constant_initializer(0))
			z_conv2_1 = tf.nn.conv2d(pool1, w_conv2_1, strides=[1, 1, 1, 1], 
				padding='SAME') + b_conv2_1
			a_conv2_1 = tf.nn.relu(z_conv2_1)

		with tf.variable_scope(name + 'conv_2_2') as scope:
			w_conv2_2 = tf.get_variable('weights', [3, 3, 128, 128], 
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv2_2 = tf.get_variable('biases', [128], 
				initializer=tf.constant_initializer(0))
			z_conv2_2 = tf.nn.conv2d(a_conv2_1, w_conv2_2, strides=[1, 1, 1, 1], 
				padding='SAME') + b_conv2_2
			a_conv2_2 = tf.nn.relu(z_conv2_2)

		pool2 = tf.nn.max_pool(a_conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1],
			padding='SAME', name=name+'pool2')

		# Conv3
		with tf.variable_scope(name + 'conv_3_1') as scope:
			w_conv3_1 = tf.get_variable('weights', [3, 3, 128, 256],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv3_1 = tf.get_variable('biases', [256],
				initializer=tf.constant_initializer(0))
			z_conv3_1 = tf.nn.conv2d(pool2, w_conv3_1, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv3_1
			a_conv3_1 = tf.nn.relu(z_conv3_1)

		with tf.variable_scope(name + 'conv_3_2') as scope:
			w_conv3_2 = tf.get_variable('weights', [3, 3, 256, 256],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv3_2 = tf.get_variable('biases', [256],
				initializer=tf.constant_initializer(0))
			z_conv3_2 = tf.nn.conv2d(a_conv3_1, w_conv3_2, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv3_2
			a_conv3_2 = tf.nn.relu(z_conv3_2)

		with tf.variable_scope(name + 'conv_3_3') as scope:
			w_conv3_3 = tf.get_variable('weights', [3, 3, 256, 256],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_conv3_3 = tf.get_variable('biases', [256],
				initializer=tf.constant_initializer(0))
			z_conv3_3 = tf.nn.conv2d(a_conv3_2, w_conv3_3, strides= [1, 1, 1, 1],
				padding='SAME') + b_conv3_3
			a_conv3_3 = tf.nn.relu(z_conv3_3)

		pool3 = tf.nn.max_pool(a_conv3_3, ksize=[1,2,2,1], strides=[1,2,2,1],
			padding='SAME', name=name+'pool3')

		# Add to store dicts
		self.outputs[name+'conv_1_1'] = a_conv1_1
		self.outputs[name+'conv_1_2'] = a_conv1_2
		self.outputs[name+'pool1']    = pool1
		self.outputs[name+'conv_2_1'] = a_conv2_1
		self.outputs[name+'conv_2_2'] = a_conv2_2
		self.outputs[name+'pool2']    = pool2
		self.outputs[name+'conv_3_1'] = a_conv3_1
		self.outputs[name+'conv_3_2'] = a_conv3_2
		self.outputs[name+'conv_3_3'] = a_conv3_3
		self.outputs[name+'pool3']    = pool3

		self.layers[name+'conv1_1'] = {'weights':w_conv1_1, 'biases':b_conv1_1}
		self.layers[name+'conv1_2'] = {'weights':w_conv1_2, 'biases':b_conv1_2}
		self.layers[name+'conv2_1'] = {'weights':w_conv2_1, 'biases':b_conv2_1}
		self.layers[name+'conv2_2'] = {'weights':w_conv2_2, 'biases':b_conv2_2}
		self.layers[name+'conv3_1'] = {'weights':w_conv3_1, 'biases':b_conv3_1}
		self.layers[name+'conv3_2'] = {'weights':w_conv3_2, 'biases':b_conv3_2}
		self.layers[name+'conv3_3'] = {'weights':w_conv3_3, 'biases':b_conv3_3}

	""" Image encoder """
	def add_img_enc(self):
		shape2 = tf.shape(self.img)[2]
		shape3 = tf.shape(self.img)[3]
		temp = tf.reshape(self.img, [self.batch_num * 2, shape2, shape3, 3])
		self._add_vgg_conv(temp, name='img_enc_')

	""" Segmentation encoder """
	def add_seg_enc(self):
		shape2 = tf.shape(self.seg)[2]
		shape3 = tf.shape(self.seg)[3]
		# Need to use float32 for conv
		temp = tf.to_float(tf.reshape(self.seg, [self.batch_num * 2, shape2, shape3, 1]))
		self._add_vgg_conv(temp, name='seg_enc_')

	""" Decoder """
	def add_dec(self):
		img_emb = self.get_output('img_enc_pool3')
		seg_emb = self.get_output('seg_enc_pool3')

		shape1 = tf.shape(img_emb)[1]
		shape2 = tf.shape(img_emb)[2]

		img_emb_reshape = tf.reshape(img_emb, [self.batch_num, 2, shape1, shape2, -1])
		seg_emb_reshape = tf.reshape(seg_emb, [self.batch_num, 2, shape1, shape2, -1])

		imgA = img_emb_reshape[:,0,:,:,:]
		imgB = img_emb_reshape[:,1,:,:,:]

		segA = seg_emb_reshape[:,0,:,:,:]
		segB = seg_emb_reshape[:,1,:,:,:]

		# imgA - imgB + segB ~= segA
		ABB = imgA - imgB + segB
		# imgB - imgA + segA ~= segB
		BAA = imgB - imgA + segA

		pred_emb = tf.concat(0, [ABB, BAA])

		temp = self.get_output('img_enc_pool2')
		shape1 = tf.shape(temp)[1]
		shape2 = tf.shape(temp)[2]

		# Upsample deconv
		with tf.variable_scope('deconv1_1') as scope:
			w_deconv1_1 = tf.get_variable('weights', [4, 4, 256, 256], 
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_deconv1_1 = tf.get_variable('biases', [256], 
				initializer=tf.constant_initializer(0))
			z_deconv1_1 = tf.nn.conv2d_transpose(pred_emb, w_deconv1_1, 
				[2*self.batch_num, shape1, shape2, 256], strides=[1,2,2,1],
				padding='SAME', name='z') + b_deconv1_1
			a_deconv1_1 = tf.nn.relu(z_deconv1_1)

		with tf.variable_scope('deconv1_2') as scope:
			w_deconv1_2 = tf.get_variable('weights', [3, 3, 256, 256], 
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_deconv1_2 = tf.get_variable('biases', [256], 
				initializer=tf.constant_initializer(0))
			z_deconv1_2 = tf.nn.conv2d_transpose(a_deconv1_1, w_deconv1_2, 
				[2*self.batch_num, shape1, shape2, 256], strides=[1,1,1,1],
				padding='SAME', name='z') + b_deconv1_2
			a_deconv1_2 = tf.nn.relu(z_deconv1_2)

		with tf.variable_scope('deconv1_3') as scope:
			w_deconv1_3 = tf.get_variable('weights', [3, 3, 256, 256], 
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_deconv1_3 = tf.get_variable('biases', [256], 
				initializer=tf.constant_initializer(0))
			z_deconv1_3 = tf.nn.conv2d_transpose(a_deconv1_2, w_deconv1_3, 
				[2*self.batch_num, shape1, shape2, 256], strides=[1,1,1,1],
				padding='SAME', name='z') + b_deconv1_3
			a_deconv1_3 = tf.nn.relu(z_deconv1_3)

		
		temp = self.get_output('img_enc_pool1')
		shape1 = tf.shape(temp)[1]
		shape2 = tf.shape(temp)[2]

		# Upsample deconv
		with tf.variable_scope('deconv2_1') as scope:
			w_deconv2_1 = tf.get_variable('weights', [4, 4, 128, 256], 
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_deconv2_1 = tf.get_variable('biases', [128], 
				initializer=tf.constant_initializer(0))
			z_deconv2_1 = tf.nn.conv2d_transpose(a_deconv1_3, w_deconv2_1, 
				[2*self.batch_num, shape1, shape2, 128], strides=[1,2,2,1],
				padding='SAME', name='z') + b_deconv2_1
			a_deconv2_1 = tf.nn.relu(z_deconv2_1)

		with tf.variable_scope('deconv2_2') as scope:
			w_deconv2_2 = tf.get_variable('weights', [3, 3, 128, 128], 
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_deconv2_2 = tf.get_variable('biases', [128], 
				initializer=tf.constant_initializer(0))
			z_deconv2_2 = tf.nn.conv2d_transpose(a_deconv2_1, w_deconv2_2, 
				[2*self.batch_num, shape1, shape2, 128], strides=[1,1,1,1],
				padding='SAME', name='z') + b_deconv2_2
			a_deconv2_2 = tf.nn.relu(z_deconv2_2)

		shape1 = tf.shape(self.img)[2]
		shape2 = tf.shape(self.img)[3]

		# Upsample deconv
		with tf.variable_scope('deconv3_1') as scope:
			w_deconv3_1 = tf.get_variable('weights', [4, 4, 64, 128], 
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_deconv3_1 = tf.get_variable('biases', [64], 
				initializer=tf.constant_initializer(0))
			z_deconv3_1 = tf.nn.conv2d_transpose(a_deconv2_2, w_deconv3_1, 
				[2*self.batch_num, shape1, shape2, 64], strides=[1,2,2,1],
				padding='SAME', name='z') + b_deconv3_1
			a_deconv3_1 = tf.nn.relu(z_deconv3_1)

		with tf.variable_scope('deconv3_2') as scope:
			w_deconv3_2 = tf.get_variable('weights', [3, 3, 2, 64], 
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_deconv3_2 = tf.get_variable('biases', [2], 
				initializer=tf.constant_initializer(0))
			z_deconv3_2 = tf.nn.conv2d_transpose(a_deconv3_1, w_deconv3_2, 
				[2*self.batch_num, shape1, shape2, 2], strides=[1,1,1,1],
				padding='SAME', name='z') + b_deconv3_2	

		# Add to store dicts
		self.outputs['deconv_1_1'] = a_deconv1_1
		self.outputs['deconv_1_2'] = a_deconv1_2
		self.outputs['deconv_1_3'] = a_deconv1_3
		self.outputs['deconv_2_1'] = a_deconv2_1
		self.outputs['deconv_2_2'] = a_deconv2_2
		self.outputs['deconv_3_1'] = a_deconv3_1
		self.outputs['deconv_3_2'] = z_deconv3_2

		self.layers['deconv1_1'] = {'weights':w_deconv1_1, 'biases':b_deconv1_1}
		self.layers['deconv1_2'] = {'weights':w_deconv1_2, 'biases':b_deconv1_2}
		self.layers['deconv1_3'] = {'weights':w_deconv1_3, 'biases':b_deconv1_3}
		self.layers['deconv2_1'] = {'weights':w_deconv2_1, 'biases':b_deconv2_1}
		self.layers['deconv2_2'] = {'weights':w_deconv2_2, 'biases':b_deconv2_2}
		self.layers['deconv3_1'] = {'weights':w_deconv3_1, 'biases':b_deconv3_1}
		self.layers['deconv3_2'] = {'weights':w_deconv3_2, 'biases':b_deconv3_2}

	"""Add pixelwise softmax loss"""
	def add_loss_op(self):
		pred = self.get_output('deconv_3_2')
		pred_A = pred[:self.batch_num,:,:,:]
		pred_B = pred[self.batch_num:,:,:,:]

		seg_A = self.seg[:,0,:,:,:]
		seg_B = self.seg[:,1,:,:,:]

		shape1 = tf.shape(pred)[1]
		shape2 = tf.shape(pred)[2]

		pred_reshape = tf.reshape(tf.concat(0, [pred_A, pred_B]), [-1, 2])
		seg_reshape = tf.reshape(tf.concat(0, [seg_A, seg_B]), [-1])
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(pred_reshape, seg_reshape)

		self.loss = tf.reduce_mean(loss)

	"""Set up training optimization"""
	def add_train_op(self):
		self.train_op = tf.train.MomentumOptimizer(self.base_lr, 
			self.momentum).minimize(self.loss)

if __name__ == '__main__':
	config = {
	'batch_num':1, # Must be 1 for ConvBaseNet
	'iter':100000, 
	'weight_decay': 0.0005,
	'base_lr': 0.0001,
	'momentum': 0.9,
	'mode': 'TEST'
	}

	# model = BaseNet(config)
	# model = ConvBaseNet(config)
	model = AnalogyNet(config)

