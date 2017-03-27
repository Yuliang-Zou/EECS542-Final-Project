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

		loss_reshape = tf.nn.sparse_softmax_cross_entropy_with_logits(pred_reshape, gt_reshape)
		loss = tf.reshape(loss_reshape, [self.batch_num, shape1, shape2, 1])
		# loss_valid = tf.reduce_sum(loss * self.mask, (1,2,3))

		# valid_pixels = tf.reduce_sum(self.mask, (1,2,3))
		# loss_avg = tf.reduce_mean(loss_valid / valid_pixels)
		loss_avg = tf.reduce_mean(loss)

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


if __name__ == '__main__':
	config = {
	'batch_num':5, 
	'iter':100000, 
	'weight_decay': 0.0005,
	'base_lr': 0.0001,
	'momentum': 0.9
	}

	model = BaseNet(config)

