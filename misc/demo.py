# Demo
# Author: Yuliang Zou
#         ylzou@umich.edu
# Date:   2017-03-27

import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from Model import BaseNet
from Dataloader import DAVIS_dataloader

try:
	import ipdb
except:
	import pdb as ipdb

# BGR mean pixel value
MEAN_PIXEL = np.array([103.939, 116.779, 123.68])

config = {
'batch_num':1, 
'epoch':10, 
'weight_decay': 0.0005,
'base_lr': 1e-8,
'momentum': 0.9
}

if __name__ == '__main__':
	model = BaseNet(config)
	data_loader = DAVIS_dataloader(config, split='val')

	saver = tf.train.Saver()
	ckpt  = '../model/BaseNet_iter_49919.ckpt'
	# ckpt = '../model/BaseNet_balanced_iter_45000.ckpt'
	# ckpt = '../model/BaseNet_aug_balanced_iter_50000.ckpt'
	# Extract ckpt into npy, if needed
	# with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
		# model.extract(ckpt, session, saver)
	# ipdb.set_trace()

	dump_path = '../demo/'

	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
		saver.restore(session, ckpt)
		print 'Model restored.'

		# minibatch = data_loader.get_next_minibatch()
		im_name = '../../example/crop_0_1.png'
		im = cv2.imread(im_name) - np.array([103.939, 116.779, 123.68])
		im_blob = np.array([im])
		# feed_dict = {model.img: minibatch[0]}
		feed_dict = {model.img: im_blob}
		pred      = session.run(model.get_output('predict'), feed_dict=feed_dict)

		for i in range(config['batch_num']):
			# img = minibatch[0][i] + MEAN_PIXEL
			# gt  = minibatch[1][i]
			temp = pred[i,:,:].reshape((-1, 2))
			temp = np.argmax(temp, axis=1)
			shape = pred[i,:,:,0].shape
			seg = temp.reshape(shape)
			# f, (ax1, ax2, ax3) = plt.subplots(1, 3)
			# ax1.imshow(img[:,:,::-1])
			# ax2.imshow(gt[:,:,0])
			# ax3.imshow(seg)
			plt.imshow(seg)
			plt.show()
			# plt.savefig(dump_path + str(i) + '.png')



