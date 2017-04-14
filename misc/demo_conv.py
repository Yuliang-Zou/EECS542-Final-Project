# Demo
# Author: Yuliang Zou
#         ylzou@umich.edu
# Date:   2017-03-27

import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from Model import ConvBaseNet
from Dataloader import DAVIS_test_dataloader

try:
	import ipdb
except:
	import pdb as ipdb

# BGR mean pixel value
MEAN_PIXEL = np.array([103.939, 116.779, 123.68])

config = {
'batch_num':1,
'iter':100000, 
'weight_decay': 0.0005,
'base_lr': 0.0001,
'momentum': 0.9,
'mode': 'TEST'
}

if __name__ == '__main__':
	model = ConvBaseNet(config)
	data_loader = DAVIS_test_dataloader()

	saver = tf.train.Saver()
	ckpt = '../model/ConvBaseNet_iter_25000.ckpt'
	# Extract ckpt into npy, if needed
	# with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
		# model.extract(ckpt, session, saver)
	# ipdb.set_trace()

	dump_path = '../demo/'

	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
		saver.restore(session, ckpt)
		print 'Model restored.'

		for i in range(data_loader.num_seq):
			while data_loader.temp_pointer == i:
				minibatch = data_loader.get_next_minibatch()
				# Initialize with zero states
				if data_loader.inside_pointer == 2:
					shape1 = minibatch[0].shape[2]
					shape2 = minibatch[0].shape[3]
					cell = np.zeros((1,shape1,shape2,64))
					hidd = np.zeros((1,shape1,shape2,64))

				feed_dict = {model.img: minibatch[0][:,1,:,:,:],
							model.seg: minibatch[1][:,1,:,:,:],
							model.cell: cell, model.hidd: hidd}
				pred, hidd, cell = session.run([model.get_output('predict'), 
					model.get_output('conv_lstm'), model.get_output('cell')], 
					feed_dict=feed_dict)

				img = minibatch[0][0,1,:,:,:] + MEAN_PIXEL
				seg = minibatch[1][0,1,:,:,0]
				pred = np.argmax(pred[0,:,:,:], axis=2)

				f, (ax1, ax2, ax3) = plt.subplots(1, 3)
				ax1.imshow(img[:,:,::-1])
				ax2.imshow(seg)
				ax3.imshow(pred)
				plt.show()
				# plt.savefig(dump_path + str(i) + '.png')
				ipdb.set_trace()



