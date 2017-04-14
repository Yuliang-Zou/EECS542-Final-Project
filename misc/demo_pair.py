# Demo
# Author: Yuliang Zou
#         ylzou@umich.edu
# Date:   2017-03-27

import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from Model import AnalogyNet
from Dataloader import DAVIS_pair_dataloader, DAVIS_test_dataloader

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
'mode': 'TRAIN'
}

if __name__ == '__main__':
	model = AnalogyNet(config)
	data_loader = DAVIS_test_dataloader(split='val')

	saver = tf.train.Saver()
	ckpt = '../model/AnalogyNet_iter_65000.ckpt'
	# Extract ckpt into npy, if needed
	# with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
		# model.extract(ckpt, session, saver)
	# ipdb.set_trace()

	dump_path = '../demo/'

	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
		saver.restore(session, ckpt)
		print 'Model restored.'

		while data_loader.has_next:
			minibatch = data_loader.get_next_minibatch()
			feed_dict = {model.img: minibatch[0],
						model.seg: minibatch[1]}
			pred      = session.run(model.get_output('deconv_3_2'), feed_dict=feed_dict)

			src_img = minibatch[0][0,0,:,:,:] + MEAN_PIXEL
			src_seg = minibatch[1][0,0,:,:,0]
			des_img = minibatch[0][0,1,:,:,:] + MEAN_PIXEL
			des_seg = minibatch[1][0,1,:,:,0]
			des_pred = np.argmax(pred[1,:,:,:], axis=2)

			f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
			ax1.imshow(src_img[:,:,::-1])
			ax2.imshow(src_seg)
			ax3.imshow(des_img[:,:,::-1])
			ax4.imshow(des_seg)
			ax5.imshow(des_pred)
			plt.show()
			# plt.savefig(dump_path + str(i) + '.png')
			ipdb.set_trace()



