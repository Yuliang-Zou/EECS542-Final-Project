# Training code
# Author: Yuliang Zou
#         ylzou@umich.edu
# Date:   2017-03-29

import numpy as np
import tensorflow as tf
from Model import AnalogyNet
from Dataloader import DAVIS_pair_dataloader
try:
	import ipdb
except:
	import pdb as ipdb


config = {
'batch_num':2, 
'iter': 70000,
'weight_decay': 0.0005,
'base_lr': 1e-3,
'momentum': 0.9
}

if __name__ == '__main__':
	# Load pre-trained model
	model_path = '../model/VGG_imagenet.npy'
	data_dict = np.load(model_path).item()

	# Set up model and data loader
	model = AnalogyNet(config)
	loss_list = []
	f = open('./AnalogyNet.txt', 'w')
	DECAY = False    # decay flag
	init = tf.initialize_all_variables()

	data_loader = DAVIS_pair_dataloader(config, split='train')
	num_iters = config['iter']

	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
		session.run(init)
		model.load(data_dict, session)
		saver = tf.train.Saver()

		loss = 0
		for i in xrange(num_iters):
			minibatch = data_loader.get_next_minibatch()
			feed_dict = {model.img: minibatch[0],
						model.seg: minibatch[1]}
			eval_list = [model.train_op, model.loss, model.loss_cross, model.loss_autoencoder]

			# _, temp_loss = session.run([model.train_op, model.loss], feed_dict=feed_dict)
			# loss += temp_loss

			val_list = session.run(eval_list, feed_dict=feed_dict)
			temp_loss = val_list[1]
			loss_list.append(temp_loss)
			f.write(str(temp_loss) + '\n')
			print('Epoch:[%d], Iter:[%d/%d]. Cross Loss: %.4f, Auto Loss: %.4f' 
				% (data_loader.get_epoch(), 
					i+1, config['iter'],
					val_list[2], val_list[3]))

			# Learning rate decay
			# if len(loss_list) > 100 and not DECAY:
			# 	avg = sum(loss_list[-100::]) / 100.0
			# 	if avg <= 0.4:
			# 		model.base_lr /= 10
			# 		DECAY = True

			# Monitor
			if (i+1) % 20 == 0 and i != 0:
				loss /= 20
				print 'Iter: {}'.format(i+1) + '/{}'.format(num_iters) + ', loss = ' + str(loss)					
				loss = 0

			# Write to saver
			if i % 5000 == 0 and i != 0:
				saver.save(session, '../model/AnalogyNet_iter_'+str(i)+'.ckpt')

	f.close()

