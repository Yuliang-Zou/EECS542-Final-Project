# Define the data loader
# Author: Yuliang Zou
#         ylzou@umich.edu
# Date:   2017-03-24

try:
	import ipdb
except:
	import pdb as ipdb

import numpy as np
from os.path import join
import cv2
import glob

# import multiprocessing


# BGR mean pixel value for VGG model
MEAN_PIXEL = np.array([103.939, 116.779, 123.68])

"""Return corresponding dataloader according to input config"""
def get_dataloader(config):
	if 'DAVIS' in config['dataset']:
		temp = config['dataset'].split('_')
		res = temp[1]
		split = temp[2]
		return DAVIS_dataloader(config, res, split)

"""
The dataloader for DAVIS

TODO: 1080p resolution need padding; multiprocessing version
"""
class DAVIS_dataloader(object):
	def __init__(self, config, res='480p', split='train'):
		# Validate split input
		if split != 'train' and split != 'val' and split != 'trainval':
			raise Exception('Please enter valid split variable!')

		# Validate resolution
		if res != '480p':
			raise Exception('Please enter valid resolution!')

		# Note that ImageSet format is a bit different from VOC
		self.root = '../data/DAVIS/'
		img_set = self.root + 'ImageSets/' + res + '/' + split + '.txt'
		with open(img_set) as f:
			self.img_list = f.read().rstrip().split('\n')

		self.num_images = len(self.img_list)
		self.temp_pointer = 0
		self.epoch = 0
		self.step = 0
		self._shuffle()

		self.batch_num = config['batch_num']

	def _shuffle(self):
		self.img_list = np.random.permutation(self.img_list)

	def _img_at(self, i):
		return self.root + self.img_list[i].split(' ')[0]

	def _seg_at(self, i):
		return self.root + self.img_list[i].split(' ')[1]

	def step_per_epoch(self):
		return (self.num_images / self.batch_num) + 1

	def get_epoch(self):
		return self.epoch

	def get_step(self):
		return self.step

	def get_next_minibatch(self):
		img_blobs = []
		seg_blobs = []
		shuffle_flag = False
		for i in range(self.batch_num):
			img = cv2.imread(self._img_at(self.temp_pointer)) - MEAN_PIXEL
			seg = cv2.imread(self._seg_at(self.temp_pointer), cv2.CV_LOAD_IMAGE_GRAYSCALE) / 255

			img_blobs.append(img)
			seg_blobs.append(seg)
			self.temp_pointer += 1

			if self.temp_pointer >= self.num_images:
				self.temp_pointer = 0
				shuffle_flag = True

		if shuffle_flag:
			self._shuffle()
			self.epoch += 1
			self.step = 0
		else:
			self.step += 1

		return np.array(img_blobs), np.array(np.expand_dims(seg_blobs, axis=3))

"""
The dataloader for augmented data
"""
class augment_dataloader(DAVIS_dataloader):
	def __init__(self, batch_num):
		# Note that ImageSet format is a bit different from VOC
		self.root = '../data/'
		img_set = self.root + 'augment/DAVIS.txt'
		with open(img_set) as f:
			self.img_list = f.read().rstrip().split('\n')

		self.num_images = len(self.img_list)
		self.temp_pointer = 0
		self.epoch = 0
		self.step = 0
		self._shuffle()

		self.batch_num = batch_num

"""
The sequence dataloader for DAVIS

TODO: multiprocessing version
"""
class DAVIS_seq_dataloader(object):
	def __init__(self, split='train'):
		self.root = '../data/DAVIS/'
		seq_set = self.root + 'SequenceSets/' + split + '.txt'
		with open(seq_set) as f:
			self.seq_list = f.read().rstrip().split('\n')

		self.num_seq = len(self.seq_list)
		self.temp_pointer = 0
                self.out_pointer = 0
		self.epoch = 0

		# Get length of each sequence
		self.seq_len = {}
		for seq in self.seq_list:
			temp = glob.glob(self.root + 'JPEGImages/480p/' + seq + '*.jpg')
			self.seq_len[seq] = len(temp)

		self.batch_num = 1
		# self._shuffle()

	def _shuffle(self):
		self.seq_list = np.random.permutation(self.seq_list)
		self.temp_pointer = 0

	def _img_at(self, name):
		return self.root + 'JPEGImages/480p/' + name

	def _seg_at(self, name):
		return self.root + 'Annotations/480p/' + name

	def get_epoch(self):
		return self.epoch

	def get_next_minibatch(self):
		img_blobs = []
		seg_blobs = []
		seq = self.seq_list[self.out_pointer]
		for i in [0]:
			name = seq + '%05d' % self.temp_pointer
			img  = cv2.imread(self._img_at(name + '.jpg'))
			seg  = cv2.imread(self._seg_at(name + '.png'), cv2.CV_LOAD_IMAGE_GRAYSCALE) / 255

			img_blobs.append(img)
			seg_blobs.append(seg)

			self.temp_pointer += 1
			if self.temp_pointer >= self.seq_len[seq]:
				self.temp_pointer = 0
                                self.out_pointer += 1

		return np.array(img_blobs), np.array(np.expand_dims(seg_blobs, axis=3))

"""
The pair dataloader for DAVIS
Used for pair matching net (Analogy-making model, Siamese net, etc.)

TODO: multiprocessing version
"""
class DAVIS_pair_dataloader(DAVIS_seq_dataloader):
	def __init__(self, config, split='train'):
		self.root = '../data/DAVIS/'
		seq_set = self.root + 'SequenceSets/' + split + '.txt'
		with open(seq_set) as f:
			self.seq_list = f.read().rstrip().split('\n')

		self.num_seq = len(self.seq_list)
		self.temp_pointer = 0
		self.epoch = 0

		# Get length of each sequence
		self.seq_len = {}
		for seq in self.seq_list:
			temp = glob.glob(self.root + 'JPEGImages/480p/' + seq + '*.jpg')
			self.seq_len[seq] = len(temp)

		self.batch_num = config['batch_num']
		self._shuffle()
		

	def get_next_minibatch(self):
		img_blobs = []
		seg_blobs = []
		for i in range(self.batch_num):
			img_blob = []
			seg_blob = []
			seq = self.seq_list[self.temp_pointer]
			# Randomly take a pair from each sequence
			idxs = np.random.permutation(self.seq_len[seq])[0:2]
			for idx in idxs:
				name = seq + '%05d' % idx
				img  = cv2.imread(self._img_at(name + '.jpg')) - MEAN_PIXEL
				seg  = cv2.imread(self._seg_at(name + '.png'), cv2.CV_LOAD_IMAGE_GRAYSCALE) / 255

				img_blob.append(img)
				seg_blob.append(seg)

			img_blobs.append(img_blob)
			seg_blobs.append(seg_blob)

			self.temp_pointer += 1
			if self.temp_pointer >= self.num_seq:
				self._shuffle()
				self.epoch += 1

		return np.array(img_blobs), np.array(np.expand_dims(seg_blobs, axis=4))

"""
The test time dataloader for DAVIS
Only the first frame has its ground truth mask

TODO: multiprocessing version
"""
class DAVIS_test_dataloader(object):
	def __init__(self, res='480p', split='val'):
		self.root = '../data/DAVIS/'
		seq_set = self.root + 'SequenceSets/' + split + '.txt'
		with open(seq_set) as f:
			self.seq_list = f.read().rstrip().split('\n')

		self.num_seq = len(self.seq_list)
		self.temp_pointer = 0
		self.inside_pointer = 1
		self.res = res
		self.has_next = True

		# Get length of each sequence
		self.seq_len = {}
		for seq in self.seq_list:
			temp = glob.glob(self.root + 'JPEGImages/' + res + '/' + seq + '*.jpg')
			self.seq_len[seq] = len(temp)

	def _img_at(self, name):
		return self.root + 'JPEGImages/' + self.res + '/' + name

	def _seg_at(self, name):
		return self.root + 'Annotations/' + self.res + '/' + name

	def has_next(self):
		return self.has_next

	"""Each time ouput first frame, first mask, and query frame"""
	def get_next_minibatch(self):
		img_blobs = []
		seg_blobs = []
		for i in [0]:
			img_blob = []
			seg_blob = []
			seq = self.seq_list[self.temp_pointer]
			# Randomly take a pair from each sequence
			idxs = [0, self.inside_pointer]
			for idx in idxs:
				img_name = seq + '%05d' % idx
				seg_name = seq + '%05d' % idx    # only for visualization
				# seg_name = seq + '%05d' % 0    # fixed
				img  = cv2.imread(self._img_at(img_name + '.jpg')) - MEAN_PIXEL
				seg  = cv2.imread(self._seg_at(seg_name + '.png'), cv2.CV_LOAD_IMAGE_GRAYSCALE) / 255

				img_blob.append(img)
				seg_blob.append(seg)

			img_blobs.append(img_blob)
			seg_blobs.append(seg_blob)

			self.inside_pointer += 1
			if self.inside_pointer >= self.seq_len[seq]:
				self.temp_pointer += 1
				self.inside_pointer = 1

			if self.temp_pointer >= self.num_seq:
				print 'Already loaded all the frames!'
				self.has_next = False
				self.temp_pointer = 0
				# ipdb.set_trace()

		return np.array(img_blobs), np.array(np.expand_dims(seg_blobs, axis=4))

if __name__ == '__main__':
	config = {
	'batch_num':5, 
	# 'seq_len': 5,    # Only useful for seq loader
	'iter':100000, 
	'weight_decay': 0.0005,
	'base_lr': 0.001,
	'momentum': 0.9
	}

	# dataloader = DAVIS_dataloader(config)
	# dataloader = DAVIS_seq_dataloader(config)
	# dataloader = DAVIS_pair_dataloader(config)
	# dataloader = DAVIS_test_dataloader()
	dataloader = augment_dataloader(config)
	minibatch = dataloader.get_next_minibatch()

	ipdb.set_trace()
