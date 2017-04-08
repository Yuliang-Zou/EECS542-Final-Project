# Preprocesing dataset

import argparse
import cv2
import glob
try:
	import ipdb
except:
	import pdb as ipdb
import matplotlib.pyplot as plt
import numpy as np
from os import makedirs
from os.path import exists, join
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='youtube_masks')
parser.add_argument('--size', type=int, default=500)
parser.add_argument('--flip', type=bool, default=True)
# Used for DAVIS only
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--res', type=str, default='480p')

opt = parser.parse_args()


"""Define some helper functions"""
def bbox(img):
	rows = np.any(img, axis=1)
	cols = np.any(img, axis=0)
	rmin, rmax = np.where(rows)[0][[0, -1]]
	cmin, cmax = np.where(cols)[0][[0, -1]]

	return rmin, rmax, cmin, cmax

def large_bbox(rc, bbox):
	row, col = rc
	rmin, rmax, cmin, cmax = bbox

	height = rmax - rmin + 1
	width  = cmax - cmin + 1

	u_delta = int(np.random.random() * height)
	d_delta = int(np.random.random() * height)
	l_delta = int(np.random.random() * width)
	r_delta = int(np.random.random() * width)

	rmin = rmin - u_delta if rmin - u_delta > 0 else 0
	rmax = rmax + d_delta if rmax + d_delta < row else row
	cmin = cmin - l_delta if cmin - l_delta > 0 else 0
	cmax = cmax + r_delta if cmax + r_delta < col else col

	return rmin, rmax, cmin, cmax

def flip(im, ms):
	flipped_im = np.ndarray((im.shape), dtype='uint8')
	flipped_im[:,:,0] = np.fliplr(im[:,:,0])
	flipped_im[:,:,1] = np.fliplr(im[:,:,1])
	flipped_im[:,:,2] = np.fliplr(im[:,:,2])

	flipped_ms = np.ndarray((ms.shape), dtype='uint8')
	flipped_ms[:,:] = np.fliplr(ms[:,:])

	return flipped_im, flipped_ms


# Only one mask in each frame
if opt.dataset == 'DAVIS':
	root = './DAVIS/'
	name = root + 'ImageSets/' + opt.res +'/' + opt.split + '.txt'
	with open(name) as f:
		im_set = f.read().rstrip().split('\n')

	dump_ms = './augment/DAVIS/Annotations/'
	dump_im = './augment/DAVIS/JPEGImages/'
	if not exists(dump_ms):
		makedirs(dump_ms)
	if not exists(dump_im):
		makedirs(dump_im)
	f = open('./augment/DAVIS.txt', 'w')

	# Iterate
	idx = 0
	for item in im_set:
		pair = item.split(' ')
		im_name = pair[0]
		ms_name = pair[1]

		ms = cv2.imread(root + ms_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
		im = cv2.imread(root + im_name)
		# Get untight bbox, sometimes no bbox
		try:
			rmin, rmax, cmin, cmax = large_bbox(ms.shape, bbox(np.array(ms)))
		except:
			continue

		ms_ = cv2.resize(ms[rmin:rmax, cmin:cmax], (opt.size, opt.size))
		im_ = cv2.resize(im[rmin:rmax, cmin:cmax], (opt.size, opt.size), interpolation=cv2.INTER_NEAREST)

		im_name = dump_im + '{0:05d}'.format(idx) + '.jpg'
		ms_name = dump_ms + '{0:05d}'.format(idx) + '.png'
		cv2.imwrite(im_name, im_)
		cv2.imwrite(ms_name, ms_)
		f.write(im_name[1:] + ' ' + ms_name[1:] + '\n')
		idx += 1

		if opt.flip:
			im_, ms_ = flip(im_, ms_)
			im_name = dump_im + '{0:05d}'.format(idx) + '.jpg'
			ms_name = dump_ms + '{0:05d}'.format(idx) + '.png'
			cv2.imwrite(im_name, im_)
			cv2.imwrite(ms_name, ms_)
			f.write(im_name[1:] + ' ' + ms_name[1:] + '\n')
			idx += 1

		# plt.imshow(ms_)
		# plt.show()
	f.close()

# Each frame might have multiple masks
elif opt.dataset == 'SegTrackv2':
	root = './SegTrackv2/'
	all_name = root + 'ImageSets/all.txt'
	im_set = {}
	with open(all_name) as f:
		vid_set = f.read().rstrip().split('\n')
	for vid in vid_set:
		with open(root + 'ImageSets/' + vid[1:].rstrip() + '.txt') as f:
			frame_set = f.read().rstrip().split('\n')
			head = frame_set.pop(0)
		im_set[head] = []
		for frame in frame_set:
			frame_name = head + frame
			im_set[head].append(frame_name)

	dump_ms = './augment/SegTrackv2/Annotations/'
	dump_im = './augment/SegTrackv2/JPEGImages/'
	if not exists(dump_ms):
		makedirs(dump_ms)
	if not exists(dump_im):
		makedirs(dump_im)
	f = open('./augment/SegTrackv2.txt', 'w')

	# Iterate
	idx = 0
	for vid in im_set:
		num_mask = 1
		if exists(root + 'GroundTruth/' + vid + str(num_mask)):
			num_mask += 1
		if num_mask > 1:
			MULTIPLE = True
		else:
			MULTIPLE = False

		for im_name in im_set[vid]:
			# Deal with different extentions
			ext = '.jpg'
			if not exists(root + 'JPEGImages/' + im_name + ext):
				ext = '.png'
			if not exists(root + 'JPEGImages/' + im_name + ext):
				ext = '.bmp'
			im = cv2.imread(root + 'JPEGImages/' + im_name + ext)
			# Single mask
			if not MULTIPLE:
				ms = cv2.imread(root + 'GroundTruth/' + im_name + '.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
				# Get untight bbox, sometimes no bbox
				try:
					rmin, rmax, cmin, cmax = large_bbox(ms.shape, bbox(np.array(ms)))
				except:
					continue
				# Crop and resize
				ms_ = cv2.resize(ms[rmin:rmax, cmin:cmax], (opt.size, opt.size))
				im_ = cv2.resize(im[rmin:rmax, cmin:cmax], (opt.size, opt.size), interpolation=cv2.INTER_NEAREST)

				im_name = dump_im + '{0:05d}'.format(idx) + '.jpg'
				ms_name = dump_ms + '{0:05d}'.format(idx) + '.png'
				cv2.imwrite(im_name, im_)
				cv2.imwrite(ms_name, ms_)
				f.write(im_name[1:] + ' ' + ms_name[1:] + '\n')
				idx += 1

				if opt.flip:
					im_, ms_ = flip(im_, ms_)
					im_name = dump_im + '{0:05d}'.format(idx) + '.jpg'
					ms_name = dump_ms + '{0:05d}'.format(idx) + '.png'
					cv2.imwrite(im_name, im_)
					cv2.imwrite(ms_name, ms_)
					f.write(im_name[1:] + ' ' + ms_name[1:] + '\n')
					idx += 1

			# Multiple masks
			if MULTIPLE:
				for j in range(1, num_mask+1):
					ms = cv2.imread(root + 'GroundTruth/' + vid + str(j) + '/' + im_name.split('/')[1] + '.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
					# Get untight bbox, sometimes no bbox
					try:
						rmin, rmax, cmin, cmax = large_bbox(ms.shape, bbox(np.array(ms)))
					except:
						continue
					# Crop and resize
					ms_ = cv2.resize(ms[rmin:rmax, cmin:cmax], (opt.size, opt.size))
					im_ = cv2.resize(im[rmin:rmax, cmin:cmax], (opt.size, opt.size), interpolation=cv2.INTER_NEAREST)

					im_name = dump_im + '{0:05d}'.format(idx) + '.jpg'
					ms_name = dump_ms + '{0:05d}'.format(idx) + '.png'
					cv2.imwrite(im_name, im_)
					cv2.imwrite(ms_name, ms_)
					f.write(im_name[1:] + ' ' + ms_name[1:] + '\n')
					idx += 1

					if opt.flip:
						im_, ms_ = flip(im_, ms_)
						im_name = dump_im + '{0:05d}'.format(idx) + '.jpg'
						ms_name = dump_ms + '{0:05d}'.format(idx) + '.png'
						cv2.imwrite(im_name, im_)
						cv2.imwrite(ms_name, ms_)
						f.write(im_name[1:] + ' ' + ms_name[1:] + '\n')
						idx += 1

# Each frame has single mask, but the file structure is terrible
elif opt.dataset == 'youtube_masks':
	root = './youtube_masks/'
	name = root + 'vid_list.txt'
	with open(name) as f:
		vid_set = f.read().rstrip().split('\n')

	dump_ms = './augment/youtube_masks/Annotations/'
	dump_im = './augment/youtube_masks/JPEGImages/'
	if not exists(dump_ms):
		makedirs(dump_ms)
	if not exists(dump_im):
		makedirs(dump_im)
	f = open('./augment/youtube_masks.txt', 'w')

	# Iterate
	idx = 0
	for vid in vid_set:
		path = root + vid + 'data/'
		folder_list = glob.glob(path + '*')
		for folder in folder_list:
			folder += '/shots/001/'
			im_folder = folder + 'images/'
			ms_folder = folder + 'labels/'
			im_list = glob.glob(im_folder + '*')
			for im_name in im_list:
				im = cv2.imread(im_name)
				ms_name = ms_folder + im_name.split('/')[-1][:-3] + 'jpg'
				ms = cv2.imread(ms_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)

				# Get untight bbox, sometimes no bbox
				try:
					rmin, rmax, cmin, cmax = large_bbox(ms.shape, bbox(np.array(ms)))
				except:
					continue

				ms_ = cv2.resize(ms[rmin:rmax, cmin:cmax], (opt.size, opt.size))
				im_ = cv2.resize(im[rmin:rmax, cmin:cmax], (opt.size, opt.size), interpolation=cv2.INTER_NEAREST)

				im_name = dump_im + '{0:05d}'.format(idx) + '.jpg'
				ms_name = dump_ms + '{0:05d}'.format(idx) + '.png'
				cv2.imwrite(im_name, im_)
				cv2.imwrite(ms_name, ms_)
				f.write(im_name[1:] + ' ' + ms_name[1:] + '\n')
				idx += 1

				if opt.flip:
					im_, ms_ = flip(im_, ms_)
					im_name = dump_im + '{0:05d}'.format(idx) + '.jpg'
					ms_name = dump_ms + '{0:05d}'.format(idx) + '.png'
					cv2.imwrite(im_name, im_)
					cv2.imwrite(ms_name, ms_)
					f.write(im_name[1:] + ' ' + ms_name[1:] + '\n')
					idx += 1

else:
	raise KeyError('No such dataset!')




