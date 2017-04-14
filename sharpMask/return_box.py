# Put the mask back to the original iamge position
import cv2
import ipdb
import numpy as np
from os import makedirs
from os.path import exists

videos = ['blackswan/', 'bmx-trees/', 'breakdance/', 'camel/', 
'car-roundabout/', 'car-shadow/', 'cows/', 'dance-twirl/', 
'dog/', 'drift-chicane/', 'drift-straight/', 'goat/', 
'horsejump-high/', 'kite-surf/', 'libby/', 'motocross-jump/',
'paragliding-launch/', 'parkour/', 'scooter-black/', 'soapbox/']

path = './output/'
dump = '../final_output/'

data = np.load(path + 'crop.npy')
num_frame = data.shape[0]
for i in range(num_frame):
	idx = data[i, 0]
	frm = data[i, 1]
	mask = np.zeros((480, 854), dtype=np.int)
	crop_name = path + 'rescrop_' + str(idx) + '_' + str(frm) + '.png'
	if exists(crop_name):
		im = cv2.imread(crop_name, 0)
		im[im>0] = 255

		x1 = data[i, 2]
		y1 = data[i, 3]
		x2 = data[i, 4]
		y2 = data[i, 5]
		mask[y1:y2+1, x1:x2+1] = im

		temp_dump = dump + videos[idx]
		if not exists(temp_dump):
			makedirs(temp_dump)
		save_name = temp_dump + '{0:05d}'.format(frm) + '.png'
		cv2.imwrite(save_name, mask)

ipdb.set_trace()