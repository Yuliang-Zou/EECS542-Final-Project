import argparse
import os
import h5py
import ipdb

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
from datetime import datetime

from faster_rcnn import network
#from faster_rcnn.faster_rcnn import FasterRCNN, RPN, UNet
from UNet_model import UNet
from faster_rcnn.utils.timer import Timer

import faster_rcnn.roi_data_layer.roidb as rdl_roidb
from faster_rcnn.roi_data_layer.layer import RoIDataLayer
from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file

from Dataloader import augment_dataloader, DAVIS_dataloader

try:
    from termcolor import cprint
except ImportError:
    cprint = None

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)


# hyper-parameters
# ------------
cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
# pretrained_model = 'data/pretrained_model/VGG_imagenet.npy'
h5_model = h5py.File('model/VGGnet_fast_rcnn_iter_70000.h5')
output_dir = 'model/'

start_step = 0
end_step = 100000
lr_decay_steps = {60000, 80000}
lr_decay = 1./10

rand_seed = 1024

# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
cfg_from_file(cfg_file)
#lr = cfg.TRAIN.LEARNING_RATE
lr = 1e-4
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
disp_interval = cfg.TRAIN.DISPLAY
log_interval = cfg.TRAIN.LOG_IMAGE_ITERS
batch_num = 1

# Set up dataloader
# data_loader = augment_dataloader(batch_num=batch_num)
data_loader = DAVIS_dataloader({'batch_num': 5})

# load net
net = UNet().cuda()
network.weights_normal_init(net, dev=0.01)
net.load_from_faster_rcnn_h5(h5_model)

# Set criterion
criterion_bce = nn.BCELoss().cuda()

net.train()

params = list(net.parameters())
# optimizer = torch.optim.Adam(params[-8:], lr=lr)
# optimizer = torch.optim.SGD(params[26:], lr=lr, momentum=momentum, weight_decay=weight_decay)
optimizer = torch.optim.SGD(params[26:], lr=lr, weight_decay=weight_decay)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# training
train_loss = 0
best_loss = 10
tp, tf, fg, bg = 0., 0., 0, 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()
for step in range(start_step, end_step+1):

    # get one batch
    img_np, seg_np = data_loader.get_next_minibatch()
    img_v = Variable(torch.from_numpy(img_np).permute(0, 3, 1, 2).float().cuda(), requires_grad=False)
    seg_v = Variable(torch.from_numpy(seg_np).permute(0, 3, 1, 2).float().cuda(), requires_grad=False) 

    # forward
    pred = net(img_v)
    # loss = criterion_bce(pred, seg_v)
    pred_view = pred.view(-1, 1)
    seg_view = seg_v.view(-1, 1)
    EPS = 1e-6
    temp = 0.65 * seg_view.mul(torch.log(pred_view+EPS)) + 0.35 * seg_view.mul(-1).add(1).mul(torch.log(1-pred_view+EPS))
    loss = -torch.mean(temp)
    train_loss += loss.data[0]
    step_cnt += 1
   
    # backward
    optimizer.zero_grad()
    loss.backward()
    # network.clip_gradient(net, 10.)
    optimizer.step()

    if loss.data[0] < 0.001:
        pred_np = pred.cpu().data.numpy()
        ipdb.set_trace()

    if step % disp_interval == 0:
        duration = t.toc(average=False)
        fps = step_cnt / duration

        log_text = 'step %d, loss: %.4f, fps: %.2f (%.2fs per batch)' % (
            step, train_loss / step_cnt, fps, 1./fps)
        log_print(log_text, color='green', attrs=['bold'])
        if train_loss / step_cnt < best_loss:
             best_loss = train_loss / step_cnt
        train_loss = 0

    if (step % 500 == 0) and step > 0:
        save_name = os.path.join(output_dir, 'vgg_unet_1e-4_{}.h5'.format(step))
        network.save_net(save_name, net)
        print('save model: {}'.format(save_name))

    if step in lr_decay_steps:
        lr *= lr_decay
        optimizer = torch.optim.SGD(params[26:], lr=lr, weight_decay=weight_decay)
        # optimizer = torch.optim.SGD(params[8:], lr=lr, momentum=momentum, weight_decay=weight_decay)
print('The best loss is: {}'.format(best_loss))

