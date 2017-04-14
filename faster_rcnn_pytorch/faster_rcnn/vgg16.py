import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import ipdb

from utils.blob import im_list_to_blob
from network import Conv2d
import network


class VGG16(nn.Module):
    def __init__(self, bn=False, mid_feat=False):
        super(VGG16, self).__init__()

        self.conv1 = nn.Sequential(Conv2d(3, 64, 3, same_padding=True, bn=bn),
                                   Conv2d(64, 64, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2, ceil_mode=True))
        self.conv2 = nn.Sequential(Conv2d(64, 128, 3, same_padding=True, bn=bn),
                                   Conv2d(128, 128, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2, ceil_mode=True))
        network.set_trainable(self.conv1, requires_grad=False)
        network.set_trainable(self.conv2, requires_grad=False)

        self.conv3 = nn.Sequential(Conv2d(128, 256, 3, same_padding=True, bn=bn),
                                   Conv2d(256, 256, 3, same_padding=True, bn=bn),
                                   Conv2d(256, 256, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2, ceil_mode=True))
        self.conv4 = nn.Sequential(Conv2d(256, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2, ceil_mode=True))
        self.conv5 = nn.Sequential(Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn))

        network.set_trainable(self.conv3, requires_grad=False)
        network.set_trainable(self.conv4, requires_grad=False)
        network.set_trainable(self.conv5, requires_grad=False)
        self.mid_feat = mid_feat

    def forward(self, im_data):
        feat1 = self.conv1(im_data)
        feat2 = self.conv2(feat1)
        feat3 = self.conv3(feat2)
        feat4 = self.conv4(feat3)
        feat5 = self.conv5(feat4)

        if self.mid_feat:
            return feat1, feat2, feat3, feat4, feat5
        else:
            return feat5

    def load_from_npz(self, params):
        # params = np.load(npz_file)
        own_dict = self.state_dict()
        for name, val in own_dict.items():
            i, j = int(name[4]), int(name[6]) + 1
            ptype = 'weights' if name[-1] == 't' else 'biases'
            key = 'conv{}_{}/{}:0'.format(i, j, ptype)
            param = torch.from_numpy(params[key])
            if ptype == 'weights':
                param = param.permute(3, 2, 0, 1)
            val.copy_(param)

    # def load_from_npy_file(self, fname):
    #     own_dict = self.state_dict()
    #     params = np.load(fname).item()
    #     for name, val in own_dict.items():
    #         # # print name
    #         # # print val.size()
    #         # # print param.size()
    #         # if name.find('bn.') >= 0:
    #         #     continue
    #
    #         i, j = int(name[4]), int(name[6]) + 1
    #         ptype = 'weights' if name[-1] == 't' else 'biases'
    #         key = 'conv{}_{}'.format(i, j)
    #         param = torch.from_numpy(params[key][ptype])
    #
    #         if ptype == 'weights':
    #             param = param.permute(3, 2, 0, 1)
    #
    #         val.copy_(param)


if __name__ == '__main__':
    vgg = VGG16()
    vgg.load_from_npy_file('/media/longc/Data/models/VGG_imagenet.npy')
