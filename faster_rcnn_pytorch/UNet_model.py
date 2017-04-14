import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1), 
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(                                         
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(128, 256, 3, 1, 1),                                   
            nn.ReLU(inplace=True),                                          
            nn.Conv2d(256, 256, 3, 1, 1),                                   
            nn.ReLU(inplace=True),                                          
            nn.Conv2d(256, 256, 3, 1, 1), 
            nn.ReLU(inplace=True)                                          
        ) 

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(2, ceil_mode=True),                                
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(512, 512, 3, 1, 1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        # Upconv
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, 1, 1), 
            nn.Sigmoid()
        )

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.zero_()

    def forward(self, im_data):
        # Conv
        feat1 = self.conv1(im_data)
        feat2 = self.conv2(feat1)
        feat3 = self.conv3(feat2)
        feat4 = self.conv4(feat3)
        feat5 = self.conv5(feat4)

        # Upconv
        # up4 = self.upconv4(feat5)
        up4 = F.upsample_bilinear(feat5, feat4.size()[2:])
        feat4_ = torch.cat((feat4, up4), 1)
        feat4_ = self.dec4(feat4_)
        # up3 = self.upconv3(feat4_)
        up3 = F.upsample_bilinear(feat4_, feat3.size()[2:])
        feat3_ = torch.cat((feat3, up3), 1)
        feat3_ = self.dec3(feat3_)
        # up2 = self.upconv2(feat3_)
        up2 = F.upsample_bilinear(feat3_, feat2.size()[2:])
        feat2_ = torch.cat((feat2, up2), 1)
        feat2_ = self.dec2(feat2_)
        # up1 = self.upconv1(feat2_)
        up1 = F.upsample_bilinear(feat2_, feat1.size()[2:])
        feat1_ = torch.cat((feat1, up1), 1)
        pred = self.dec1(feat1_)

        return pred

    def load_from_faster_rcnn_h5(self, params):
        pairs = {'rpn.features.conv1.0': 'conv1.0', 'rpn.features.conv1.1': 'conv1.2', 'rpn.features.conv2.0': 'conv2.1', 'rpn.features.conv2.1': 'conv2.3', 'rpn.features.conv3.0': 'conv3.1', 'rpn.features.conv3.1': 'conv3.3', 'rpn.features.conv3.2': 'conv3.5', 'rpn.features.conv4.0': 'conv4.1', 'rpn.features.conv4.1': 'conv4.3', 'rpn.features.conv4.2': 'conv4.5', 'rpn.features.conv5.0': 'conv5.1', 'rpn.features.conv5.1': 'conv5.3', 'rpn.features.conv5.2': 'conv5.5'}
        own_dict = self.state_dict()
        for k, v in pairs.items():
            key = '{}.conv.weight'.format(k)
            vkey = '{}.weight'.format(v)
            param = torch.from_numpy(np.array(params[key]))
            own_dict[vkey].copy_(param) 

            key = '{}.conv.bias'.format(k)
            vkey = '{}.bias'.format(v)
            param = torch.from_numpy(np.array(params[key]))
            own_dict[vkey].copy_(param)

