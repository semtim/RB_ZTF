import torch
import torch.nn as nn
import timm
import numpy as np
from astropy.wcs import WCS, utils
from astropy.coordinates import SkyCoord
from copy import deepcopy
import os
from astropy.io import fits


def img_prep(hdus, fill_value=0, shape=(28, 28)):
    img = deepcopy(hdus[0].data)
    img = np.nan_to_num(img, nan=fill_value)
    # L_2 normalization
    #norm_img = img / np.sqrt(np.sum(img**2))

    # min max normalization
    norm_img = (img - np.min(img)) / (np.max(img) - np.min(img))

    # fill img to shape
    cur_shape = norm_img.shape
    if cur_shape == shape:
        return norm_img

    coords = (hdus[0].header['OIDRA'], hdus[0].header['OIDDEC'])
    coord = SkyCoord(*coords, unit='deg', frame='icrs')
    currentWCS = WCS(hdus[0].header, hdus)
    pix_coord = utils.skycoord_to_pixel(coord, currentWCS)
    pix_coord = (int(pix_coord[0]), int(pix_coord[1]))

    if pix_coord[1] >= int((shape[1] - 1)/2):
        y_shift = 0
    else:
        y_shift = shape[1] - cur_shape[0]

    if pix_coord[0]  >= int((shape[0] - 1)/2):
        x_shift = 0
    else:
        x_shift = shape[0] - cur_shape[1]

    filled_img = np.full(shape, fill_value)
    filled_img[y_shift:cur_shape[0]+y_shift, x_shift:cur_shape[1]+x_shift] = norm_img

    return filled_img



def get_img_stack(oid):
    imgs = []
    for root, dirs, files in os.walk('data/' + str(oid)):
        for filename in sorted(files):
            with fits.open('data/' + str(oid) + '/' + filename) as f:
                a = img_prep(f)
                imgs.append(a)

    return torch.tensor(np.array(imgs))



class RBclassifier(nn.Module):
    def __init__(self, hidden_size, conv_model='simplecnn'):
        super().__init__()
        self.hidden_size = hidden_size
        self.conv_name = conv_model


        if conv_model == 'simplecnn':
            self.conv_model = SimpleCNN()
            with torch.inference_mode():
                x = torch.rand(1, 1, 28, 28)
                out = self.conv_model(x)
                conv_out_shape = out.shape
                conv_out_size = conv_out_shape[1] * conv_out_shape[2] * conv_out_shape[3]


        else:
            self.conv_model = timm.create_model('tf_efficientnet_b0', features_only=True, in_chans=1)
        
            self.conv_model.conv_stem = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), bias=False)
            self.conv_model.blocks[1][0].conv_dw = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), groups=96, bias=False)
            #self.conv_model.blocks[2][0].conv_dw
            #self.conv_model.blocks[3][0].conv_dw
            self.conv_model.blocks[5][0].conv_dw = nn.Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), groups=672, bias=False)

            with torch.inference_mode():
                x = torch.rand(1, 1, 28, 28)
                out = self.conv_model(x)
                conv_out_shape = out[-1].shape
                conv_out_size = conv_out_shape[1] * conv_out_shape[2] * conv_out_shape[3]



        
        self.rnn_layer = nn.LSTM(input_size=conv_out_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)


    def forward(self, x):
        #x.shape = batch, seq, ch, img_size
        shape = x.shape
        conv_out = self.conv_model(x.view(-1, 1, 28, 28))
        
        #x in rnn layer shape = batch, seq, -1
        if self.conv_name == 'simplecnn':
            out, (h, c) = self.rnn_layer(conv_out.view(shape[0], shape[1], -1))
        else:
            out, (h, c) = self.rnn_layer(conv_out[-1].view(shape[0], shape[1], -1))
        
        #x in classifier_layer = batch, hidden_size
        y = self.fc(h.view(shape[0], self.hidden_size))
        return y






class SimpleCNN(nn.Module):
    def __init__(self, fc=False):
        super().__init__()
        self.layers = nn.Sequential(
                                    nn.Conv2d(1, 16, 3, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2),
                                    nn.Conv2d(16, 32, 3, padding=1),
                                    nn.MaxPool2d(2),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 64, 3, padding=1, bias=False),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(64),
                                    )
        if fc:
            self.fc = nn.Sequential(nn.Linear(64*7**2, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 2)
                                   )
        else:
            self.fc = False


    def forward(self, x):
        #x.shape = batch, ch, img_size
        x = self.layers(x)
        if self.fc:
            x = self.fc(nn.Flatten()(x))

        return x



class CustomResnet3d(nn.Module):
    def __init__(self):
        super(CustomResnet3d, self).__init__()
        self.layer1 = nn.Sequential( nn.Conv3d(1, 16, kernel_size=(3,3,3), stride=(1,1,2), padding=1, bias=False),
                                     nn.BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                     nn.ReLU(inplace=True)
                                    )
        self.basic16 = nn.Sequential( *[BasicBlock(in_channels=16, out_channels=16) for i in range(7)] )
        self.basic_reshape1 = BasicBlock(in_channels=16, out_channels=32, downsample=True)
        self.basic32 = nn.Sequential( *[BasicBlock(in_channels=32, out_channels=16) for i in range(7)] )
        self.basic_reshape2 = BasicBlock(in_channels=32, out_channels=64, downsample=True)
        self.basic64 = nn.Sequential( *[BasicBlock(in_channels=64, out_channels=64) for i in range(8)] )
        self.gap = nn.AdaptiveAvgPool3d((7, 7, 1))
        self.linear = nn.Linear(7*7*64, 2)
        self.init()

    def forward(self, x):
        x = self.layer1(x)

        x = self.basic16(x)
        x = self.basic_reshape1(x)

        x = self.basic32(x)
        x = self.basic_reshape2(x)

        x = self.basic64(x)

        x = self.gap(x)
        x = nn.Flatten()(x)
        scores = self.linear(x)
        return scores

    def init(self):
        gain = torch.nn.init.calculate_gain("relu")
        torch.nn.init.xavier_normal_(self.layer1[0].weight, gain=gain)
        torch.nn.init.xavier_normal_(self.linear.weight, gain=gain)
        torch.nn.init.zeros_(self.linear.bias)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):  # You can add params here
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.downsample_layer = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=(2, 2, 4),  bias=False))
        if not downsample:
          self.layer = nn.Sequential(nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), padding=1, bias=False),
                                   nn.BatchNorm3d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), padding=1, bias=False),
                                   nn.BatchNorm3d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                  )
        else:
          self.layer = nn.Sequential(
                                   nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), padding=1, bias=False),
                                   nn.BatchNorm3d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=1, stride=(2, 2, 4),  bias=False),
                                   nn.BatchNorm3d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                  )
        self.init()

    def init(self):
        gain = torch.nn.init.calculate_gain("relu")
        for child in self.layer.children():
            if isinstance(child, nn.Conv2d):
                torch.nn.init.xavier_normal_(child.weight, gain=gain)
                if child.bias is not None:
                    torch.nn.init.zeros_(child.bias)
        torch.nn.init.xavier_normal_(self.downsample_layer[0].weight, gain=gain)
 

    def forward(self, x):
        identity = x
        if self.downsample:
          identity = self.downsample_layer(identity)
        return self.layer(x) + identity  