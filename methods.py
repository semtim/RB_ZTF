import torch
import torch.nn as nn
import timm
import numpy as np
from astropy.wcs import WCS, utils
from astropy.coordinates import SkyCoord
from astropy.io import fits
from copy import deepcopy
import os
from torch.utils.data import Dataset, Sampler
import json
from random import shuffle
import math
from torch.nn.utils.rnn import pad_sequence

#############################################
# img preprocessing
def img_prep(hdus, fill_value=0, shape=(28, 28)):
    img = deepcopy(hdus[0].data)
    img = np.nan_to_num(img, nan=fill_value)
    # L_2 normalization
    #norm_img = img / np.sqrt(np.sum(img**2))

    # min max normalization
    norm_img = (img - np.min(img)) / (np.max(img) - np.min(img))

    #max normalization
    #norm_img = img / np.max(img)
    
    
    # std normalization
    #norm_img = (img - np.mean(img)) / np.std(img)
    
    
    # mu-+3sigma -> 0, 1
    #norm_img = (img - np.mean(img) + 3 * np.std(img)) / (6 * np.std(img))
    
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


# return frames sequence by path to obj dir
def get_frames_seq(path):
    frames = []
    frame_names = sorted(os.listdir(path))
    for name in frame_names:
        with fits.open(f'{path}/{name}') as f:
            frame = img_prep(f)
            frames.append(frame)

    return torch.tensor(np.array(frames)).view(-1, 1, 28, 28)
######################################

######################################
# Datasets
class AllFramesDataset(Dataset):
    def __init__(self, oids, path='data/'):
        self.oids = oids
        self.imgs_paths = []
        for oid in oids:
            imgs_names = os.listdir(f'{path}{oid}')
            self.imgs_paths += [path + f'{oid}/' + name for name in imgs_names]
        
    def __getitem__(self,idx):
        with fits.open(self.imgs_paths[idx]) as f:
                item = img_prep(f)
        return torch.tensor(item).view(1, 28, 28)
    
    def __len__(self):
        return len(self.imgs_paths)




class FramesSequenceData(Dataset):
    def __init__(self, oids, labels, path='data/'):
        self.oids = oids
        self.labels = torch.tensor(labels).long()
        self.obj_path = [path + f'{oid}/' for oid in oids]
        
    def __getitem__(self,idx):
        item = get_frames_seq(self.obj_path[idx])
        label = self.labels[idx]
        return item, label
    
    def __len__(self):
        return len(self.obj_path)



class BySequenceLengthSampler(Sampler):

    def __init__(self, data_source,  
                bucket_boundaries, batch_size=64, drop_last=True):
        self.data_source = data_source
        ind_n_len = []
        for i, (p, _) in enumerate(data_source):
            ind_n_len.append( (i, p.shape[0]) )

        self.ind_n_len = ind_n_len
        self.bucket_boundaries = bucket_boundaries
        self.batch_size = batch_size
        self.drop_last = drop_last

        if self.drop_last:
            print("WARNING: drop_last=True, dropping last non batch-size batch in every bucket ... ")

        self.boundaries = list(self.bucket_boundaries)
        self.buckets_min = torch.tensor([np.iinfo(np.int32).min] + self.boundaries)
        self.buckets_max = torch.tensor(self.boundaries + [np.iinfo(np.int32).max])
        self.boundaries = torch.tensor(self.boundaries)

    def shuffle_tensor(self, t):
        return t[torch.randperm(len(t))]
        
    def __iter__(self):
        data_buckets = dict()
        # where p is the id number and seq_len is the length of this id number. 
        for p, seq_len in self.ind_n_len:
            pid = self.element_to_bucket_id(p,seq_len)
            if pid in data_buckets.keys():
                data_buckets[pid].append(p)
            else:
                data_buckets[pid] = [p]

        for k in data_buckets.keys():

            data_buckets[k] = torch.tensor(data_buckets[k])

        iter_list = []
        for k in data_buckets.keys():

            t = self.shuffle_tensor(data_buckets[k])
            batch = torch.split(t, self.batch_size, dim=0)

            if self.drop_last and len(batch[-1]) != self.batch_size:
                batch = batch[:-1]

            iter_list += batch

        shuffle(iter_list) # shuffle all the batches so they arent ordered by bucket
        # size
        for i in iter_list: 
            yield i.numpy().tolist() # as it was stored in an array
    
    def __len__(self):
        return len(self.data_source)
    
    def element_to_bucket_id(self, x, seq_length):

        valid_buckets = (seq_length >= self.buckets_min)*(seq_length < self.buckets_max)
        bucket_id = valid_buckets.nonzero()[0].item()

        return bucket_id


def collate(examples):
    labels = []
    seq_list = []
    for frame_seq, label in examples:
        labels += [label]
        seq_list += [frame_seq]

    return pad_sequence(seq_list, batch_first=True), torch.tensor(labels)

######################################

######################################
def check_if_r(oid):
    bands = {'1':'g', '2':'r', '3':'i'}
    str_oid = str(oid)
    if len(str_oid) != 15:
        return True if bands[str_oid[4]]=='r' else False

    else:
        return True if bands[str_oid[3]]=='r' else False

#get oids and tags (in r filter) from json
def get_only_r_oids(filepath):
    file = open(filepath)
    obj_list = json.load(file)
    file.close()

    oids = []
    tags = []
    for data in obj_list:
        if check_if_r(data['oid']):
            oids.append(data['oid'])
            tags.append(data['tags'])

    targets = [] # 1-artefact,  0-transient
    for tag_list in tags:
        if 'artefact' in tag_list:
            targets.append(1)
        else:
            targets.append(0)
    
    return oids, targets
######################################    
    

######################################
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




class SimpleCNN3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
                                    nn.Conv3d(1, 16, 3, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2),
                                    nn.Conv3d(16, 32, 3, padding=1),
                                    nn.MaxPool2d(2),
                                    nn.ReLU(),
                                    nn.Conv3d(32, 64, 3, padding=1),
                                    nn.ReLU(),
                                    nn.AdaptiveAvgPool3d((7, 7, 1)),
                                    nn.Linear(64*7*7, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 2)
                                   )
        
        

    def forward(self, x):
        #x.shape = batch, ch, img_size
        x = self.layers(x)
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