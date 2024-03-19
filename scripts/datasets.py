import os
from torch.utils.data import Dataset, Sampler
import torch
import numpy as np
import random
from random import shuffle
import json
from torch.nn.utils.rnn import pad_sequence
from astropy.wcs import WCS, utils
from astropy.coordinates import SkyCoord
from astropy.io import fits
from copy import deepcopy

# img preprocessing
def img_prep(hdus, fill_value=0.0, shape=(28, 28)):
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

    return torch.tensor(np.array(frames)).reshape(-1, 1, 28, 28).float()
######################################

# Datasets
class AllFramesDataset(Dataset):
    def __init__(self, oids, path='../data/', transform=None):
        self.oids = oids
        self.imgs_paths = []
        self.transform = transform
        for oid in oids:
            imgs_names = os.listdir(f'{path}{oid}')
            self.imgs_paths += [path + f'{oid}/' + name for name in imgs_names]
        
    def __getitem__(self,idx):
        with fits.open(self.imgs_paths[idx]) as f:
                item = img_prep(f)
                res = torch.tensor(item).reshape(1, 28, 28).float()

        if self.transform:
            res = self.transform(res)
            
        return res
    
    def __len__(self):
        return len(self.imgs_paths)



class EmbsSequenceData(Dataset):
    def __init__(self, oids, labels, path='../embeddings/', label_type='long', return_oid=False):
        self.oids = oids
        if label_type == 'long':
            self.labels = torch.tensor(labels).long()
        elif label_type == 'float':
            self.labels = torch.tensor(labels).float()
        self.obj_path = [path + f'{oid}.npy' for oid in oids]
        self.return_oid = return_oid
        
    def __getitem__(self,idx):
        embs = np.load(self.obj_path[idx])
        label = self.labels[idx]
        if self.return_oid:
            return torch.tensor(embs), label, self.oids[idx]
        else:
            return torch.tensor(embs), label
    
    def __len__(self):
        return len(self.obj_path)
    
    

class FramesSequenceData(Dataset):
    def __init__(self, oids, labels, path='../data/', return_oid=False):
        self.oids = oids
        self.labels = torch.tensor(labels).long()
        self.obj_path = [path + f'{oid}/' for oid in oids]
        self.return_oid = return_oid
        
    def __getitem__(self,idx):
        item = get_frames_seq(self.obj_path[idx])
        label = self.labels[idx]
        if self.return_oid:
            return item, label, self.oids[idx]
        else:
            return item, label
    
    def __len__(self):
        return len(self.obj_path)



class BySequenceLengthSampler(Sampler):

    def __init__(self, data_source,  
                bucket_boundaries, batch_size=64, drop_last=True, shuffle=True, return_oid=False):
        self.data_source = data_source
        ind_n_len = []
        if return_oid:
            for i, (p, _, _) in enumerate(data_source):
                ind_n_len.append( (i, p.shape[0]) )
        else:
            for i, (p, _) in enumerate(data_source):
                ind_n_len.append( (i, p.shape[0]) )

        self.ind_n_len = ind_n_len
        self.bucket_boundaries = bucket_boundaries
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        
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
            pid = self.element_to_bucket_id(p, seq_len)
            if pid in data_buckets.keys():
                data_buckets[pid].append(p)
            else:
                data_buckets[pid] = [p]

        for k in data_buckets.keys():

            data_buckets[k] = torch.tensor(data_buckets[k])

        iter_list = []
        for k in data_buckets.keys():

            if self.shuffle:
                t = self.shuffle_tensor(data_buckets[k])
                batch = torch.split(t, self.batch_size, dim=0)
            else:
                batch = torch.split(data_buckets[k], self.batch_size, dim=0)

            if self.drop_last and len(batch[-1]) != self.batch_size:
                batch = batch[:-1]

            iter_list += batch

        if self.shuffle:
            shuffle(iter_list) # shuffle all the batches so they arent ordered by bucket size
            
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

def collate_with_oid(examples):
    labels = []
    seq_list = []
    oids = []
    for frame_seq, label, oid in examples:
        labels += [label]
        seq_list += [frame_seq]
        oids += [oid]
    return pad_sequence(seq_list, batch_first=True), torch.tensor(labels), oids
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





def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
