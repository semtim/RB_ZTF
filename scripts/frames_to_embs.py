import sys
sys.path.insert(0, '../..')
from RB_ZTF.scripts.vae import *
from RB_ZTF.scripts.datasets import *

import torch
import torch.nn as nn
import numpy as np

oids, labels = get_only_r_oids('../akb.ztf.snad.space.json')

encoder = VAEEncoder(latent_dim=36 * 2)
encoder.load_state_dict(torch.load('../trained_models/vae/encoder.zip'))

encoder.eval()
for seq in encoder.encoder.children():
    for child in seq.children():
        if isinstance(child, nn.BatchNorm2d):
            child.track_running_stats = False
                    
for param in encoder.parameters():
    param.requires_grad = False    # freeze all encoder parameters 
    
    
for i, oid in enumerate(oids):
    path = f'../data/{oid}'
    frames = get_frames_seq(path)
    embs = encoder(frames)
    np.save(f'../embeddings/{oid}.npy', embs.numpy())
