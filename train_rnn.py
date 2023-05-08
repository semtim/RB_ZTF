import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import numpy as np
from datasets import *
from rnn import *
from losses import *

set_random_seed(7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Creating dataset..')
oids, labels = get_only_r_oids('akb.ztf.snad.space.json')
#data = FramesSequenceData(oids, labels)
data = EmbsSequenceData(oids, labels)


train, test = random_split(data, [0.8, 0.2])

print('Making dataloaders..')
bucket_boundaries = [200, 400, 600, 800]
train_sampler = BySequenceLengthSampler(train, bucket_boundaries, 32, drop_last=False)
test_sampler = BySequenceLengthSampler(test, bucket_boundaries, 32, drop_last=False)


train_loader = DataLoader(train, batch_size=1, 
                        batch_sampler=train_sampler, 
                        num_workers=16,
                        collate_fn=collate,
                        drop_last=False, pin_memory=False)

test_loader = DataLoader(test, batch_size=1, 
                        batch_sampler=test_sampler, 
                        num_workers=16,
                        collate_fn=collate,
                        drop_last=False, pin_memory=False)



model = RBclassifier(hidden_size=128, rnn_type='GRU') #, mod_emb=True, device=device
model.train()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = rnn_loss_handler
print('Model training..')
res = []
for i in tqdm(range(1, 181)):
    res.append(
        train_rnn(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            epoch=i,
            device=device
        )
    )
    
torch.save(model.state_dict(), 'trained_models/rnn_vae/gru_tversky.zip')
    
np.save('trained_models/rnn_vae_result/gru_tversky.npy', res)
