import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import numpy as np
from datasets import *
from rnn import *
from losses import *
import os
from ranger21 import Ranger21

set_random_seed(7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Creating dataset..')
oids, labels = get_only_r_oids('akb.ztf.snad.space.json')
#data = FramesSequenceData(oids, labels)
data = EmbsSequenceData(oids, labels, label_type='float')


train, test = random_split(data, [0.8, 0.2])

print('Making dataloaders..')
bucket_boundaries = [200, 400, 600, 800]
train_sampler = BySequenceLengthSampler(train, bucket_boundaries, 32, drop_last=False, shuffle=True)
test_sampler = BySequenceLengthSampler(test, bucket_boundaries, 32, drop_last=False, shuffle=False)


train_loader = DataLoader(train, batch_size=1, 
                        batch_sampler=train_sampler, 
                        num_workers=8,
                        collate_fn=collate,
                        drop_last=False, pin_memory=False)

test_loader = DataLoader(test, batch_size=1, 
                        batch_sampler=test_sampler, 
                        num_workers=8,
                        collate_fn=collate,
                        drop_last=False, pin_memory=False)



model = RBclassifier(hidden_size=128, latent_dim=78, rnn_type='GRU', out_size=2) #, mod_emb=True, device=device
model.train()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) #Ranger21(model.parameters(), lr=1e-3, num_epochs=500, num_batches_per_epoch=58,weight_decay=8e-4)
criterion = nn.CrossEntropyLoss() #rnn_loss_handler
print('Model training..')
res = []
for i in tqdm(range(1, 501)):
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

model_dir = 'trained_models/rnn/baseline'
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)

torch.save(model.state_dict(), f'{model_dir}/model.zip')

np.save(f'{model_dir}/result.npy', res)
