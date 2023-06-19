import torch
import numpy as np
from itertools import chain
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from datasets import *
from vae import *
from losses import *

set_random_seed(7)


######################################

oids, targets = get_only_r_oids('akb.ztf.snad.space.json')

frames_dataset = AllFramesDataset(oids)
train_loader = DataLoader(frames_dataset, batch_size=256, shuffle=True, num_workers=32)


latent_dim = 78

learning_rate = 5e-5
encoder = VAEEncoder(latent_dim=latent_dim * 2)
decoder = Decoder(latent_dim=latent_dim)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = encoder.to(device)
decoder = decoder.to(device)

optimizer = torch.optim.Adam(
    chain(encoder.parameters(), decoder.parameters()), lr=learning_rate
)

losses = []
for i in tqdm(range(1, 101)):
    losses.append(
        train_vae(
            enc=encoder,
            dec=decoder,
            optimizer=optimizer,
            loader=train_loader,
            epoch=i,
            single_pass_handler=vae_pass_handler,
            loss_handler=vae_loss_handler,
            device=device
        )
    )

torch.save(encoder.state_dict(), 'trained_models/vae/encoder_ld78.zip')
torch.save(decoder.state_dict(), 'trained_models/vae/decoder_ld78.zip')

np.save('trained_models/vae/loss_ld78.npy', losses)