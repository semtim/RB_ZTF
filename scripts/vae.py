import torch
import torch.nn as nn


# VAE
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim # latent space size
        hidden_dims = [32, 64, 128, 256, 512] # num of filters in layers
        modules = []
        in_channels = 1 # initial value of channels
        for h_dim in hidden_dims[:-1]: # conv layers
            modules.append(
                nn.Sequential(
                    nn.Conv2d(                    
                        in_channels=in_channels, # num of input channels 
                        out_channels=h_dim, # num of output channels 
                        kernel_size=3, 
                        stride=2, # convolution kernel step
                        padding=1, # save shape 
                        bias=False,
                    ),
                    nn.BatchNorm2d(h_dim),  
                    nn.LeakyReLU(), 
                )
            )
            in_channels = h_dim # changing number of input channels for next iteration

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, bias=False), # changing the kernel size, because  size of the array (2*2)
                nn.BatchNorm2d(512),
                nn.LeakyReLU(),
            )
        )
        modules.append(nn.Flatten()) # to vector, size 512 * 2*2 = 2048
        modules.append(nn.Linear(512 * 2 * 2, latent_dim)) 

        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        x = self.encoder(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        hidden_dims = [512, 256, 128, 64, 32] # num of filters in layers
        self.linear = nn.Linear(in_features=latent_dim, out_features=512) 

        modules = []
        for i in range(len(hidden_dims) - 1): # define ConvTransopse layers
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=hidden_dims[i],
                        out_channels=hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=hidden_dims[-1],
                    out_channels=hidden_dims[-1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=hidden_dims[-1], out_channels=1, kernel_size=7, padding=1),
                nn.ReLU(),
            )
        )

        self.decoder = nn.Sequential(*modules)

    def forward(self, x):
        x = self.linear(x) # from latents space to Linear 
        x = x.view(-1, 512, 1, 1) # reshape
        x = self.decoder(x) # reconstruction
        return x




def train_vae(
    enc,
    dec,
    loader,
    optimizer,
    single_pass_handler,
    loss_handler,
    epoch,
    device,
            ):
    '''
    Function to train model, parameters: 
      enc - encoder
      dec - decoder
      loader - loader of data
      optimizer - optimizer
      single_pass_handler - return reconstructed image, use for loss 
      loss_handler - loss function 
      epoch - num of epochs
      '''
    ep_loss = 0
    for batch_idx, data in enumerate(loader): 
        batch_size = data.size(0)
        optimizer.zero_grad()
        data = data.to(device)

        latent, output = single_pass_handler(enc, dec, data, device) # reconstructed image drom decoder 

        loss = loss_handler(data, output, latent) # compute loss
        loss.backward()
        optimizer.step()
        ep_loss += loss.item()
    return ep_loss / len(loader)





class VAEEncoder(Encoder):
    def __init__(self, latent_dim):
        if latent_dim % 2 != 0: # check for the parity of the latent space
            raise Exception("Latent size for VAEEncoder must be even")
        super().__init__(latent_dim)


def vae_split(latent):
    size = latent.shape[1] // 2 # divide the latent representation into mu and log_var
    mu = latent[:, :size] 
    log_var = latent[:, size:]  
    return mu, log_var


def vae_reparametrize(mu, log_var, device='cpu'): 
    sigma = torch.exp(0.5 * log_var) 
    z = torch.randn(mu.shape[0], mu.shape[1]).to(device) 
    return z * sigma + mu 


def vae_pass_handler(encoder, decoder, data, device, *args, **kwargs): 
    latent = encoder(data) 
    mu, log_var = vae_split(latent) 
    sample = vae_reparametrize(mu, log_var, device) 
    recon = decoder(sample)
    return latent, recon
