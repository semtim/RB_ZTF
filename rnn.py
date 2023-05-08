import torch
import torch.nn as nn
from vae import *


#### only RNN
class RBclassifier(nn.Module):
    def __init__(self, hidden_size, rnn_type='LSTM', bidirectional=False, device='cpu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.device = device
        self.latent_dim = 36
        self.rnn_in_size = self.latent_dim
        self.bidir = bidirectional
        
        if rnn_type == 'LSTM':
            self.rnn_layer = nn.LSTM(input_size=self.rnn_in_size, hidden_size=hidden_size, batch_first=True,
                                    bidirectional=self.bidir)
        elif rnn_type == 'GRU':
            self.rnn_layer = nn.GRU(input_size=self.rnn_in_size, hidden_size=hidden_size, batch_first=True,
                                    bidirectional=self.bidir)
        
        if self.bidir:
            D = 2
        else:
            D = 1
        
        self.clf = nn.Sequential(
                                 nn.Linear(D*hidden_size, 2),
                                 #nn.Linear(64, 2)
                                 nn.Sigmoid(),
                                )

    def forward(self, x):
        #x shape = batch, seq, latent_dim*2
        embds = x[:, :, :self.latent_dim]
        if self.rnn_type == 'LSTM':
            out, (h, c) = self.rnn_layer(embds)
        elif self.rnn_type == 'GRU':
            out, h = self.rnn_layer(embds)
        
        #x in classifier_layer = batch, hidden_size
        if self.bidir:
            h_concat = torch.hstack((h[0], h[1]))
            y = self.clf(h_concat)
        else:
            y = self.clf(h[0])
        return y


#######################################################

# Train and validate
def train_rnn(
    model,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    epoch,
    device,
):

    result = {'train_loss':[], 'train_acc':[], 'test_loss':[], 'test_acc':[],
             'test_prec':[], 'test_rec':[]}
    ep_loss, ep_acc = 0, 0
    batch_count = 0
    for batch_idx, (data, lab) in enumerate(train_loader): 
        batch_size = data.size(0)
        optimizer.zero_grad()
        data = data.to(device)
        lab = lab.to(device)

        output = model(data) 

        loss = criterion(output, lab)
        loss.backward()
        optimizer.step()
        ep_loss += loss.item()
        
        _, pred_lab = torch.max(output, 1)
        ep_acc += metrics.accuracy_score(lab.cpu(), pred_lab.cpu())
        batch_count += 1

    model.cpu()
    model.device = 'cpu'
    test_res = validate(model, test_loader, criterion)
    model.device = device
    model.cuda()
    
    result['train_loss'].append( ep_loss / batch_count )
    result['train_acc'].append( ep_acc / batch_count )
    result['test_loss'].append(test_res[-1])
    result['test_acc'].append(test_res[0])
    result['test_prec'].append(test_res[2])
    result['test_rec'].append(test_res[1])
    
    return result



@torch.inference_mode()
def validate(
             model,
             loader,
             criterion,
             ):
 
    batch_count = 0
    acc, prec, rec, loss = 0, 0, 0, 0
    for x, label in loader:
        output = model(x)
        pred_value, pred_label = torch.max(output, 1)
        acc += metrics.accuracy_score(label, pred_label)
        prec += metrics.precision_score(label, pred_label)
        rec += metrics.recall_score(label, pred_label)
        loss += criterion(output, label)
        batch_count += 1
    acc /= batch_count
    rec /= batch_count
    prec /= batch_count
    loss = loss.item() / batch_count

    return acc, rec, prec, loss

#######################################################



# RNN + trained VAE
class RBclassifierVAE(nn.Module):
    def __init__(self, hidden_size, rnn_type='LSTM', mod_emb=False, device='cpu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.device = device
        self.mod_emb = mod_emb
        self.latent_dim = 36
        self.rnn_in_size = self.latent_dim
        self.encoder = VAEEncoder(latent_dim=self.latent_dim * 2)
        self.encoder.load_state_dict(torch.load('trained_models/vae/encoder.zip'))
        
        self.encoder.eval()
        for seq in self.encoder.encoder.children():
            for child in seq.children():
                if isinstance(child, nn.BatchNorm2d):
                    child.track_running_stats = False
                    
        for param in self.encoder.parameters():
            param.requires_grad = False    # freeze all encoder parameters 
        
        if mod_emb:
            self.decoder = Decoder(latent_dim=self.latent_dim)
            self.decoder.load_state_dict(torch.load('trained_models/vae/decoder.zip'))
            
            self.decoder.eval()
            for seq in self.decoder.decoder.children():
                for child in seq.children():
                    if isinstance(child, nn.BatchNorm2d):
                        child.track_running_stats = False
                    
            for param in self.decoder.parameters():
                param.requires_grad = False    # freeze all decoder parameters
            
            self.rnn_in_size += 1
        
        if rnn_type == 'LSTM':
            self.rnn_layer = nn.LSTM(input_size=self.rnn_in_size, hidden_size=hidden_size, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn_layer = nn.GRU(input_size=self.rnn_in_size, hidden_size=hidden_size, batch_first=True)
            
        self.fc = nn.Linear(hidden_size, 2)


    def forward(self, x):
        #x.shape = batch, seq, channels, 28, 28
        shape = x.shape
        
        enc_input = x.view(shape[0]*shape[1], 1, 28, 28)
        enc_output = self.encoder(enc_input)
        embds = enc_output[:, :self.latent_dim]
        
        if self.mod_emb:
            enc_input = enc_input.to('cpu')
            enc_output = enc_output.to('cpu')
            embds = embds.to('cpu')
            self.decoder.cpu()
            
            mu, log_var = vae_split(enc_output) 
            sample = vae_reparametrize(mu, log_var)
            
            recon = self.decoder(sample)
            
            vae_loss = torch.zeros(shape[0]*shape[1])
            for i, _ in enumerate(enc_input):
                vae_loss[i] = vae_loss_handler(enc_input[i].view(1, 1, 28, 28),
                                               recon[i].view(1, 1, 28, 28),
                                               enc_output[i].view(1, -1)
                                              ).item()
            vae_loss = vae_loss.view(-1, 1)
            embds = torch.hstack((embds, vae_loss))
            embds = embds.to(self.device)
            
        embds = embds.view(shape[0], shape[1], self.rnn_in_size)
        
        #x in rnn layer shape = batch, seq, latent_dim
        if self.rnn_type == 'LSTM':
            out, (h, c) = self.rnn_layer(embds)
        elif self.rnn_type == 'GRU':
            out, h = self.rnn_layer(embds)
        
        #x in classifier_layer = batch, hidden_size
        y = self.fc(h.view(shape[0], self.hidden_size))
        return y

#######################################################
