import torch
import torch.nn as nn
from RB_ZTF.scripts.vae import *
from sklearn import metrics
import numpy as np

#### only RNN
class RBclassifier(nn.Module):
    def __init__(self, hidden_size, latent_dim, rnn_type='LSTM', bidirectional=False, device='cpu', out_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.device = device
        self.latent_dim = latent_dim
        self.rnn_in_size = self.latent_dim
        self.bidir = bidirectional
        self.out_size = out_size
        
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
        
        if out_size == 1:
            self.clf = nn.Sequential(
                                 nn.Linear(D*hidden_size, 1),
                                 #nn.ReLU(),
                                 #nn.Linear(64, 2)
                                 nn.Sigmoid(),
                                )
        elif out_size == 2:
            self.clf = nn.Sequential(
                                 #nn.Mish(),
                                 nn.Linear(D*hidden_size, 2),
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
        
        if self.out_size == 1:
            return y.view(-1)
        elif self.out_size == 2:
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
    writer,
):

    result = {'train_loss':[], 'train_acc':[], 'train_roc_auc':[],
              'test_loss':[], 'test_acc':[], 'test_roc_auc':[]
             }
    ep_loss = 0
    batch_count = 0
    all_lab, all_out = [], []
    for batch_idx, (data, lab) in enumerate(train_loader): 
        all_lab.append(lab)
        batch_size = data.size(0)
        optimizer.zero_grad()
        data = data.to(device)
        lab = lab.to(device)

        output = model(data) 
        all_out.append(output.to('cpu'))
        loss = criterion(output, lab)
        loss.backward()
        optimizer.step()
        ep_loss += loss.item()
        batch_count += 1


    all_out, all_lab = torch.concat(all_out), torch.concat(all_lab)
    train_scores = get_scores(pred_from_out(all_out, model.out_size), all_lab)
    train_scores['loss'] = ep_loss / batch_count
    
    model.cpu()
    model.device = 'cpu'
    test_res = validate(model, test_loader, criterion)
    model.device = device
    model.cuda()
    
    result['train_loss'].append( train_scores['loss'] )
    result['train_acc'].append( train_scores['accuracy'] )
    result['train_roc_auc'].append( train_scores['roc_auc'] )

    result['test_loss'].append( test_res['loss'] )
    result['test_acc'].append( test_res['accuracy'] )
    result['test_roc_auc'].append( test_res['roc_auc'] )
    
    writer.add_scalars("Loss", {'train': train_scores['loss'],
                                'test': test_res['loss']}, epoch)
    
    writer.add_scalars("Accuracy", {'train': train_scores['accuracy'],
                                    'test': test_res['accuracy']}, epoch)
    
    writer.add_scalars("ROC-AUC", {'train': train_scores['roc_auc'],
                                   'test': test_res['roc_auc']}, epoch)

    writer.add_scalars("F1-score", {'train': train_scores['f1'],
                                   'test': test_res['f1']}, epoch)
    
    return result


@torch.inference_mode()
def pred_from_out(output, out_size):
    if out_size == 2:
        return nn.Softmax(dim=1)(output)[:, 1]
    else:
        return output


@torch.inference_mode()
def get_pred(
             model,
             loader,
             ):
    if model.out_size == 2:
        result, gt = torch.zeros(1,2), torch.zeros(1)
        for x, label in loader:
            output = model(x)
            result = torch.vstack((result, output))
            gt = torch.hstack((gt, label))
        return nn.Softmax(dim=1)(result[1:])[:, 1], gt[1:], result[1:]
    else:
        result, gt = torch.zeros(1), torch.zeros(1)
        for x, label in loader:
            output = model(x)
            result = torch.hstack((result, output))
            gt = torch.hstack((gt, label))    
        return result[1:], gt[1:]



@torch.inference_mode()
def get_scores(out, gt):
    result = {}
    
    fpr, tpr, thresholds = metrics.roc_curve(gt, out)             
    result['tpr'] = tpr
    result['fpr'] = fpr
    result['thresholds'] = thresholds
    result['roc_auc'] = metrics.roc_auc_score(gt, out)
    
    f1scores = []
    for tr in thresholds:
        current_predict = (out >=tr).long()
        f1scores.append(metrics.f1_score(gt, current_predict))
        #result['f1'].append(f1scores)
    result['f1'] = np.max(f1scores)
    
    ind = np.argmax(f1scores)
    cur_best_thr = result['thresholds'][ind]
    
    result['precision'] = metrics.precision_score(gt, (out >= cur_best_thr).long())
    result['recall'] = metrics.recall_score(gt, (out >= cur_best_thr).long())
    result['accuracy'] = metrics.accuracy_score(gt, (out >= cur_best_thr).long())
    return result

@torch.inference_mode()
def validate(
             model,
             loader,
             criterion,
             ):

    if model.out_size == 1:
        out, gt = get_pred(model, loader)
    else:
        proba, gt, out = get_pred(model, loader)
        gt = gt.long()
    result = get_scores(proba, gt)
    result['loss'] = criterion(out, gt).item()
    return result

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
