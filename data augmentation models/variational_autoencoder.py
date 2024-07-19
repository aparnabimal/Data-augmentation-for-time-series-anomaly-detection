# Reference: https://github.com/noveens/svae_cf/blob/master/main_svae.ipynb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import time 

is_cuda_available = torch.cuda.is_available()


class Encoder(nn.Module):
    def __init__(self, hyper_params):
        super(Encoder, self).__init__()
        self.lstm1 = nn.LSTM(
            hyper_params['num_dimensions'],
            hyper_params['hidden_size'],
            batch_first=True,
            num_layers=1
        )
        self.linear1 = nn.Linear(hyper_params['hidden_size'], hyper_params['latent_size'])
        self.activation = nn.LeakyReLU()
        
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = x[:, -1, :]
        x = self.activation(self.linear1(x))
        return x

class Decoder(nn.Module):
    def __init__(self, hyper_params, sequence_length):
        super(Decoder, self).__init__()
        self.sequence_length = sequence_length
        self.linear1 = nn.Linear(hyper_params['latent_size'], hyper_params['hidden_size'])
        self.linear2 = nn.Linear(hyper_params['hidden_size'], hyper_params['num_dimensions'] * hyper_params['sequence_length'])
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        return x.view(x.size(0), self.sequence_length, -1)

class Model(nn.Module):
    def __init__(self, hyper_params, sequence_length):
        super(Model, self).__init__()
        self.hyper_params = hyper_params
        self.encoder = Encoder(hyper_params)
        self.decoder = Decoder(hyper_params, sequence_length)
        self.mu_linear = nn.Linear(hyper_params['latent_size'], hyper_params['latent_size'])
        self.log_sigma_linear = nn.Linear(hyper_params['latent_size'], hyper_params['latent_size'])

    def sample_latent(self, mu, log_sigma):
        sigma = torch.exp(log_sigma)
        std_z = torch.randn_like(sigma)
        if is_cuda_available: std_z = std_z.cuda()
        
        self.z_mean = mu
        self.z_log_sigma = log_sigma

        return mu + sigma * std_z

    def forward(self, x):
        h_enc = self.encoder(x)
        
        mu = self.mu_linear(h_enc)
        log_sigma = self.log_sigma_linear(h_enc)

        sampled_z = self.sample_latent(mu, log_sigma)
        dec_out = self.decoder(sampled_z)

        return dec_out, self.z_mean, self.z_log_sigma


hyper_params = {
    'num_dimensions': 1,  
    'hidden_size': 150,
    'latent_size': 64,
    'sequence_length': 12,
    'kld_weight': 0.1,
}

class VAELoss(nn.Module):
    def __init__(self, kld_weight):
        super(VAELoss, self).__init__()
        self.kld_weight = kld_weight

    def forward(self, decoder_output, mu_q, logvar_q, y_true, anneal, batch_size, dataset_size):
        # KL Divergence
        kld = -0.5 * torch.sum(1 + logvar_q - mu_q.pow(2) - logvar_q.exp())
        kld_normalized = kld / dataset_size

        # Reconstruction Loss (MSE)
        recon_loss = nn.MSELoss()(decoder_output, y_true)

        # Combine the two losses with the specified weight for KLD term
        total_loss = recon_loss + self.kld_weight * anneal * kld_normalized

        return total_loss, recon_loss, kld
