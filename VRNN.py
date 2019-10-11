from __future__ import print_function
import argparse
import torch
import torchvision
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision.models import vgg16
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import ast
from torch.nn import functional as F
from math import log10
import random
import torch.utils.data
import torchvision.utils as vutils
import torch.backends.cudnn as cudn
import argparse
import pickle


import pytorch_msssim
from ConvLSTM import ConvLSTM

batch_size = 1

class VRNN(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, bias=False):
        hidden_dim
        super(VRNN, self).__init__()
        self.have_cuda = True
        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias =bias
        

        self.ConvLSTM = ConvLSTM(input_size=(16, 16),
                 input_dim= 1024,
                 hidden_dim=[512,256, 128, 64, 32],
                 kernel_size=(3, 3),
                 num_layers=5, 
                 batch_first=True,
                 bias=True)
        
        self.prior = vgg16(pretrained = False).features[:-1]
        self.encoder = vgg16(pretrained = False)
        self.encoder.features[0] = nn.Conv2d(6,64,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.encoder = self.encoder.features[:-1]
        self.feature = vgg16(pretrained = False).features[:-1]

        self.decoder = nn.Sequential(
                       nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
                       nn.ReLU(True),
                       nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                       nn.ReLU(True),
                       nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                       nn.ReLU(True),
                       nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                       nn.ReLU(True),
                       nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                       nn.ReLU(True),
                       nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                       nn.ReLU(True),
                       nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
                       nn.ReLU(True)
                       )
        self.decoder2 = nn.Sequential(
                       nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1),
                       nn.ReLU(True),
                       nn.ConvTranspose2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                       nn.ReLU(True),
                       nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1),
                       nn.ReLU(True),
                       nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                       nn.ReLU(True),
                       nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                       nn.ReLU(True),
                       nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                       nn.ReLU(True),
                       nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
                       nn.ReLU(True)
                       )
        self.fc11 = nn.Linear(512, 20)
        self.fc12 = nn.Linear(512, 20)
        self.fc2 = nn.Linear(20,512)
                          
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.have_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

                          
                          
    def forward(self,x):
        h = Variable(torch.zeros(x[0].size()).cuda())
        print("shape of h:{}".format(h.size()))
        # hidden_state
        hidden_state = self.ConvLSTM._init_hidden(batch_size= batch_size)      
        for t in range(3):
            # calculate the mean and std of prior
            
            prior_t = self.prior(h)
            prior_vector = prior_t.view(-1, 512)
            prior_mean_t, prior_logvar_t = self.fc11(prior_vector), self.fc12(prior_vector)
            prior_std_t = prior_logvar_t.mul(0.5).exp_()
                          
            # calculate the mean and std of encoder
            enc_t = self.encoder(torch.cat((x[t], h),1))
            enc_t = enc_t.view(-1,512)
            enc_mean_t, enc_logvar_t = self.fc11(enc_t), self.fc12(enc_t)
            enc_std_t = enc_logvar_t.mul(0.5).exp_()              
            z_t = self.reparametrize(enc_mean_t, enc_logvar_t)
                          
            # decoder
            deconv_input=self.fc2(z_t)
            z_t_resize = deconv_input.view(-1,512,16,16)
            dec_t = self.decoder(torch.cat((z_t_resize,prior_t),1))
            
            
            # Recurrence
            feature_x = self.feature(x[t])
            h, hidden_state= self.ConvLSTM(torch.cat((feature_x,z_t_resize),1), hidden_state)
            h = self.decoder2(h)
            if (t == 0):
                kld_loss = -0.5 * torch.sum(1 + enc_logvar_t - enc_mean_t.pow(2) - enc_logvar_t.exp())
            else:
                kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            
            recon_loss=0
            
            recon_loss += self.recons_loss(dec_t, x[t])
                          
        return dec_t, kld_loss, recon_loss
                              
    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) + 
        (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
        std_2.pow(2) - 1)
        return 0.5 * torch.sum(kld_element)
    
    def recons_loss(self,recon_x, x):
        msssim = ((1-pytorch_msssim.msssim(x,recon_x)))/2
        f1 =  F.l1_loss(recon_x, x)
        return msssim+f1
    
    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(5):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states
