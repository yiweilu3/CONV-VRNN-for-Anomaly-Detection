from __future__ import print_function
import sys
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import vgg16
import numpy as np
import ast
import os
import random
import torch.utils.data
import torch.backends.cudnn as cudn
import pytorch_msssim
from ConvLSTM import ConvLSTM
import torchvision.utils as vutils

from VRNN import VRNN
from Training_Dataset import TrainingDataset
from GDL import GDL
from loss_function import loss_function
from roll_axis import roll_axis
from Load_Data import load_in_images

# Load Dataset
print('System Arguements: ', sys.argv)
if (len(sys.argv) > 1):
    overall_images = load_in_images(sys.argv[1])
else:
    overall_images = load_in_images()

# Model Training
torch.manual_seed(1)
batch_size = 1
num_epochs=500
model=VRNN(input_size=(16, 16),
                 input_dim= 1024,
                 hidden_dim=[512,256, 128, 64, 32],
                 kernel_size=(3, 3),
                 num_layers=5)
model.cuda()

# Define Transforms
tf = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
# Define Dataloader
train_data = TrainingDataset(overall_images, tf)
dataloader = DataLoader(train_data,batch_size=batch_size)
gl = GDL()
# optimizer
total_step = len(dataloader)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
for eidx, epoch in enumerate(range(num_epochs)):
    folder_path = './results/{}'.format(eidx+1)
    if not os.path.exists(folder_path):
    	os.makedirs(folder_path)
    for idx, data in enumerate(dataloader):
            img, gt = data
            for x in range(len(img)):
                img[x] = Variable(img[x].cuda())
            gt = Variable(gt.cuda())
            recon_batch, kld_loss, recon_loss = model(img)

            #imgs = recon_batch.data.cpu().numpy()[0, :]
            #imgs = roll_axis(imgs)
            #img_path = os.path.join(folder_path,'fig{}.png'.format(idx+1))
            #imsave(img_path ,  imgs)


            msssim, f1, psnr_error = loss_function(recon_batch, gt)
            gdl_loss = gl(recon_batch, gt)
            loss= 100* gdl_loss + recon_loss + 10*(msssim + f1)+kld_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, psnr_error:{:.4f}, msssim:{:.4f}, gdl_loss:{:.4f}, kld_loss:{:.4f}, recon_loss:{:.4f}'.format(epoch+1, num_epochs, idx+1, total_step, loss.item(), psnr_error,msssim, gdl_loss,kld_loss, recon_loss))
    if not os.path.exists('./models'):
    	os.makedirs('./models')
    torch.save(model.state_dict(), './models/{}_model-ped.pt'.format(eidx + 1))


