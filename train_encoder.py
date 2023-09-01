# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 08:48:09 2023

@author: Fabrice
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import numpy as np
#from torchvision.models import inception_v3
#from scipy.linalg import sqrtm
#from torch.autograd import Variable
#from torch.nn.utils import spectral_norm
#import torch.nn.functional as F
#from modeles import *
from training import networks_stylegan2
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


checkpoint_dir = "checkpointed"
image_size=256
batch_size = 16

# Dataset

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    print(f"Directory '{checkpoint_dir}' created.")
else:
    print(f"Directory '{checkpoint_dir}' already exists.")
    
dataset = ImageFolder(root="dataset/train_set",
                      transform=transforms.Compose([
                          transforms.Resize(image_size),
                          transforms.CenterCrop(image_size),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      ]))

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# load generator weights

with open('training-runs/00005-stylegan2-trainset-gpus1-batch16-gamma0.8192/network-snapshot-000050.pkl', 'rb') as f:
    generator = pickle.load(f)['G'].to(device)
generator.eval()


# load discriminator weights
with open('training-runs/00005-stylegan2-trainset-gpus1-batch16-gamma0.8192/network-snapshot-000050.pkl', 'rb') as f:
    discriminator = pickle.load(f)['D'].to(device)
discriminator.eval()


#train encoder, with izif method

k = 1 # weight factor associated with the discriminator
#freeze GAN weights
for p in discriminator.parameters():
    p.requires_grad = False  
for p in generator.parameters():
    p.requires_grad = False  
    
#import Encoder network        
netE = networks_stylegan2.encoder.to(device)

# optimizer used in f-AnoGAN
optimizer_E = optim.Adam(netE.parameters(), lr=0.0001, betas=(0.0, 0.9)) 
    
crit = nn.MSELoss()

#encoder training loops    
for e in range(1000):
    losses = []
    netE.train()
    for (x, _) in dataloader: #assumes dataloader contains single-class data
        x = x.to(device)
        code = netE(x)
        rec_image = generator(code,0)
        rec_x = rec_image.view(-1,rec_image.size(1), 256, 256)
        d_input = torch.cat((x, rec_x), 0)
        #-------------------------------------------------------------------------------------------------------

        f_x, f_gx = discriminator.extract_feature(d_input,0).chunk(2,0)
        

        #----------------------------------------------------------------------------------------------------        
        loss = crit(rec_x, x) + k * crit(f_gx, f_x.detach()) # izif Loss
        optimizer_E.zero_grad()
        loss.backward()
        optimizer_E.step()
        losses.append(loss.item())
    if (e + 1) % 30 == 0:
        print(e, np.mean(losses))
        netE.eval()
        code2=netE(x)
        rec2_image = generator(code2,0)
        rec2_x = rec2_image.view(-1,rec2_image.size(1), 256, 256)
        d_input = torch.cat((x, rec2_x), 0)
        save_image(d_input*0.5+0.5, 'rec'+str(e)+'.png')
        # Save model checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"encoded_{e+1}.pth")
        torch.save(netE.state_dict(), checkpoint_path)
        print(f"Saved model checkpointed at epoch {e+1}")
torch.save(netE.state_dict(), checkpoint_dir+"/encoder_end.pth")