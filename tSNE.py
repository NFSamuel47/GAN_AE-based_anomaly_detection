# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 11:04:10 2023

@author: Fabrice
"""

import torch
#import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
#import torchvision.models as models
#☺from data_utils import get_data,CustomImageDataset
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from training import networks_stylegan2
import os
import numpy as np
import pandas as pd
# from tsne import bh_sne
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
#import pickle

image_size = 256

''' 
The test_set directory must have two subdirectories, 
one containing nevus images, and the other 
containing melanoma images
'''

test_images = ImageFolder(root="dataset/test_set",
                      transform=transforms.Compose([
                          transforms.Resize(image_size),
                          transforms.CenterCrop(image_size),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      ]))


root_path=os.getcwd()
save_dir="results"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print("path successfully created")
else:
    print("path already exits")
''' 

'''


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#loads encoder networks and trained weights
encoder=networks_stylegan2.encoder.to(device)
encoder.load_state_dict(torch.load("checkpointed/encoder_end.pth", map_location=torch.device('cpu')))
print("encoder weights imported")

image_loader = DataLoader(test_images, 16, shuffle=False)
def gen_features():
    encoder.eval()
    targets_list = []
    outputs_list = []

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(image_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            targets_np = targets.data.cpu().numpy()

            outputs = encoder(inputs)
            outputs_np = outputs.data.cpu().numpy()
            
            targets_list.append(targets_np[:, np.newaxis])
            outputs_list.append(outputs_np)
            
            if ((idx+1) % 10 == 0) or (idx+1 == len(image_loader)):
              print(idx+1, '/', len(image_loader))
            #break

    targets = np.concatenate(targets_list, axis=0)
    outputs = np.concatenate(outputs_list, axis=0).astype(np.float64)

    return targets, outputs

def tsne_plot(save_dir, targets, outputs):
    print('generating t-SNE plot...')
    # tsne_output = bh_sne(outputs)
    tsne = TSNE(random_state=0, perplexity = 30)
    tsne_output = tsne.fit_transform(outputs)

    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df['targets'] = targets

    plt.rcParams['figure.figsize'] = 10, 10
    sns.scatterplot(
        x='x', y='y',
        hue='targets',
        palette=sns.color_palette("hls", 2),
        data=df,
        marker='o',
        legend="full",
        alpha=0.5
    )

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')

    plt.savefig(os.path.join(save_dir,'tsne_NEV&MEL_new2.png'), bbox_inches='tight')
    print('done!')

targets, outputs = gen_features()
tsne_plot(save_dir, targets, outputs)