# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 14:24:30 2023

@author: Fabrice
"""

import os
import torch
import pandas as pd
import numpy as np
from training import networks_stylegan2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import pickle


out_dir = "generated_img"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    print(f"Directory '{out_dir}' created.")
else:
    print(f"Directory '{out_dir}' already exists.")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
image_size = 256
input_dir="test_data"    
datatensor = ImageFolder(input_dir,
                      transform=transforms.Compose([
                          transforms.Resize(image_size),
                          transforms.CenterCrop(image_size),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      ]))

dataloader = DataLoader(datatensor, batch_size=batch_size, shuffle=False)

# chargement des poids du générateur

with open('training-runs/00006-stylegan2-trainset-gpus1-batch16-gamma0.8192/network-snapshot-000560.pkl', 'rb') as f:
    generator = pickle.load(f)['G'].to(device)
generator.eval()


encoder=networks_stylegan2.encoder.to(device)
encoder.load_state_dict(torch.load("checkpointed/encoder_end.pth", map_location=torch.device('cpu')))
encoder.eval()

# fetch image name's in subdirectories of input_dir
list_name = [os.path.basename(path) for path, _ in datatensor.imgs]

#initialise empty lists for latent vectors
latent_vectors = []

#construct a list of all latent vectors
with torch.no_grad():
    for (x, _) in dataloader: #and name in list_name:
        x=x.to(device)
        code=encoder(x) 
        for vec in code:           
            latent_vectors.append(vec.cpu().numpy())
    
#associate each latent vector with the name of image it comes from, in a dataframe    
latent_df = pd.DataFrame(latent_vectors, columns=[f"dim_{i+1}" for i in range(512)])
latent_df['image_name'] = list_name

#save images from latent vectors in out_dir directory
for index, row in latent_df.iterrows():
    latent_vector = row.iloc[:-1].values.astype(np.float64)  # Exclure la dernière colonne (noms d'image)
    image_name = row['image_name']

    latent_tensor = torch.tensor(latent_vector, dtype=torch.float64).to(device)
    generated_image = generator(latent_tensor.unsqueeze(0), 0).unsqueeze(0)
    generated_image = generated_image * 0.5 + 0.5

    save_image(generated_image, out_dir + '/' + image_name)

#save pandas dataframe in a csv file
csv_filename = "latent_vectors.csv"
latent_df.to_csv(csv_filename, index=False)
print(f"Latent vectors saved to {csv_filename}")
#_____________________________________________________________________

from torch.nn.functional import pairwise_distance

# # Sélectionnez les vecteurs latents à partir du DataFrame
# latent_vectors = latent_df.iloc[:, :512].values

# # Choisissez l'indice de l'image de référence (image i)
# indice_reference = 25  # Vous pouvez choisir n'importe quel indice

# # Vecteur latent de l'image de référence
# vector_reference = latent_vectors[indice_reference]
# image_reference = latent_df.iloc[indice_reference, 512]  # Nom de l'image de référence

# # Liste pour stocker les résultats
# results = []; results2 = []

# # Boucle pour calculer les distances par rapport à l'image de référence
# for j in range(len(latent_vectors)):
#     vector_j = latent_vectors[j]
#     image_j = latent_df.iloc[j, 512]  # Nom de l'image j
#     diff = ((vector_reference - vector_j)**2)
#     score = diff.mean()
#     results2.append([image_reference, image_j, score])
#     distance = pairwise_distance(torch.tensor(vector_reference).unsqueeze(0), torch.tensor(vector_j).unsqueeze(0)).item()
#     results.append([image_reference, image_j, distance])

# # Créez un DataFrame à partir des résultats
# results_df = pd.DataFrame(results, columns=['ImageReference', 'Image', 'Distance'])
# results2_df = pd.DataFrame(results2, columns=['ImageReference', 'Image', 'score'])
# # Triez le DataFrame par ordre croissant de distances
# results_df = results_df.sort_values(by='Distance')

# results2_df = results2_df.sort_values(by='score')

# # Réinitialisez les index du DataFrame trié
# results_df = results_df.reset_index(drop=True)
# results2_df = results2_df.reset_index(drop=True)

# # Affichez le DataFrame résultant
# print(results_df.head(5))
# print(results2_df.head(5))

# Sélectionnez les vecteurs latents à partir du DataFrame
latent_vectors = latent_df.iloc[:, :512].values
image_names = latent_df.iloc[:, 512].values

# Liste pour stocker les résultats
results = []

# Boucle pour calculer les distances entre toutes les paires d'images
for i in range(len(latent_vectors)):
    vector_i = latent_vectors[i]
    image_i = image_names[i]  # Nom de l'image i
    
    distances = []
    for j in range(len(latent_vectors)):
        if i != j:  # Pour éviter de calculer la distance avec la même image
            vector_j = latent_vectors[j]
            distance = pairwise_distance(torch.tensor(vector_i).unsqueeze(0), torch.tensor(vector_j).unsqueeze(0)).item()
            distances.append(distance)
    
    # Trier les distances par ordre croissant et prendre les 5 plus faibles
    distances.sort()
    smallest_distances = distances[:5]
    
    # Calculer la moyenne des 5 plus faibles distances
    avg_smallest_distances = sum(smallest_distances) / len(smallest_distances)
    
    results.append([image_i, avg_smallest_distances])

# Créez un DataFrame à partir des résultats
results_df = pd.DataFrame(results, columns=['Image', 'AverageDistance'])

# Triez le DataFrame par ordre croissant de distances moyennes
results_df = results_df.sort_values(by='AverageDistance')

# Réinitialisez les index du DataFrame trié
results_df = results_df.reset_index(drop=True)

# Affichez le DataFrame résultant
print(results_df)