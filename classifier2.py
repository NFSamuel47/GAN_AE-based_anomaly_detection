#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 15:17:37 2023

@author: samuel.nchare
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import pathlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, RocCurveDisplay


#Creation of directory where graphics will be saved
graphic_dir = "graphics"
if not os.path.exists(graphic_dir):
    os.makedirs(graphic_dir)
    print(f"Directory '{graphic_dir}' created.")
else:
    print(f"Directory '{graphic_dir}' already exists.")


#import datasets
dataset_mse0 = pd.read_excel("score_a0_izif.xlsx")
dataset_mse05 = pd.read_excel("score_a05_izif.xlsx")
dataset_mse1 = pd.read_excel("score_a1_izif.xlsx")
dataset_cos0 = pd.read_excel("score_a0_cosine.xlsx")
dataset_cos05 = pd.read_excel("score_a05_cosine.xlsx")
dataset_cos1 = pd.read_excel("score_a1_cosine.xlsx")
dataset_euc0 = pd.read_excel("score_a0_euclidean.xlsx")
dataset_euc05 = pd.read_excel("score_a05_euclidean.xlsx")
dataset_euc1 = pd.read_excel("score_a1_euclidean.xlsx")
dataset_man0 = pd.read_excel("score_a0_manhattan.xlsx")
dataset_man05 = pd.read_excel("score_a05_manhattan.xlsx")
dataset_man1 = pd.read_excel("score_a1_manhattan.xlsx")
dataset_pd0 = pd.read_excel("score_a0_torchpd.xlsx")
dataset_pd05 = pd.read_excel("score_a05_torchpd.xlsx")
dataset_pd1 = pd.read_excel("score_a1_torchpd.xlsx")


dataset_afs=pd.read_excel("avgFeatScore_ntwk560.xlsx")
dataset_als = pd.read_excel("avgLatentScore_ntwk560.xlsx")

# data normalization
def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


# definition of violin_plot function
def violin_plot(plot_name, data=dataset_mse0):
    
    '''by default, data=dataset_mse0. this argument must be changed if the dataset is different;
       plot_name is the name of plot to use with a directory in order to save figure.
       example : plot_name = '/violin10.png' 
    '''   
    plt.figure() #create an empty figure in order to avoid fusion of traceplots after several instances.
    x='label'; y='score'; palette='Blues_r'; saturation=0.7   
    v_plot=sns.violinplot(x=x, y=y, data=data, palette=palette, saturation=saturation)
    violin=v_plot.get_figure()
    return violin.savefig(graphic_dir + plot_name)
    plt.close(violin)  # close the figure to release memory

def sigmoid_transform (data, val_lim):
    
    
    ''' apply sigmoid function to anomaly scores in order to transform them between 0 and 1;
        data is a dataframe with contains a column named 'score';
        val_lim is a treshold value beyond with score are considered in class 1.
    '''
        
    pred = []; pred_roc = []
    #definition of sigmoid function 
    def sigmoid (x):
        return 1/(1 + np.exp(-x))
          
    for score in data.score:
        ypred = ypred_roc = sigmoid(score)
        if ypred <= sigmoid(val_lim):
            ypred=0 
        else:
            ypred=1
        pred_roc.append(ypred_roc); pred.append(ypred)
    return pred, pred_roc

def plot_confusion_matrix (y_true, y_pred, pcm_name):
    ''' plot and save confusion matrix in the specified directory:
        y_true is the target label; y_pred is predicted label;
        pcm_name is the name of the figure in the directory.
        example: pcm_name = '/confmatrx10'
    '''
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp = disp.plot(cmap=plt.cm.Blues_r)
    return disp.figure_.savefig(graphic_dir + pcm_name)
    
def plot_classification_report(y_test, y_pred, title='Classification Report', figsize=(8, 3), dpi=400, save_fig_path=None, **kwargs):
    """
    Plot the classification report of sklearn
    
    Parameters
    ----------
    y_test : pandas.Series of shape (n_samples,)
        Targets.
    y_pred : pandas.Series of shape (n_samples,)
        Predictions.
    title : str, default = 'Classification Report'
        Plot title.
    fig_size : tuple, default = (8, 3)
        Size (inches) of the plot.
    dpi : int, default = 70
        Image DPI.
    save_fig_path : str, defaut=None
        Full path where to save the plot. Will generate the folders if they don't exist already.
    **kwargs : attributes of classification_report class of sklearn
    
    Returns
    -------
        fig : Matplotlib.pyplot.Figure
            Figure from matplotlib
        ax : Matplotlib.pyplot.Axe
            Axe object from matplotlib
    """    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
    clf_report = classification_report(y_test, y_pred, output_dict=True, **kwargs)
    keys_to_plot = [key for key in clf_report.keys() if key not in ('accuracy', 'macro avg', 'weighted avg')]
    df = pd.DataFrame(clf_report, columns=keys_to_plot).T
    #the following line ensures that dataframe are sorted from the majority classes to the minority classes
    df.sort_values(by=['support'], inplace=True) 
    
    #first, let's plot the heatmap by masking the 'support' column
    rows, cols = df.shape
    mask = np.zeros(df.shape)
    mask[:,cols-1] = True
 
    ax = sns.heatmap(df, mask=mask, annot=True, cmap="mako", fmt='.3g',
            vmin=0.0,
            vmax=1.0,
            linewidths=2, linecolor='white'
                    )
    
    #then, let's add the support column by normalizing the colors in this column
    mask = np.zeros(df.shape)
    mask[:,:cols-1] = True    
    
    ax = sns.heatmap(df, mask=mask, annot=True, cmap="mako", cbar=False,
            linewidths=2, linecolor='white', fmt='.0f',
            vmin=df['support'].min(),
            vmax=df['support'].sum(),         
            norm=mpl.colors.Normalize(vmin=df['support'].min(),
                                      vmax=df['support'].sum())
                    ) 
            
    plt.title(title)
    plt.xticks(rotation = 0)
    plt.yticks(rotation = 360)
         
    if (save_fig_path != None):
        path = pathlib.Path(save_fig_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig_path)
    
    return fig, ax

def plot_roc_curve (y_true, val_pred, prc_name):
    
    ''' plot and save roc curve in the specified directory:
        y_true is the target label; val_pred is predicted value, by sigmoid function;
        pcm_name is the name of the figure in the directory.
        example: prc_name = '/roc10'
    '''
    plt.figure(dpi=400)
    roc=RocCurveDisplay.from_predictions(y_true, val_pred)
    return roc.figure_.savefig(graphic_dir + prc_name)
# #---------------------------------------------------------------------------------------------------------
# ######################################################
# #                   Non normalized data              #
# #                                                    #
# ######################################################

# ## Case1: k = 1

# #Trace and save violinplot
# violin_plot(plot_name='/violon_avg1080.png', data=dataset_avg1080)

# #transform anomaly scores to prediction labels and prediction values
# pred1, value1 = sigmoid_transform(data=dataset_avg1080, val_lim=37)

# #plot and save confusion matrix
# plot_confusion_matrix(y_true=dataset_avg1080.label, y_pred=pred1, pcm_name='/confMatrix_avg1080.png')

# #plot and save classification report
# fig_k1, ax_k1 = plot_classification_report(dataset_avg1080.label, pred1, 
#                     title='Classification Report for avgscore1080',
#                     figsize=(8, 3), dpi=400,
#                     target_names=["naevus", "melanoma"], 
#                     save_fig_path = graphic_dir + "/report_avg1080.png")

# #plot and save roc curve
# plot_roc_curve(y_true=dataset_avg1080.label, val_pred=value1, prc_name='/roc_avg1080.png')

# #----------------------------------------------------------------------------------------------------------

# ## Case2: k = 0

# #Trace and save violinplot
# violin_plot(plot_name='/violon_k0.png', data=dataset_k0)

# #transform anomaly scores to prediction labels and prediction values
# pred2, value2 = sigmoid_transform(data=dataset_k0, val_lim=0.024)

# #plot and save confusion matrix
# plot_confusion_matrix(y_true=dataset_k0.label, y_pred=pred2, pcm_name='/confMatrix_k0.png')

# #plot and save classification report
# fig_k2, ax_k2 = plot_classification_report(dataset_k0.label, pred2, 
#                     title='Classification Report for k=0',
#                     figsize=(8, 3), dpi=400,
#                     target_names=["naevus", "melanoma"], 
#                     save_fig_path = graphic_dir + "/report_k0.png")

# #plot and save roc curve
# plot_roc_curve(y_true=dataset_k0.label, val_pred=value2, prc_name='/roc_k0.png')

#----------------------------------------------------------------------------------------------------------


######################################################
#             normalized data  (anomaly score)       #
#             before applying sigmoid function       #
######################################################


## Case1: alpha = 0

#Trace and save violinplot
violin_plot(plot_name='/violon_euc0.png', data=dataset_euc0)

# create a dataframe with scores normalized
norm_score = normalize_data(dataset_euc0.score)
dataset_euc0n = dataset_euc0.assign(score=norm_score)

#transform anomaly scores to prediction labels and prediction values
val_lim=norm_score.median()
pred0, value0 = sigmoid_transform(data=dataset_euc0n, val_lim=val_lim)

#plot and save confusion matrix
plot_confusion_matrix(y_true=dataset_euc0n.label, y_pred=pred0, pcm_name='/confMatrix_euc0.png')

#plot and save classification report
fig_a0, ax_0 = plot_classification_report(dataset_euc0n.label, pred0, 
                    title='Classification Report: Euclidean distance and alpha = 0',
                    figsize=(8, 3), dpi=400,
                    target_names=["naevus", "melanoma"], 
                    save_fig_path = graphic_dir + "/report_euc0.png")

#plot and save roc curve
plot_roc_curve(y_true=dataset_euc0n.label, val_pred=value0, prc_name='/roc_euc0.png')

# #---------------------------------------------------------------------------------------------------------

## Case2: alpha = 0.5

#Trace and save violinplot
violin_plot(plot_name='/violon_euc05.png', data=dataset_euc05)

# create a dataframe with scores normalized
norm_score = normalize_data(dataset_euc05.score)
dataset_euc05n = dataset_euc05.assign(score=norm_score)

#transform anomaly scores to prediction labels and prediction values
val_lim=norm_score.median()
pred05, value05 = sigmoid_transform(data=dataset_euc05n, val_lim=val_lim)

#plot and save confusion matrix
plot_confusion_matrix(y_true=dataset_euc05n.label, y_pred=pred05, pcm_name='/confMatrix_euc05.png')

#plot and save classification report
fig_a05, ax_05 = plot_classification_report(dataset_euc05n.label, pred05, 
                    title='Classification Report for Euclidean distance and alpha = 0.5',
                    figsize=(8, 3), dpi=400,
                    target_names=["naevus", "melanoma"], 
                    save_fig_path = graphic_dir + "/report_euc05.png")

#plot and save roc curve
plot_roc_curve(y_true=dataset_euc05n.label, val_pred=value05, prc_name='/roc_euc05.png')

# #---------------------------------------------------------------------------------------------------------

## Case3: alpha = 1

#Trace and save violinplot
violin_plot(plot_name='/violon_euc1.png', data=dataset_euc1)

# create a dataframe with scores normalized
norm_score = normalize_data(dataset_euc1.score)
dataset_euc1n = dataset_euc1.assign(score=norm_score)

#transform anomaly scores to prediction labels and prediction values
val_lim=norm_score.median()
pred1, value1 = sigmoid_transform(data=dataset_euc1n, val_lim=val_lim)

#plot and save confusion matrix
plot_confusion_matrix(y_true=dataset_euc1n.label, y_pred=pred1, pcm_name='/confMatrix_euc1.png')

#plot and save classification report
fig_a1, ax_1 = plot_classification_report(dataset_euc1n.label, pred1, 
                    title='Classification Report for Euclidean distance and alpha = 1',
                    figsize=(8, 3), dpi=400,
                    target_names=["naevus", "melanoma"], 
                    save_fig_path = graphic_dir + "/report_euc1.png")

#plot and save roc curve
plot_roc_curve(y_true=dataset_euc1n.label, val_pred=value1, prc_name='/roc_euc1.png')


# #---------------------------------------------------------------------------------------------------------
# ######################################################
# # normalized data, without too higher anomaly score  #
# #         before applying sigmoid function           #
# ######################################################


# ## Case 4: k = 1

# #Delete anomaly score higher than 20 for naevus and 30 for melanoma
# dataset_k1n2 = dataset_k1.copy()
# index_score1 = dataset_k1n2[ (dataset_k1n2['score'] >= 20) & (dataset_k1n2['label'] == 0) ].index
# index_score2 = dataset_k1n2[ (dataset_k1n2['score'] >= 30) & (dataset_k1n2['label'] == 1) ].index
# dataset_k1n2.drop(index_score1 , inplace=True)
# dataset_k1n2.drop(index_score2 , inplace=True)

# #Trace and save violinplot
# violin_plot(plot_name='/violon_k1n2.png', data=dataset_k1n2)

# # create a dataframe with score replaced by normalized score
# norm_score2 = normalize_data(dataset_k1n2.score)
# dataset_k1n2 = dataset_k1n2.assign(score=norm_score2)

# #transform anomaly scores to prediction labels and prediction values
# pred4, value4 = sigmoid_transform(data=dataset_k1n2, val_lim=0.34)

# #plot and save confusion matrix
# plot_confusion_matrix(y_true=dataset_k1n2.label, y_pred=pred4, pcm_name='/confMatrix_k1n2.png')

# #plot and save classification report
# fig_k4, ax_k4 = plot_classification_report(dataset_k1n2.label, pred4, 
#                     title='Classification Report for k=1 & normalized scores & without out of range scores',
#                     figsize=(8, 3), dpi=400,
#                     target_names=["naevus", "melanoma"], 
#                     save_fig_path = graphic_dir + "/report_k1n2.png")

# #plot and save roc curve
# plot_roc_curve(y_true=dataset_k1n2.label, val_pred=value4, prc_name='/roc_k1n2.png')
