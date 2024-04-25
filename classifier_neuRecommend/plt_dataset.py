from load_spikes import load_spike
from model_load import test

import os
from time import time
import json

# import tables
import numpy as np
from tqdm import tqdm
import pylab as plt 
import pandas as pd
from joblib import dump, load
from scipy.stats import zscore

from sklearn.decomposition import PCA as pca
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# load file
final_dataset = load_spike()
[neg_waveforms, pos_waveforms, neg_label, pos_label, fin_labels] = final_dataset


def plot_sample():
    t = np.linspace(0, 75, 75)
    waveform1 = pos_waveforms[10]
    waveform2 = neg_waveforms[1]
    ################
    #### plot ######
    ################

    plt.figure(figsize=(10, 4))  # Set the figure size as needed

    # Plot waveform 1
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    plt.plot(t, waveform1, label='waveform #10')
    plt.title('Sample True Waveform')
    plt.xlabel('Time (au)')
    plt.ylabel('Amplitude (mV)')
    plt.grid(True)
    plt.legend()

    # Plot sample noise
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    plt.plot(t, waveform2, label='waveform #1')
    plt.title('Sample True Noise')
    plt.xlabel('Time (au)')
    plt.ylabel('Amplitude (mV)')
    plt.grid(True)
    plt.legend()

    # Display the plots
    plt.tight_layout()  # Adjust the layout to make room for the labels/titles
    plt.show()


def plot_all():
    ### plot all waveforms
    fig, ax = plt.subplots(1,2, figsize=(12, 6))
    ax[0].imshow(zscore(pos_waveforms,axis=-1), interpolation='nearest', aspect = 'auto')
    ax[0].set_title('True Spikes')
    ax[0].set_xlabel('Time (AU)')
    ax[0].set_ylabel('Waveform #')
    ax[1].imshow(zscore(neg_waveforms,axis=-1), interpolation='nearest', aspect = 'auto')
    ax[1].set_title('True Noise')
    ax[1].set_xlabel('Time (AU)')
    ax[1].set_ylabel('Waveform #')
    plt.tight_layout()
    plt.show()


def plot_zscore(pos_waveforms, neg_waveforms):
    # Calculate z-scores
    z_pos_waveforms = zscore(pos_waveforms, axis=-1)
    z_neg_waveforms = zscore(neg_waveforms, axis=-1)
    
    # Determine the global min and max for a unified color scale
    vmin = min(z_pos_waveforms.min(), z_neg_waveforms.min())
    vmax = max(z_pos_waveforms.max(), z_neg_waveforms.max())

    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))

    # Plot for positive waveforms
    pos_img = ax[0].imshow(z_pos_waveforms, interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)
    ax[0].set_title('True Spikes')
    ax[0].set_xlabel('Time (AU)')
    ax[0].set_ylabel('Waveform #')

    # Plot for negative waveforms
    neg_img = ax[1].imshow(z_neg_waveforms, interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)
    ax[1].set_title('True Noise')
    ax[1].set_xlabel('Time (AU)')
    ax[1].set_ylabel('Waveform #')

    # Add a shared color bar
    cbar = fig.colorbar(pos_img, ax=ax.ravel().tolist(), shrink=0.95)
    cbar.set_label('Z-score')

    # Adjust layout
    #plt.tight_layout()
    plt.show()
plot_sample()
#plot_zscore(pos_waveforms, neg_waveforms)



