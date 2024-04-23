# load from labeled waveform dataset (final_dataset.h5)

import os
from time import time
import json

import tables
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
# import xgboost as xgb



h5_path = 'final_dataset.h5'
# model_save_dir = path_vars['model_save_dir'] # Directory in which trained model will be saved


# Load equal numbers of waveforms for pos,neg, split into train,test
# Since positive samples are >> negative, we will subsample from them
neg_path = '/sorted/neg'
pos_path = '/sorted/pos'

neg_waveforms = []
pos_waveforms = []

h5 = tables.open_file(h5_path, 'r')
for x in h5.iter_nodes(neg_path):
    neg_waveforms.append(x[:])

neg_waveforms = np.concatenate(neg_waveforms, axis=0)

# pos_waveforms needs to be of length 75, or 750 that can be downsampled
pos_node_list = list(h5.iter_nodes(pos_path))
# Waveforms with same length as neg_waveforms
pos_matched_units = [x for x in pos_node_list
                     if x.shape[1] == neg_waveforms.shape[1]]
waveforms_per_unit = neg_waveforms.shape[0]//len(pos_matched_units)

# with tables.open_file(h5_path,'r') as h5:
for x in pos_matched_units:
    ind = np.min([x.shape[0], waveforms_per_unit])
    pos_waveforms.append(x[:ind, :])
pos_waveforms = np.concatenate(pos_waveforms, axis=0)
h5.close()

neg_label = [0]*neg_waveforms.shape[0]
pos_label = [1]*pos_waveforms.shape[0]
fin_labels = np.concatenate([neg_label, pos_label])
print(fin_labels.shape)
print(pos_waveforms.shape)
print(neg_waveforms.shape)
plt.plot(neg_waveforms[10,:])
plt.show()