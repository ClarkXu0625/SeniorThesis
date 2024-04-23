from load_spikes import load_spike

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

# load file
final_dataset = load_spike()
[neg_waveforms, pos_waveforms, neg_label, pos_label, fin_labels] = final_dataset


#####################################################
## preparing dataset, PCA for dimension reduction ###
#####################################################

def zscore_custom(x):
    return zscore(x, axis=-1)
zscore_transform = FunctionTransformer(zscore_custom)

# Aggregate all waveforms
fin_data = np.concatenate([neg_waveforms, pos_waveforms])
# Zscore to homogenize amplitudes
zscore_fin_data = zscore_transform.transform(fin_data)
# No need to fit PCA to ALL datapoints
pca_obj = pca(n_components=10).fit(zscore_fin_data[::1000])
print(f'Explained variance : {np.sum(pca_obj.explained_variance_ratio_)}')
pca_data = pca_obj.transform(zscore_fin_data)

# Scale PCA components
scaler_obj = StandardScaler().fit(pca_data)
X = scaler_obj.transform(pca_data)
y = fin_labels

print(X[1])
