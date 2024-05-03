# import tables
import numpy as np
from scipy.stats import zscore

from sklearn.decomposition import PCA as pca
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer


def pca_transform(neg_waveforms, pos_waveforms, fin_labels):

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

    return [X, y]


def feature_extraction(waveform):
    '''feature consist of 10 digits
    1. If peak lower than -20
    2. If peak lower than -30
    3. If peak lower than -50
    '''
    feature = []
    feature.append(waveform[30] < -20)
    feature.append(waveform[30] < -30)
    feature.append(waveform[30] < -50)
    

