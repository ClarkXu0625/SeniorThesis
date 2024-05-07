# import tables
import numpy as np
from scipy.stats import zscore

from sklearn.decomposition import PCA as pca
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
import pywt
import json
# import xgboost as xgb
import os

    #######################################################################
    ## preparing dataset, wavele transformation for dimension reduction ###
    #######################################################################
def apply_wavelet_transform(signal, wavelet='db1'):
    # You can adjust the 'level' based on your specific needs or leave it to determine automatically
    coeffs = pywt.wavedec(signal, wavelet, level=2)  # Auto-select the level of decomposition
    threshold = 0.2
    coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]

    # Reconstruct the signal using the thresholded coefficients
    reconstructed_signal = pywt.waverec(coeffs, 'db1')
    
    return reconstructed_signal

def wavelet_transform(fin_data):
    wavelet_coeffs = []

    for signal in fin_data:
        coeffs = apply_wavelet_transform(signal)
        wavelet_coeffs.append(coeffs)

    return np.array(wavelet_coeffs)

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


def signal_to_noise_ratio(waveform):
    # assume signal region ranges from 20th time step to 40th time step.
    # waveform is numpy array
    signal_region = waveform[20:40]
    noise_region = np.concatenate((waveform[0:20], waveform[40:len(waveform)]))
    
    
    signal_amplitude = np.max(signal_region) - np.min(signal_region)
    noise_level = np.std(noise_region)
    
    return signal_amplitude/noise_level

def get_snr(dataset):
    size = len(dataset)
    neg_SNR = np.zeros(size)
    for i in range(0, size):
        neg_SNR[i] = signal_to_noise_ratio(dataset[i])
        
    return neg_SNR

def is_pos_deflection(waveform):
    return (waveform[30] == np.max(waveform[20:30]))

def is_max_or_min(waveform):
    if is_pos_deflection:
        return (waveform[30] == np.max(waveform))
    else:
        return (waveform[30] == np.min(waveform))
    
def get_derivative(waveform):
    derivative = np.zeros(len(waveform)-1)
    for i in range(0, len(derivative)):
        derivative[i] = waveform[i+1]-waveform[i]
    return derivative
    

def feature_extraction(waveform, a=0, b=8):
    '''feature consist of the rest 10 digits
    11. If the waveform downward deflection point is above -20 mV;
    12. If the waveform downward deflection point is in range between -20 ~ -30 mV;
    13. If the waveform downward deflection point is in range between -30 ~ -50 mV;
    14. If the waveform is positively deflected;
    15. If the deflection point is either the maximum or minimum point of the waveform
    16. If the waveform SNR is below 5.55;
    17. SNR value
    18

    '''
    feature = []
    snr = signal_to_noise_ratio(waveform)

    feature.append(waveform[30] < -20)  # 11
    feature.append(waveform[30] < -30)  # 12
    feature.append(waveform[30] < -50)  # 13
    feature.append(is_pos_deflection(waveform)) # 14
    feature.append(is_max_or_min(waveform))     # 15   
    feature.append(snr<5.55)    # 16
    feature.append(snr)         # 17
    feature.append(waveform[30])    # 18
    return feature[a:b]

# def get_cluster_pred(X_train, y_train, X):
#     model_save_dir = 'classifier_neuRecommend/model_new'
#     optim_params_path = os.path.join(model_save_dir, 'optim_params.json')

#     with open(optim_params_path, 'r') as outfile:
#         best_params = json.load(outfile)

#     clf = xgb.XGBClassifier(**best_params)
#     clf.fit(X_train, y_train)


#     print("************************************")
#     # Predicting the labels for the test set
#     y_pred = clf.predict(X)
#     return y_pred



