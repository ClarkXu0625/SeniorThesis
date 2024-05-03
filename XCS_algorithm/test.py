from skXCS import XCS
from skXCS import StringEnumerator
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from classifier_neuRecommend.transform import pca_transform
from classifier_neuRecommend.load_spikes import load_spike

[neg_waveforms, pos_waveforms, neg_label, pos_label, fin_labels] = load_spike()
[X, y] = pca_transform(neg_waveforms, pos_waveforms, fin_labels)
fin_data = np.concatenate([neg_waveforms, pos_waveforms])

#print(fin_data[0])
print(np.array(fin_data))