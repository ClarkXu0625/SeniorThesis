import numpy as np
from classifier_neuRecommend.transform import pca_transform, wavelet_transform, feature_extraction
from classifier_neuRecommend.load_spikes import load_spike

from skXCS import XCS
from skXCS import StringEnumerator
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

[neg_waveforms, pos_waveforms, neg_label, pos_label, fin_labels] = load_spike()
[X, y] = pca_transform(neg_waveforms, pos_waveforms, fin_labels)

fin_data = np.concatenate([neg_waveforms, pos_waveforms])

attr_num = 3
X_new = np.empty(shape=(len(X), X.shape[1]+attr_num))

for i in range(0, len(fin_data)):
    waveform = fin_data[i]
    X_new[i] = np.concatenate([X[i], feature_extraction(waveform, attr_num)])

# X = wavelet_transform(fin_data)
# y = fin_labels
print('Transform finished')
print(len(X_new[0]))

# Split into Train, Test, and Validation sets
# X_train, X_test, y_train, y_test = \
#     train_test_split(X, y, test_size=0.1, random_state=1)
X_train, X_test, y_train, y_test, waveform_train, waveform_test = \
    train_test_split(X_new, y, fin_data, test_size=0.1, random_state=1)



print("Train started")
# Initialize and train model, setting hyperparameters to maximize the perfomance of classification
# model = XCS(
#     N=2000,                 # Increased population size
#     beta=0.2,               # Adjust learning rate
#     theta_GA=25,            # Adjust GA threshold
#     e_0=0.01,         # Lower minimum error threshold
#     theta_sub=50,           # Experience threshold
#     learning_iterations=5000
# )
model = XCS(N=1000,
            learning_iterations=5000)

trainedModel = model.fit(X_train,y_train)

print("Training completed.")

print(model)

# Predict using the trained model
predictions = []
for i in tqdm(range(len(X_test)), desc="Prediction progress"):
    prediction = model.predict(np.array([X_test[i]]))
    predictions.append(prediction)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Calculate precision
precision = precision_score(y_test, predictions)
print("Precision:", precision)

# Calculate recall
recall = recall_score(y_test, predictions)
print("Recall:", recall)

# Calculate F1 score
f1 = f1_score(y_test, predictions)
print("F1 Score:", f1)

# export the iteration data
trainedModel.export_iteration_tracking_data()

# export the rule population
trainedModel.export_final_rule_population()


#########################################################
### find wrong predictions ##################3
#############################################

# Create a DataFrame from your test data, test labels, and predictions
# df = pd.DataFrame(waveform_test)
# df['waveform'] = df.apply(lambda row: row.tolist(), axis=1) # combine element into single list element
# df_waveform = df[['waveform']]
# df_waveform['TrueLabel'] = y_test
# df_waveform['Prediction'] = np.array(predictions)

# # Filtering rows where predictions are wrong
# wrong_data_df = df_waveform[df_waveform['TrueLabel'] != df_waveform['Prediction']]

# wrong_data_df.to_csv('wrong_predictions.csv', index=False)

print(len(predictions) == len(y_test))

wrong_data = [X[i] for i in range(len(predictions)) if predictions[i] != y_test[i]]
print(len(wrong_data))