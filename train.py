import numpy as np
from classifier_neuRecommend.transform import pca_transform
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
# X = np.concatenate([neg_waveforms, pos_waveforms])
# y = fin_labels

# Split into Train, Test, and Validation sets
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.1, random_state=1)

# X_train, X_val, y_train, y_val = \
#     train_test_split(X_train, y_train, test_size=0.25, random_state=1)



# Initialize and train model, setting hyperparameters to maximize the perfomance of classification
# model = XCS(
#     N=2000,                 # Increased population size
#     beta=0.2,               # Adjust learning rate
#     theta_GA=25,            # Adjust GA threshold
#     e_0=0.01,         # Lower minimum error threshold
#     theta_sub=50,           # Experience threshold
#     learning_iterations=5000
# )
model = XCS(learning_iterations=5000)

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
df = pd.DataFrame(X_test)
df['TrueLabel'] = y_test
df['Prediction'] = predictions

# Filtering rows where predictions are wrong
wrong_data_df = df[df['TrueLabel'] != df['Prediction']]

wrong_data_df.to_csv('wrong_predictions.csv', index=False)