import numpy as np
import pandas as pd
from classifier_neuRecommend.pca_transform import transform
from classifier_neuRecommend.load_spikes import load_spike

from skXCS import XCS
from skXCS import StringEnumerator
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

[neg_waveforms, pos_waveforms, neg_label, pos_label, fin_labels] = load_spike()
[X, y] = transform(neg_waveforms, pos_waveforms, fin_labels)

# Split into Train, Test, and Validation sets
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.5, random_state=1)

X_train, X_val, y_train, y_val = \
    train_test_split(X_train, y_train, test_size=0.25, random_state=1)


# #Initialize and train model
# model = XCS(learning_iterations = 500)  # 5000
# trainedModel = model.fit(X_train,y_train)

# print(trainedModel)


# trainedModel.predict(X_test)
# trainedModel.score(X_test, y_test)


# Initialize and train model
model = XCS(learning_iterations = 500)  # 5000
trainedModel = model.fit(X_train,y_train)

print("Training completed.")

print(model)

# Predict using the trained model
predictions = []
for i in tqdm(range(len(X_test)), desc="Prediction progress"):
    prediction = model.predict(np.array([X_test[i]]))
    predictions.append(prediction)

# Calculate accuracy
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)