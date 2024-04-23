from joblib import load
from sklearn.metrics import accuracy_score, classification_report

def test(X_test, y_test, path):

    model = load(path)


    # Assuming `X_test` is your test data
    predictions = model.predict(X_test)

    # Assuming `y_test` are the true labels corresponding to `X_test`
    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))
