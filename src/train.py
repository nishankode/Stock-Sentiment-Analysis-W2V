from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

def train_model(X_train, X_test, y_train, y_test):
    model = LogisticRegression(class_weight='balanced')

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    with open('../Output/best_model.pickle', 'wb') as f:
        pickle.dump(model, f)


    return y_pred, classification_report(y_test, y_pred)