import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix

class ModelEvaluator():
    def __init__(self, model, image_df):
        self.model = model
        self.image_df = image_df

def print_classifier_metrics(X_train, X_test, y_train, y_test, sklearn_model):
    y_pred_train = sklearn_model.predict_proba(X_train)
    y_pred_test = sklearn_model.predict_proba(X_test)

    print("Train log_loss: {}, Test log_loss: {}".format(
        log_loss(y_train, y_pred_train), log_loss(y_test, y_pred_test)))

    y_pred_train_cat = sklearn_model.predict(X_train)
    y_pred_test_cat = sklearn_model.predict(X_test)
    print("Train accuracy: {}, Test accuracy: {}".format(
            accuracy_score(y_train, y_pred_train_cat),
            accuracy_score(y_test, y_pred_test_cat))
         )

    #Add top-k accuracy

def confusion_df(X, y_true, sklearn_model):
    y_pred = sklearn_model.predict_proba(X)
    return pd.DataFrame(confusion_matrix(y_true, y_pred),
                           index=['Actual {}'.format(label) for label in sklearn_model.classes_],
                           columns=['Predicted {}'.format(label) for label in sklearn_model.classes_])

def predicted_probs_vs_true_df(X, y_true, sklearn_model):
    probs = sklearn_model.predict_proba(X)
    df = pd.DataFrame(probs,
                      index = X.index,
                      columns = sklearn_model.classes_)
    df['True'] = y_true
    return df
