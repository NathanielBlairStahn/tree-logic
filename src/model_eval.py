import numpy as np
import pandas as pd

#from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, matthews_corrcoef

def top_k_accuracy(y_true, y_pred, classes, k=3):
    ranked_predictions = classes[np.argsort(y_pred, axis=1)[:,-k:]]
    #print(ranked_predictions)
    return np.array([y_true.iloc[i] in ranked_predictions[i] for i in range(len(y_true))]).mean()

class ModelEvaluator():
    def __init__(self, model): #, image_df):
        self.model = model
        #self.image_df = image_df

    def print_classifier_metrics(self, X_train, X_test, y_train, y_test):
        y_pred_train = self.model.predict_proba(X_train)
        y_pred_test = self.model.predict_proba(X_test)

        y_pred_train_cat = self.model.predict(X_train)
        y_pred_test_cat = self.model.predict(X_test)

        print("Train log_loss: {}, Test log_loss: {}".format(
            log_loss(y_train, y_pred_train), log_loss(y_test, y_pred_test)))

        print("Train Matthews CC: {}, Test Matthews CC: {}".format(
                matthews_corrcoef(y_train, y_pred_train_cat),
                matthews_corrcoef(y_test, y_pred_test_cat))
             )

        print("Train accuracy: {}, Test accuracy: {}".format(
                accuracy_score(y_train, y_pred_train_cat),
                accuracy_score(y_test, y_pred_test_cat))
             )

        #Add top-k accuracy
        print("Train top_2_accuracy: {}, Test top_2_accuracy: {}".format(
                top_k_accuracy(y_train, y_pred_train, self.model.classes_, k=2),
                top_k_accuracy(y_test, y_pred_test, self.model.classes_, k=2))
            )

        print("Train top_3_accuracy: {}, Test top_3_accuracy: {}".format(
                top_k_accuracy(y_train, y_pred_train, self.model.classes_, k=3),
                top_k_accuracy(y_test, y_pred_test, self.model.classes_, k=3))
            )

    def confusion_df(self, X, y_true):
        y_pred = self.model.predict(X)
        return pd.DataFrame(confusion_matrix(y_true, y_pred),
                               index=['Actual {}'.format(label) for label in self.model.classes_],
                               columns=['Predicted {}'.format(label) for label in self.model.classes_])

    def predicted_probs_vs_true_df(self, X, y_true):
        probs = self.model.predict_proba(X)
        df = pd.DataFrame(probs,
                          index = X.index,
                          columns = self.model.classes_)
        df['True'] = y_true
        return df
