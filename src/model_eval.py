import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import itertools

#from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, matthews_corrcoef

def top_k_predictions(y_true, y_pred_probs, classes, k=3):
    return np.array(classes[np.argsort(y_pred_probs, axis=1)[:,:-k-1:-1]])

def top_k_accuracy(y_true, y_pred_probs, classes, k=3):
    top_k_predictions = classes[np.argsort(y_pred_probs, axis=1)[:,-k:]]
    #print(ranked_predictions)
    return np.array([y_true.iloc[i] in top_k_predictions[i] for i in range(len(y_true))]).mean()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix\n(Number of images by class)',
                          cmap=plt.cm.Blues,
                          fontsize=22,
                          label_size=14,
                         colorbar=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = np.round(cm/ cm.sum(axis=1)[:, np.newaxis],2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=label_size+2)
    if colorbar:
        plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=label_size)
    plt.yticks(tick_marks, classes, fontsize=label_size)

    fmt = '.2f' if normalize else 'd'
    #formatter = FuncFormatter(to_percent)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j],fmt),
                 fontdict={'size': fontsize, 'weight': 'bold'},
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=label_size)
    plt.xlabel('Predicted label', fontsize=label_size)

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
