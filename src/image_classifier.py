import numpy as np
import pandas as pd

import os
#from time import time

from PIL import Image
import imagehash

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

def best_logistic_regression():
    """
    Returns the logistic regression model that had the best performance on
    predicting tree species from the extracted image features.

    Currently returns an untrained model; I could pickle the trained model and
    have an option to return that instead.
    """
    #Best parameters found so far. Performing cross-validation may find better ones.
    return LogisticRegression(multi_class='multinomial', solver='sag', C=0.001, max_iter=2000)

def best_gradient_booster():
    """
    Returns the gradient boosting model that had the best performance on
    predicting tree species from the extracted image features.

    Currently returns an untrained model; I could pickle the trained model and
    have an option to return that instead.
    """
    #Best parameters found so far. Performing cross-validation may find better ones.
    return GradientBoostingClassifier(learning_rate=0.01, n_estimators=200, subsample=0.5, max_depth=5)

class ImageClassifier():
    def __init__(self):
        self.feature_extractor = InceptionV3(include_top=False
                                        , weights='imagenet'
                                        , pooling='avg')

        self.num_features = self.feature_extractor.output.shape[1].value # = 2048 for IV3
        self.feature_columns = [f'incv3_out_{i}' for i in range(self.num_features)]

        self.rescale_factor = 1.0/255 #To rescale RGB values to the range [0,1]

        self.target_size = (299,299) #(height,width)
        self.num_channels = 3 #one channel for each color: R, G, B

        self.predictor = None #self.simple_nn_model(num_categories=3)

    def extract_features_from_array(self, image_array, rescale=True):
        """
        Returns the array of features extracted through InceptionV3
        from an array of image data.

        Parameters
        ----------
        image_array : array
            4D array of images, of shape
            (num_images, height, width, channels), which is
            (num_images, 299, 299, 3) for Inception V3.

            The 3 channels are the RGB values for a pixel, in the range [0,255].

        Returns
        -------
        features : array
            The extracted features, an array of float64 of size
            (num_images, num_features), which is (num_images, 2048)
            for Inception V3.
        """
        #Rescale all RGB values in the image array
        #self.rescale_factor should probably be either 1
        #to keep values the same, or 1/255 to rescale to [0,1].
        if rescale:
            image_array = image_array * self.rescale_factor

        return self.feature_extractor.predict(image_array) #features array

    def extract_features_from_path_df(self,
                                      image_df,
                                      base_directory,
                                      folder_col='folder',
                                      file_col='filename',
                                      verbose=True):
        """Extracts features from images whose paths are stored in a DataFrame
        """

        #Create an array to store one image at a time.
        #Later we can introduce a batch size greater than 1.
        #batch_size = 1
        num_images = len(image_df)
        image_array = np.empty((1, *self.target_size, self.num_channels))
        features = np.empty((len(image_df), len(self.feature_columns)))

        #Get indices of the folder and file columns in image_df.
        #This allows us to use .iloc to access the file paths and store the
        #results in the features array in row order rather than index order,
        #so we don't have to worry about what the index of image_df looks like.
        folder_idx = image_df.columns.get_loc(folder_col)
        file_idx = image_df.columns.get_loc(file_col)

        for row_num in range(num_images):
            if verbose and row_num % 100 == 0:
                timestamp = pd.Timestamp.now()
                print(f'{row_num} images processed. Time = {timestamp}.')

            #Get the filepath of the current image from the dataframe.
            image_path = os.path.join(base_directory,
                                      image_df.iloc[row_num, folder_idx],
                                      image_df.iloc[row_num, file_idx])

            #Load the image using Keras.image
            img = image.load_img(image_path, target_size=self.target_size)
            #The image array will contain unscaled RGB values in [0,255].
            #Add a dimension to the 3-D image array (299,299,3)  to make it 4-D (1,299,299,3).
            image_array[0] = image.img_to_array(img)#[np.newaxis,:]
            #print(image_array.shape)
            #Pass the image array to the feature extractor and store the features.
            #This method will rescale the features to the range [0,1] before
            #passing them through Inception V3.
            features[row_num] = self.extract_features_from_array(image_array)

        if verbose:
            timestamp = pd.Timestamp.now()
            print(f'{row_num} images processed. Time = {timestamp}.')

        features_df = pd.DataFrame(features, index = image_df.index, columns=self.feature_columns)
        return image_df.join(features_df)

    def simple_nn_model(self, num_categories, hidden_units=1024):
        features = self.feature_extractor.output
        # print(features.shape)
        # print(features.shape[1:])
        # print(type(features.shape))
        # print(tuple(features.shape[1:]))
        input_shape = tuple(features.shape.as_list())[1:] #=(2048,)
        model = Sequential()
        # model.add(BatchNormalization(input_shape=input_shape))

        #A do-nothing first layer to pass in the input shape
        model.add(Dropout(rate=1.0,input_shape=input_shape))
        model.add(Dense(hidden_units, kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(rate=0.5))
        # model.add(Dense(512, activation="relu"))
        #model.add(Dropout(rate=0.5))
        model.add(Dense(num_categories, activation='softmax', kernel_regularizer=l2(0.001)))
        model.compile(optimizer="adam",
            loss='categorical_crossentropy',
            metrics=['accuracy', 'categorical_accuracy', 'top_k_categorical_accuracy'])
        return model

    def fit_predictor(self, X_train, y_train, epochs=18, batch_size=32,verbose=1):
        self.predictor.fit(self, X_train.values, y_train.values, epochs=epochs, batch_size=batch_size,verbose=verbose)

    def predict_from_features(self, X):
        return self.predictor.predict(X.values)
