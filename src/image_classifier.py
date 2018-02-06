import numpy as np
import pandas as pd

import os
from time import time

from PIL import Image
import imagehash

from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential, Model

class ImageClassifier():
    def __init__(self, categories):
        self.feature_extractor = InceptionV3(include_top=False
                                        , weights='imagenet'
                                        , pooling='avg')

        self.num_features = feature_extractor.output.shape[1].value #=2048
        self.feature_columns = ['incv3_out_{}'.format(i)
                        for i in range(num_features)]
        #InceptionV3 takes in images of size 299x299
        self.target_size = (299,299)
        #Rescale colors to be in range [0,1]. See:
        #https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
        #https://www.linkedin.com/pulse/keras-image-preprocessing-scaling-pixels-training-adwin-jahn/
        #https://github.com/Arsey/keras-transfer-learning-for-oxford102/issues/1
        #It looks like this might not matter unless we want to train a neural
        #network on the images, but I'm not sure...
        self.rescale_factor = 1.0/255

        self.categories = categories
        self.label_dict = {label: i for i, label in enumerate self.categories}

        self.predictor = self.simple_nn_model()

    def extract_features_from_image_paths(self, image_df, base_directory, batch_size=32, label_col='species', file_col='filename'):

        image_array = np.empty((batch_size, *self.target_size, 3))
        features = np.empty((len(image_df), len(self.feature_columns)))

        num_batches = len(image_df) // batch_size
        last_batch_size = len(image_df) % batch_size
        row_idx = 0

        for batch in range(num_batches):
            for batch_idx in range(batch_size):
                image_path = os.path.join(base_directory,
                    image_df.loc[row_idx, label_col],
                    image_df.loc[row_idx, file_col])

                img = image.load_img(image_path, target_size=self.target_size)
                #The image array will contain unscaled RGB values in [0,255]
                image_array[batch_idx] = image.img_to_array(img)
                row_idx += 1

            features[batch*batch_size: (batch+1)*batch_size)] = (
                self.extract_features(image_array, rescale=True)
                )

        for batch_idx in range(last_batch_size):
            image_path = os.path.join(base_directory,
                image_df.loc[row_idx, label_col],
                image_df.loc[row_idx, file_col])

            img = image.load_img(image_path, target_size=self.target_size)
            #The image array will contain unscaled RGB values in [0,255]
            image_array[batch_idx] = image.img_to_array(img)
            row_idx += 1

        features[batch*batch_size:] = (
            self.extract_features(image_array[:last_batch_size], rescale=True)
            )

        # #Original version without batches -- will use too much memory
        # if df is too big...
        # #Create array to store each image as a 3-tensor
        # image_array = np.empty((len(image_df), *self.target_size, 3))
        # t0 = time()
        # print(f'Loading images...time={t0}')
        # for row_idx in image_df.index:
        #     image_path = os.path.join(base_directory,
        #         image_df.loc[row_idx, label_col],
        #         image_df.loc[row_idx, file_col])
        #
        #     img = image.load_img(image_path, target_size=self.target_size)
        #     #The image array will contain unscaled RGB values in [0,255]
        #     image_array[row_idx] = image.img_to_array(img)
        #
        # t1 = time()
        # print(f'Images loaded. Time={t1}'.format(t1-t0))
        # print('Extracting features...')
        # #The default rescale=True will rescale RGB values to be in [0,1]
        # features = self.extract_features(image_array, rescale=True)

        print('Features extracted. Returning new dataframe.')
        features_df = pd.DataFrame(features, columns=self.feature_columns)

        return image_df.join(features_df)

    def extract_features(self, image_array,rescale=True):
        '''
        Returns the array of feature extracted from through InceptionV3
        from an array of image data.

        INPUT: image_array is a 4D array of images, of shape
        (num_images, height, width, channels), which is
        (num_images, 299, 299, 3) for Inception V3.

        RETURNS: an array of float64 of size (num_images, num_features),
        which is (num_images, 2048) for Inception V3.
        '''
        #Rescale all RGB values in the image array
        #self.rescale_factor should probably be either 1
        #to keep values the same, or 1/255 to rescale to [0,1].
        if rescale:
            image_array = image_array * self.rescale_factor

        features = self.feature_extractor.predict(image_array)
        return features

    def simple_nn_model(self):
        features = self.feature_extractor.output
        model = Sequential()
        model.add(BatchNormalization(input_shape=features.shape[1:]))
        model.add(Dense(1024, activation = "relu"))
        model.add(Dense(4, activation='softmax'))
        model.compile(optimizer="SGD",
            loss='categorical_crossentropy',
            metrics=['accuracy', 'categorical_accuracy', 'top_k_categorical_accuracy'])
        return model

    def fit(self, X_train, y_train, epochs=18, batch_size=32,verbose=1):
        self.predictor.fit(self, X_train.values, y_train.values, epochs=epochs, batch_size=batch_size,verbose=verbose)


    def predict(self, image_df):
        '''
        INPUT: pandas dataframe containing an image id in one column
        and the filename or file object in another column.

        OUTPUT: pandas dataframe containing column names equal to labels
        and values equal to predicted probabilities
        '''
        return None
