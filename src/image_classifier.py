import numpy as np
import pandas as pd

import os
from time import time

from PIL import Image
import imagehash

from keras.applications.inception_v3 import InceptionV3

class ImageClassifier():
    def __init__(self, predictor=None):
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

        self.predictor = predictor

    def extract_features_from_image_paths(self, image_file_df, base_directory, label_col='species', file_col='filename'):
        #Create array to store each image as a 3-tensor
        image_array = np.empty((len(image_df), 299, 299, 3))
        t0 = time()
        print('Loading images...')
        for row_idx in image_df.index:
            image_path = os.path.join(base_directory,
                image_df.loc[row_idx, label_col],
                image_df.loc[row_idx,file_col])

            img = image.load_img(image_path, target_size=(299,299))
            #The image array will contain unscaled RGB values in [0,255]
            image_array[row_idx] = image.img_to_array(img)

        print('Images loaded. Extracting features...')
        #The default rescale=True will rescale RGB values to be in [0,1]
        features = self.extract_features(image_array, rescale=True)

        print('Features extracted. Returning new dataframe.')
        features_df = pd.DataFrame(features, columns=self.feature_columns)

        return image_df.join(features_df)

    def extract_features(self, image_array, rescale=True):
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



    def predict(self, image_df):
        '''
        INPUT: pandas dataframe containing an image id in one column
        and the filename or file object in another column.

        OUTPUT: pandas dataframe containing column names equal to labels
        and values equal to predicted probabilities
        '''
        return None
