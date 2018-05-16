import numpy as np
import pandas as pd

import os
#from time import time

from PIL import Image
import imagehash

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
#from keras.models import Sequential, Model

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
