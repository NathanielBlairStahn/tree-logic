import numpy as np
import pandas as pd

import os

from PIL import Image
import imagehash

from keras.applications.inception_v3 import InceptionV3

class ImageClassifier():
    def __init__(self, feature_extractor=None, predictor=None):
        self.target_size = None
        if feature_extractor is None:
            feature_extractor = InceptionV3(include_top=False
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

        self.feature_extractor = feature_extractor
        self.predictor = predictor
        self.image_extensions = ['.png', '.jpg', '.jpeg']

    def add_images_to_df(self, image_df, base_directory, species_names=None, skip_duplicates=False):
    #If no species names are passed, use all subfolders of the base folder as species names
    if species_names is None:
        with os.scandir(base_directory) as base_contents:
            species_names = [entry.name for entry in base_contents if entry.is_dir()]

    #Get the current length of the dataframe -- we'll be adding rows to the end, one for each image file found
    row_num = len(image_df)

    #Iterate through the species names, which are assumed to be subfolders of the base directory.
    for species_name in species_names:

        #Get the full path to the species folder
        species_directory = os.path.join(base_directory, species_name)

        #Get all the DirEntry objects in the folder, and add the relevant information to the dataframe
        with os.scandir(species_directory) as dir_entries:
            for dir_entry in dir_entries:
                #Skip anything that's not an image file
                if os.path.splitext(dir_entry.name)[1] not in image_extensions:
                    continue

                #Use Keras to load the image file into a PIL image object with the correct target size
                #for InceptionV3
                img = image.load_img(dir_entry.path, target_size=(299,299))

                #Refer to https://pypi.python.org/pypi/ImageHash for imagehash documentation.
                #imagehash.phash(img) computes the perception hash of the image (http://www.phash.org/).
                #Note that this hash SHOULDN'T change if the image is loaded in with a target_size
                #different from (299,299), but it's probably best to always use the same target size
                #just to be safe.
                p_hash = str(imagehash.phash(img))

                #Create a new row containing the data for the current file
                new_row = pd.Series({'p_hash': p_hash
                                    , 'filename': dir_entry.name
                                    , 'species': species_name
                                    , 'tags': ''})

                #If skip_duplicates is True, check whether the dataframe already has an image with
                #the same hash, and only add it if not.
                #NOTE: Add an option to update file path for same hash -- in case original was deleted
                if skip_duplicates and p_hash in image_df['p_hash'].values:
                    print("Dataframe already contains an image with this hash value: {}".format(new_row))
                    continue

                image_df.loc[row_num] = new_row
                row_num += 1

    def extract_features_from_image_paths(self, image_file_df, base_directory, ):
        #Create array to store each image as a 3-tensor
        image_array = np.empty((len(image_df), 299, 299, 3))
        print('Loading images...')
        for row_idx in image_df.index:
            image_path = os.path.join(base_directory,
                image_df.loc[row_idx,'species'],
                image_df.loc[row_idx,'filename'])

            img = image.load_img(image_path, target_size=(299,299))
            image_array[row_idx] = image.img_to_array(img)

        #Rescale all RGB values in the image array
        #self.rescale_factor should probably be either 1
        #to keep values the same, or 1/255 to rescale to [0,1].
        image_array = image_array * self.rescale_factor

        #print('Images loaded. Extracting features...')
        features = self.extract_features(image_array)

        print('Features extracted. Returning new dataframe.')
        features_df = pd.DataFrame(features, columns=self.feature_columns)

        return image_df.join(features_df)

    def extract_features(self, image_array):
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
