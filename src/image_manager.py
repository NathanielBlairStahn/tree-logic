import numpy as np
import pandas as pd

import os

from PIL import Image
import imagehash

class ImageManager():
    def __init__(self):
        self.image_extensions = ['.png', '.jpg', '.jpeg']


    def add_images_to_df(self,
                         image_df,
                         base_directory,
                         label_col='species',
                         file_col='filename',
                         hash_fcn=('p_hash', imagehash.phash),
                         species_names=None,
                         skip_duplicates=False):
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
                if os.path.splitext(dir_entry.name)[1] not in self.image_extensions:
                    continue

                #Use Keras to load the image file into a PIL image object with the correct target size
                #for InceptionV3
                img = image.load_img(dir_entry.path, target_size=(299,299))

                #Refer to https://pypi.python.org/pypi/ImageHash for imagehash documentation.
                #imagehash.phash(img) computes the perception hash of the image (http://www.phash.org/).
                #Note that this hash SHOULDN'T change if the image is loaded in with a target_size
                #different from (299,299), but it's probably best to always use the same target size
                #just to be safe.
                hash_val = str(hash_fcn[1](img))

                #Create a new row containing the data for the current file
                new_row = pd.Series({hash_fcn[0]: hash_val
                                    , file_col: dir_entry.name
                                    , label_col: species_name
                                    , 'tags': ''})

                #If skip_duplicates is True, check whether the dataframe already has an image with
                #the same hash, and only add it if not.
                #NOTE: Add an option to update file path for same hash -- in case original was deleted
                if skip_duplicates and p_hash in image_df['p_hash'].values:
                    print("Dataframe already contains an image with this hash value: {}".format(new_row))
                    continue

                image_df.loc[row_num] = new_row
                row_num += 1
