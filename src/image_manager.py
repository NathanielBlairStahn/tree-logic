import numpy as np
import pandas as pd

import os

from PIL import Image
import imagehash

#from keras.preprocessing import image

from collections import defaultdict

class ImageManager():
    def __init__(self, base_directory,
                 image_df = None):
        self.image_extensions = ['.png', '.jpg', '.jpeg']
        self.base_directory = base_directory #'tree_photos'
        self.hash_fcn = imagehash.phash

        #self.target_size = (299,299)

        #Define the columns in the DataFrame
        self.label_col = 'species'
        self.file_col = 'filename'
        self.hash_col = 'p_hash'
        columns = [self.hash_col, self.file_col, self.label_col]

        # #Initialize the image dictionary
        # if image_dict is None:
        #     self.image_dict = defaultdict(list)
        # else:
        #     self.image_dict = image_dict
        self.image_dict = defaultdict(list)

        #Initialize the image DataFrame
        if image_df is None:
            self.image_df = pd.DataFrame(columns=columns)
        else:
            self.image_df = image_df

        # if len(self.image_dict) == 0 and len(self.image_df) == 0:
        #     self.synced = True
        # else:
        #     self.synced = False

    def knows_image(self, PIL_image):
        '''
        Checks whether a PIL Image has a hash that matches one we've
        already seen, as recorded in image_dict.

        INPUT: PIL_image - a PIL Image object
        RETURNS: True of the hash of the image is in the dictionary,
            False otherwise.
        '''
        return self.hash_fcn(PIL_image) in self.image_dict

    def image_hashes_to_paths(directory,
                              image_dict = defaultdict(list),
                              hash_fcn = imagehash.phash):
        '''
        INPUTS:
            directory (string) - name of the directory containing image files to hash.
            image_dict (ddefaultdict(list)) - a dictionary mapping previously found
            image hashes to a list of their paths. If no dictionary is given, a new
            one is created.

        RETURNS: The updated default dictionary mapping hash values of images in the directory
        to a list of image paths with the same hash.
        '''
        # # Dictionary to map hash values to image files
        # image_dict = defaultdict(list)

        #Loop through all images in the directory, and make lists of
        #images that have the same hash
        for image_filename in os.listdir(directory):
            image_path = os.path.join(directory, image_filename)
            try:
                img = Image.open(image_path)
                #Use 'perception hash' to compute the hash of an image: http://www.phash.org/
                #phash  uses the Discrete Cosine Transform to convert images into
                #frequency space
                h = str(hash_fcn(img))
                image_dict[h].append(image_path)
            except OSError:
                print("Cannot open file: {}".format(image_path))

        return image_dict

    def remove_duplicates(self):
        '''
        Removes duplicate images found in image_dict. All duplicate
        images after the first occurrence in the list of repeats are removed.

        INPUT: image_dict - dictionary of image paths to remove images from.
        The key is the image's hash vlaue
        OUTPUT: None
        '''
        # # Dictionary to map hash values to image files
        # image_dict = create_image_dict(directory)


        #Find all hashes that had more than one image, and remove duplicates
        for hash_val, image_paths in self.image_dict.items():
            if len(image_paths) > 1:
                #Remove any image files after the first
                for image_path in image_paths[1:]:
                    os.remove(image_path)
                #Keep only first file in the dictionary
                self.image_dict[hash_val] = image_paths[0:1]

    def update_image_dict(self, subdirectories=None):
        #If no subdirectories are passed, use all subfolders of the base folder as species names
        if subdirectories is None:
            with os.scandir(self.base_directory) as base_contents:
                subdirectories = [entry.name for entry in base_contents if entry.is_dir()]

        #Iterate through the species names, which are assumed to be subfolders of the base directory.
        for subdirectory in subdirectories:

            #Get the full path to the subdirectory
            folder_path = os.path.join(base_directory, subdirectory)

            #Get all the DirEntry objects in the folder, and add the relevant information to the dataframe
            with os.scandir(folder_path) as dir_entries:
                for dir_entry in dir_entries:
                    #Skip anything that's not an image file
                    if os.path.splitext(dir_entry.name)[1].lower() not in self.image_extensions:
                        continue

                    try:
                        # #Use Keras to load the image file into a PIL image object with the correct target size
                        # #for InceptionV3
                        # img = image.load_img(dir_entry.path, target_size=(299,299))

                        # It shouldn't matter whether we open the image with Keras or PIL,
                        # or whether we resize the image on opening
                        img = Image.open(dir_entry.path)


                        #Refer to https://pypi.python.org/pypi/ImageHash for imagehash documentation.
                        #imagehash.phash(img) computes the perception hash of the image (http://www.phash.org/).
                        #Note that this hash SHOULDN'T change if the image is loaded in with a target_size
                        #different from (299,299), but it's probably best to always use the same target size
                        #just to be safe.
                        hash_val = self.hash_fcn(img)

                        if hash_val in self.image_dict and dir_entry.path in self.image_dict[hash_val]:
                            #We already know about this image, so move to the next file
                            continue
                        else:
                            #Image file is new, so add it's path (and the hash if we don't have it)
                            self.image_dict[hash_val].append(dir_entry.path)
                    except OSError:
                        print("Cannot open file: {}".format(dir_entry.path))


    def sync_images(self, subdirectories):
        self.update_image_dict(subdirectories)

    def add_images_to_df(self,
                         label_col='species',
                         file_col='filename',
                         hash_fcn=('p_hash', imagehash.phash),
                         subdirectories=None,
                         skip_duplicates=False):
    #If no subdirectories are passed, use all subfolders of the base folder as species names
    if subdirectories is None:
        with os.scandir(self.base_directory) as base_contents:
            subdirectories = [entry.name for entry in base_contents if entry.is_dir()]

    #Get the current length of the dataframe -- we'll be adding rows to the end, one for each image file found
    row_num = len(self.image_df)

    #Iterate through the species names, which are assumed to be subfolders of the base directory.
    for subdirectory in subdirectories:

        #Get the full path to the subdirectory
        folder_path = os.path.join(base_directory, subdirectory)

        #Get all the DirEntry objects in the folder, and add the relevant information to the dataframe
        with os.scandir(folder_path) as dir_entries:
            for dir_entry in dir_entries:
                #Skip anything that's not an image file
                if os.path.splitext(dir_entry.name)[1].lower() not in self.image_extensions:
                    continue

                #Use Keras to load the image file into a PIL image object with the correct target size
                #for InceptionV3
                img = image.load_img(dir_entry.path)#, target_size=self.target_size)


                #Refer to https://pypi.python.org/pypi/ImageHash for imagehash documentation.
                #imagehash.phash(img) computes the perception hash of the image (http://www.phash.org/).
                #Note that this hash SHOULDN'T change if the image is loaded in with a target_size
                #different from (299,299), but it's probably best to always use the same target size
                #just to be safe.
                hash_val = self.hash_fcn(img)

                if hash_val in self.image_dict and dir_entry.path in self.image_dict[hash_val]:
                    #
                    pass
                else:
                    self.image_dict[hash_val].append(dir_entry.path)

                #Create a new row containing the data for the current file
                new_row = pd.Series({hash_fcn[0]: hash_val
                                    , file_col: dir_entry.name
                                    , label_col: subdirectory
                                    , 'tags': ''})

                #If skip_duplicates is True, check whether the dataframe already has an image with
                #the same hash, and only add it if not.
                #NOTE: Add an option to update file path for same hash -- in case original was deleted
                if skip_duplicates and p_hash in image_df['p_hash'].values:
                    print("Dataframe already contains an image with this hash value: {}".format(new_row))
                    continue

                image_df.loc[row_num] = new_row
                row_num += 1

    def get_image_paths(image_df, base_directory, indices=None):
        if indices is None:
            indices = image_df.index

        directories = image_df.loc[indices, 'species']
        filenames = image_df.loc[indices,'filename']

        img_paths = [os.path.join(base_directory, directory, filename)
                    for directory, filename in zip(directories, filenames)]

    return img_paths

    def generate_images_from_paths(img_paths):
        return (image.load_img(path, target_size=(299,299)) for path in img_paths)
