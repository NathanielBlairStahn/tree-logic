import numpy as np
import pandas as pd

import os

from PIL import Image
import imagehash

#import datetime

#from keras.preprocessing import image

#from collections import defaultdict

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
        self.time_added_col = 'time_added'
        self.time_verified_col = 'time_verified'
        columns = [self.hash_col, self.file_col, self.label_col,
                    self.time_added_col, self.time_verified_col]

        #DataFrame to keep track of when files are synced
        self.syncs_df = pd.DataFrame(columns = ['time_started', 'time_completed'])

        #self.synced = False

        # #Initialize the image dictionary
        # if image_dict is None:
        #     self.image_dict = defaultdict(list)
        # else:
        #     self.image_dict = image_dict
        #self.image_dict = defaultdict(list)

        #Initialize the image DataFrame
        if image_df is None:
            self.image_df = pd.DataFrame(columns=columns)
            self.image_dict = {}
        else:
            self.image_df = image_df
            self._sync_dict_with_df() #This initializes self.image_dict

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

    def _update_image_dict(self, subdirectories=None):
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

    # def sync_images(self, subdirectories):
    #     self._update_image_dict(subdirectories)

    def _sync_dict_with_df():
        self.image_dict = {hash_val:
             os.path.join(base_directory, directory, filename)
             for hash_val, directory, filename
             in zip(self.image_df[self.hash_col],
                    self.image_df[self.label_col],
                    self.image_df[self.file_col]
                    )
            }

    def sync_images(self, subdirectories):
        '''
        Syncs the specified subdirectories of self.base_directory into the
        dataframe and dictionary.

        INPUT: subdirectories (list of strings) - list of subdirectories of
        base_directory to sync into the dataframe.
        RETURNS: (time_started, time_completed) - tuple of pd.Timestamps (?)
        reperesenting when the syncing started and when it stopped.

        When this function completes, the following are guaranteed:

        1) Every unique image in the given subdirectories will have its hash
        contained in the DataFrame, with its path being a valid path to a file
        in one of the listed subdirectories.
        2) The dictionary will contain every image hash listed in the DataFrame,
        and the path listed in the DataFrame for that hash will be the first
        path listed for the hash in the dictionary.
        3) If multiple files in the listed subdirectories have the same image
        hash, the paths to all of these files will be listed in the dictionary
        for that hash; an arbitrary one of these paths will be listed first and
        will math the path listed in the DataFrame.
        4) If there was an image hash in the DataFrame (before sync_images was
        called) that matched the hash of an image in the given subdirectories,
        the path for that hash will be updated to match a valid path in the
        given subdirectories. The old file path will be listed in the dictionary
        for that hash, in a position after the first.
        5) Any image hash in the DataFrame that does not match an image in the
        given subdirectories will have the same path listed as before sync_images
        was called.
        '''
        self._sync_dict_with_df()

        #If no subdirectories are passed, use all subfolders of the base folder as species names
        if subdirectories is None:
            with os.scandir(self.base_directory) as base_contents:
                subdirectories = [entry.name for entry in base_contents if entry.is_dir()]

        #Get the current length of the dataframe -- we'll be adding rows to the end, one for each image file found
        row_num = len(self.image_df)

        #Get the number of the current sync
        sync_num = len(self.syncs_df)

        #Record the time we start syncing
        time_started = pd.Timestamp.now()

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
                        # img = image.load_img(dir_entry.path)#, target_size=self.target_size)

                        # It shouldn't matter whether we open the image with Keras or PIL,
                        # or whether we resize the image on opening
                        img = Image.open(dir_entry.path)
                    except OSError:
                        #Print error and go to next file
                        print("Cannot open file: {}".format(dir_entry.path))
                        continue


                    #Refer to https://pypi.python.org/pypi/ImageHash for imagehash documentation.
                    #imagehash.phash(img) computes the perception hash of the image (http://www.phash.org/).
                    #Note that this hash SHOULDN'T change if the image is loaded in with a target_size
                    #different from (299,299), but it's probably best to always use the same target size
                    #just to be safe.
                    hash_val = self.hash_fcn(img)

                    #Check if the dictionary contains this hash value.
                    #If so, the image has already been added to the dataframe,
                    #and the first path listed in the dictionary is the one listed
                    #in the dataframe.
                    if hash_val in self.image_dict:
                        #If the current path matches the first listed (which is guaranteed
                        #to be the same as the path in the DataFrame), mark that we verified
                        #the path at the current time.
                        if dir_entry.path == self.image_dict[hash_val][0]:
                            self.image_df.loc[self.image_df[self.hash_col] == hash_val,
                                              self.time_verified_col] = pd.Timestamp.now()
                        else:
                            #If the current filepath is different from the first one listed,
                            #we have found a duplicate, so add the new path.
                            self.image_dict[hash_val].append(dir_entry.path)

                        #At this point, we can move on to the next file, because
                        #we know the image is already in the dataframe, and we
                        #have recorded the new path if it was a duplicate
                        continue

                    #At this point, we know we have found a new image, so add a
                    #new row to the dataframe recording its location.

                    timestamp = pd.Timestamp.now()
                    #Create a new row containing the data for the current file
                    new_row = pd.Series({self.hash_col: hash_val
                                        , self.file_col: dir_entry.name
                                        , self.label_col: subdirectory
                                        , self.time_added_col: timestamp
                                        , self.time_verified_col: timestamp
                                        })
                    self.image_df.loc[row_num] = new_row
                    row_num += 1

        time_completed = pd.Timestamp.now()

        return (time_started, time_completed)

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
