import numpy as np
import pandas as pd

import os

from PIL import Image
import imagehash

#import datetime

#from keras.preprocessing import image

#from collections import defaultdict

def swap_list_elements(lst, i, j):
    temp = lst[i]
    lst[i] = lst[j]
    lst[j] = temp

def export_df(df, filepath):
    df.to_csv(filepath, sep='|')

def load_df(filepath):
    return pd.read_csv(filepath, sep='|', index_col=0)

class ImageManager():
    # IMAGES_PATH = "image_log.csv"
    # SYNCS_PATH = "syncs_log.csv"
    def __init__(self, base_directory,
                 images_path = None,
                 syncs_path = None,
                 hash_fcn = imagehash.phash,
                 hash_col = 'p_hash',
                 file_col = 'filename',
                 folder_col = 'folder'):
        self.image_extensions = ['.png', '.jpg', '.jpeg']
        self.base_directory = base_directory #'tree_photos'
        self.hash_fcn = hash_fcn

        #self.target_size = (299,299)

        #Define the columns in the DataFrame
        self.folder_col = folder_col
        self.file_col = file_col
        self.hash_col = hash_col
        self.time_added_col = 'time_added'
        self.time_verified_col = 'time_verified'

        columns = [self.hash_col, self.file_col, self.folder_col,
                    self.time_added_col, self.time_verified_col]

        #self.synced = False

        # #Initialize the image dictionary
        # if image_dict is None:
        #     self.image_dict = defaultdict(list)
        # else:
        #     self.image_dict = image_dict
        #self.image_dict = defaultdict(list)

        #Initialize the image DataFrame
        if images_path is None:
            self.image_df = pd.DataFrame(columns=columns)
            self.image_dict = {}
        else:
            self.load_image_df(images_path)
            self._sync_dict_with_df() #This initializes self.image_dict

        # if len(self.image_dict) == 0 and len(self.image_df) == 0:
        #     self.synced = True
        # else:
        #     self.synced = False

        #DataFrame to keep track of when files are synced
        if syncs_path is None:
            self.syncs_df = pd.DataFrame(columns = ['time_started', 'time_completed', 'folders'])
        else:
            self.load_syncs_df(syncs_path)

    #Yak!!!

    def export_logs(self, images_filename, syncs_filename):
        # self.image_df.to_csv(images_filename, sep='|')
        # self.syncs_df.to_csv(syncs_filename, sep='|')
        export_df(self.image_df, images_filename)
        export_df(self.syncs_df, syncs_filename)

    def load_image_df(self, path):
        # image_df = pd.read_csv(path, sep='|', index_col=0)
        image_df = load_df(path)
        image_df.loc[:,self.hash_col] = (
            image_df[self.hash_col].apply(imagehash.hex_to_hash)
            )
        image_df[self.time_added_col] = pd.to_datetime(
            image_df[self.time_added_col], unit='ns'
            )
        image_df[self.time_verified_col] = pd.to_datetime(
            image_df[self.time_verified_col], unit='ns'
            )

        self.image_df = image_df

    def load_syncs_df(self, path):
        syncs_df = pd.read_csv(path, sep='|', index_col=0)
        syncs_df['time_started'] = pd.to_datetime(
            syncs_df['time_started'], unit='ns'
            )
        syncs_df['time_completed'] = pd.to_datetime(
            syncs_df['time_completed'], unit='ns'
            )
        # syncs_df.loc[:,['time_started', 'time_completed']] = (
        #     syncs_df.loc[:,['time_started', 'time_completed']]
        #     .applymap(pd.Timestamp)
        # )
        # (syncs_df.loc[:,'folders']
        #     = syncs_df.loc[:,'folders']
        #     .apply(eval)
        # )
        self.syncs_df = syncs_df

    def knows_image(self, PIL_image):
        '''
        Checks whether a PIL Image has a hash that matches one we've
        already seen, as recorded in image_dict.

        INPUT: PIL_image - a PIL Image object
        RETURNS: True of the hash of the image is in the dictionary,
            False otherwise.
        '''
        return self.hash_fcn(PIL_image) in self.image_dict

    def get_duplicates(self):
        '''
        Returns the subdictionary of self.image_dict with hashes
        that have more than one path listed.
        '''
        return {im_hash: paths for im_hash, paths
                in self.image_dict.items() if len(paths) > 1}

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
                    if os.path.exists(image_path):
                        # print("OMG! You were about to remove {image_path}!")
                        os.remove(image_path)
                #Keep only first file in the dictionary
                self.image_dict[hash_val] = image_paths[0:1]

    def _sync_dict_with_df(self):
        '''
        Internal method to ensure that the dictionary contains the same
        hashes and paths as the DataFrame.
        It's possible that if I do everything right, I won't actually
        need to call this.
        '''
        self.image_dict = {hash_val:
             [os.path.join(self.base_directory, directory, filename)]
             for hash_val, directory, filename
             in zip(self.image_df[self.hash_col],
                    self.image_df[self.folder_col],
                    self.image_df[self.file_col]
                    )
            }

    def sync_images(self, subdirectories=None):
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

        #Get the current length of the dataframe -- we'll be adding rows to the end,
        #one for each new image file found
        row_num = len(self.image_df)

        #Get the number of the current sync
        sync_num = len(self.syncs_df)

        #Record the time we start syncing
        time_started = pd.Timestamp.now()

        #Iterate through the species names, which are assumed to be subfolders of the base directory.
        for subdirectory in subdirectories:

            #Get the full path to the subdirectory
            folder_path = os.path.join(self.base_directory, subdirectory)

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

                    #If the hash value was NOT in the dictionary, we know we have found
                    #a new image, so add the path to the dictionary, and add a new row
                    #to the dataframe for this hash.

                    #New hash with new path
                    self.image_dict[hash_val] = [dir_entry.path]

                    #Create a new row containing the data for the current file
                    timestamp = pd.Timestamp.now()
                    new_row = pd.Series({self.hash_col: hash_val
                                        , self.file_col: dir_entry.name
                                        , self.folder_col: subdirectory
                                        , self.time_added_col: timestamp
                                        , self.time_verified_col: timestamp
                                        })
                    self.image_df.loc[row_num] = new_row
                    row_num += 1

        #After the for loop completes, we know we have accounted for every image
        #in the given subdirectories. However, some of the paths listed in the
        #DataFrame may still be incorrect, e.g. if an image was moved since the
        #last sync. In particular, if an image was moved to a different path in
        #the subdirectories we checked, we will have found the new path and stored
        #it in the dictionary, but we still need to re-order the paths in the
        #dictionary and update the path in the DataFrame. We need to find all
        #images that need such an update, i.e.:
        #
        #Find all hashes with more than one path listed in the dictionary but
        #that have not been verified during this sync. We need to update the
        #paths for these hashes because:
        #
        # 1) Having more than one filepath listed means we found at least one valid
        #  path during this sync that did not match the path in the DataFrame.
        #
        # 2) Being unverified during this sync means that none of the paths found
        #  in this sync matched the path in the Dataframe.
        #
        #Therefore, we know we have at least one valid path for the hash, but it
        #does not match what was in the DataFrame, so we should update the path
        #to one we know is correct.

        #Find hashes with multiple paths:
        duplicates = set(self.get_duplicates().keys())
        #Find hashes whose paths were unverified during this sync:
        #The path was unverified during this sync if its recorded verification
        #timestamp is earlier than the time when we started this sync.
        unverified = self.image_df.loc[
            self.image_df[self.time_verified_col] < time_started, self.hash_col]
        #Take the intersection to find the hashes whose paths we need to update:
        hashes_to_update = duplicates.intersection(unverified)

        #To update the paths, we need to:
        #1) Swap the 0th and 1st elments of self.image_dict[hash_val].
        #2) Replace the folder and filename fields for the hash in the DataFrame.
        #3) Update the verification time for the hash in the DataFrame.

        for h in hashes_to_update:
            #Swap the 0th filepath with the 1st
            swap_list_elements(self.image_dict[h],0,1)
            #Get the folder path and filename from the new path
            folder_path, filename = os.path.split(self.image_dict[h][0])
            #Strip off the base directory to get the folder name
            folder_name = os.path.basename(folder_path)
            #Update the folder and filename in DataFrame
            #cols_to_update = [self.file_col, ]
            self.image_df.loc[self.image_df[self.hash_col] == h, self.file_col] = filename
            self.image_df.loc[self.image_df[self.hash_col] == h, self.folder_col] = folder_name
            self.image_df.loc[self.image_df[self.hash_col] == h, self.time_verified_col] = pd.Timestamp.now()

        time_completed = pd.Timestamp.now()

        #Record when the sync was started and completed
        self.syncs_df.loc[sync_num, 'time_started'] = time_started
        self.syncs_df.loc[sync_num, 'time_completed'] = time_completed
        self.syncs_df.loc[sync_num, 'folders'] = repr(subdirectories)

        #return (time_started, time_completed)

    def _get_folder_and_filename(image_path):
        #Get the folder path and filename from the path
        folder_path, filename = os.path.split(image_path)
        #Strip off the base directory to get the folder name
        folder_name = os.path.basename(folder_path)
        return folder_name, filename

    # def update_path(self, image_hash, image_path):
    #     timestamp = pd.Timestamp.now()
    #     if image_hash in self.image_dict:
    #         #Move current path to the end; we'll replace it with the new one
    #         self.image_dict[image_hash].append(self.image_dict[image_hash][0])
    #     else:
    #         #Create a new path list if the hash doesn't already exist
    #         self.image_dict = list()
    #         #Record the time when
    #         self.image_df.loc[self.image_df[self.hash_col] == image_hash, self.time_added_col] = timestamp
    #     #Update the dictionary with the new path
    #     self.image_dict[image_hash][0] = image_path
    #
    #     #Get the folder name and filename from the path
    #     folder_name, filename = _get_folder_and_filename(image_path)
    #     timestamp = pd.Timestamp.now()
    #
    #     pd.Series({self.folder_col: folder_name,
    #                self.file_col: filename})
    #
    #     self.image_df.loc[
    #         self.image_df[self.hash_col] == image_hash,
    #         [self.folder_col, self.file_col]] = [folder_name, filename]


    #Copied from build_a_model.ipynb, won't work yet
    def get_image_paths(image_df, base_directory, indices=None):
        if indices is None:
            indices = image_df.index

        directories = image_df.loc[indices, 'species']
        filenames = image_df.loc[indices,'filename']

        img_paths = [os.path.join(base_directory, directory, filename)
                    for directory, filename in zip(directories, filenames)]

        return img_paths

    #Copied from build_a_model.ipynb, may or may not work...
    def generate_images_from_paths(self, img_paths):
        #return (image.load_img(path, target_size=(299,299)) for path in img_paths)
        return (Image.open(path) for path in img_paths)
