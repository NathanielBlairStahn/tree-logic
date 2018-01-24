import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image

class ImageClassifier():
    def __init__(self):
        self.cnn = None #tensorflow graph

    def predict_one(image_file):
        '''
        INPUT: an image file object
        OUTPUT: pandas dtataframe with column names equal to labels
        and values equal to predicted probabilities
        '''
        return None

    def predict(image_files):
        '''
        INPUT: numpy array (or list?) of image image files,
        or dataframe containing an image id in one column
        and the filename or file object in another column.
        OUTPUT: pandas dataframe containing
        '''
        return None
