from keras.applications.inception_v3 import InceptionV3

from PIL import Image

class ImageClassifier():
    def __init__(self, feature_extractor=None, predictor=None):
        if feature_extractor is None:
            feature_extractor = InceptionV3(include_top=False
                                            , weights='imagenet'
                                            , pooling='avg')
        self.feature_extractor = feature_extractor
        self.predictor = predictor

    def extract_features(self, image_df):
        pass

    def predict(self, image_files):
        '''
        INPUT: pandas dataframe containing an image id in one column
        and the filename or file object in another column.

        OUTPUT: pandas dataframe containing column names equal to labels
        and values equal to predicted probabilities
        '''
        return None
