from PIL import Image
#Hash images using the ImageHash library
#https://pypi.python.org/pypi/ImageHash
import imagehash
import os
from collections import defaultdict



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


def remove_duplicate_images(image_dict):
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
    for hash_val, image_paths in image_dict.items():
        if len(image_paths) > 1:
            for image_path in image_paths[1:]:
                os.remove(image_path)

# #Based Yichen Qiu's code:
# #from https://github.com/YichenQiu/Interior-Design-Style-Classifier/blob/master/duplicate_image/dedup_image.py
# def dedup_image(directories):
#     for label in directories[:1]:
#         d=defaultdict(list)
#         for image in os.listdir('Interior-Design-Style-Classifier/{}'.format(label)):
#             im=Image.open('Interior-Design-Style-Classifier/{}/{}'.format(label,image))
#             h=str(imagehash.dhash(im))
#             d[h]+=[image]
#         lst=[]
#         for k,v in d.items():
#             if len(v)>1:
#                 lst.append(list(v))
#         for item in lst:
#             for image in item [1:]:
#                 os.unlink("Interior-Design-Style-Classifier/{}/{}".format(label,image))

if __name__=="__main__":
    #directories=['Bohemian','Coastal','Industrial','Scandinavian']
    #Ran this on Thurs 1/25 around 8:20pm
    #remove_duplicate_images('tree_photos/pseudotsuga menziesii/')
    base_directory = 'tree_photos/'
    subdirectories = ['thuja_plicata', 'acer_macrophylum']
    directories = [os.path.join(base_directory, subdirectory)
                    for subdirectory in subdirectories]
    for directory in directories:
        image_dict = image_hashes_to_paths(directory)
        remove_duplicate_images(image_dict)
