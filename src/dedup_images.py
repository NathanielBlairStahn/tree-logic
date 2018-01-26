from PIL import Image
import imagehash
import os
from collections import defaultdict


def remove_duplicate_images(directory):
    '''
    Removes duplicate images from the given directory. All duplicate
    images after the first occurrence in the directory are removed.

    INPUT: list of strings (path names)
    OUTPUT: None
    '''
    # Dictionary to map hash values to image files
    image_dict = defaultdict(list)

    #Loop through all images in the directory, and make lists of
    #images that have the same hash
    for image_filename in os.listdir(directory):
        image_path = os.path.join(directory, image_filename)
        try:
            img = Image.open(image_path)
            h = str(imagehash.phash(img))
            image_dict[h].append(image_path)
        except OSError:
            print("Cannot open file: {}".format(image_path))


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
    subdirectories = ['acer macrophylum', 'platanus acerifolia']
    directories = [os.path.join(base_directory, subdirectory)
                    for subdirectory in subdirectories]
    for directory in directories:
        remove_duplicate_images(directory)
