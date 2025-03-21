#import libraries
import os
import numpy as np
import pandas as pd
from PIL import Image


## import images
images_path = '/home/jacks.local/cpatten/ContrastiveLearning_NBIDE/NBIDE/x3p_dataframe/'
im_names = [f for f in os.listdir(images_path) if f.endswith('.csv')]
images_dict = {}

for im in im_names:
    image = pd.read_csv(os.path.join(images_path, im), encoding="latin_1")
    images_dict[im.replace('.csv','')] = image.pivot(index='y', columns='x', values='value')
    
    
## import info   
info_path = '/home/jacks.local/cpatten/ContrastiveLearning_NBIDE/NBIDE/info.csv'
info = pd.read_csv(info_path, encoding="latin_1")


## transform images to consistant aspect ratio and domain
rng = np.random.default_rng()
images_224 = []
images_444 = []

for spec in info['Specimen']:
    image = images_dict[spec] #grab the image with name 'spec'

    image_min = image.min(axis=None, skipna=True)
    image_max = image.max(axis=None, skipna=True)
    transform_image = (253/(image_max-image_min))*(image-image_min)+1
    transform_image[transform_image.isna()] = 0
    image_numpy = transform_image.to_numpy()

    pillow_224 = Image.fromarray(image_numpy).resize((224,224), resample=Image.Resampling.NEAREST) #convert to 224x224 image
    pillow_444 = Image.fromarray(image_numpy).resize((444,444), resample=Image.Resampling.NEAREST) #convert to 444x444 image

    numpy_224 = np.asarray(pillow_224) #convert to numpy
    numpy_444 = np.asarray(pillow_444) #convert to numpy

    normalized_224 = numpy_224/255 #normalize
    normalized_444 = numpy_444/255 #normalize

    images_224.append(np.asarray(normalized_224)) #add to list of images
    images_444.append(np.asarray(normalized_444)) #add to list of images

images_224 = np.asarray(images_224)
images_444 = np.asarray(images_444)


## save images
np.save('/home/jacks.local/cpatten/ContrastiveLearning_NBIDE/images/processed/images_224.npy', images_224)
np.save('/home/jacks.local/cpatten/ContrastiveLearning_NBIDE/images/processed/images_444.npy', images_444)