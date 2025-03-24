#import libraries
import os
import numpy as np
import pandas as pd
from PIL import Image


## import images
images_path = 'path_to_images'
im_names = [f for f in os.listdir(images_path) if f.endswith('.csv')]
images_dict = {}

for im in im_names:
    image = pd.read_csv(os.path.join(images_path, im), encoding="latin_1")
    images_dict[im.replace('.csv','')] = image.pivot(index='y', columns='x', values='value')
    
    
## import info   
info_path = 'metadata_path'
info = pd.read_csv(info_path, encoding="latin_1")


## transform images to consistant aspect ratio and domain
rng = np.random.default_rng()
images_224 = []

for spec in info['Specimen']:
    image = images_dict[spec] #grab the image with name 'spec'

    image_min = image.min(axis=None, skipna=True)
    image_max = image.max(axis=None, skipna=True)
    transform_image = (253/(image_max-image_min))*(image-image_min)+1
    transform_image[transform_image.isna()] = 0
    image_numpy = transform_image.to_numpy()

    pillow_224 = Image.fromarray(image_numpy).resize((224,224), resample=Image.Resampling.NEAREST) #convert to 224x224 image

    numpy_224 = np.asarray(pillow_224) #convert to numpy

    normalized_224 = numpy_224/255 #normalize

    images_224.append(np.asarray(normalized_224)) #add to list of images

images_224 = np.asarray(images_224)

processed_224 = images_224

r0 = 50
r1 = 110
fit_points = [np.linspace(-111.5, 111.5, 224), np.linspace(-111.5, 111.5, 224)]
ut, vt = np.meshgrid(np.linspace(-np.pi, np.pi, np.rint(2*np.pi*(r1-r0)).astype(int)),
                     np.linspace(r0, r1, r1-r0), indexing='ij')
test_points_pol = np.array([ut.ravel(), vt.ravel()]).T
test_points_euc = np.array([ut.ravel(), vt.ravel()]).T
test_points_euc[:,0] = test_points_pol[:,1] * np.cos(test_points_pol[:,0])
test_points_euc[:,1] = test_points_pol[:,1] * np.sin(test_points_pol[:,0])
test_points = test_points_euc
test_points.shape
    
pro_cyls = np.zeros((144,377,60))
for cart in range(144):
    pro_cyls[cart] = interpn(fit_points, processed_224[cart], test_points, method="slinear").reshape(377, 60)
    print(cart)


## save images
save_path = 'polar_images_path'
np.save(save_path, pro_cyls)
