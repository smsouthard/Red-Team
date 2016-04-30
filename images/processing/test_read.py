import numpy as np
from PIL import Image
import pandas as pd


driver_list = pd.read_csv('/home/smsouthard/Data/train/driver_imgs_list.csv')

mypath = '/home/smsouthard/Data/train/'

driver_list['path'] = mypath \
                      + driver_list['classname'] \
                      + '/' \
                      + driver_list['img']



image_list = []

for link in driver_list['path']:
    image_list.append(np.asarray(Image.open(link).convert('L')).flatten())

image_list = np.stack(image_list)
