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

for link in driver_list['path'][0:99]:
    image_list.append(np.asarray(Image.open(link).convert('L'), dtype='float32').flatten())

labels = []

for record in driver_list['classname']:
  labels.append(int(record[1:]))


image_list = (np.stack(image_list), np.asarray(labels, dtype='float32'))
