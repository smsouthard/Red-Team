import numpy as np
from PIL import Image
from os import walk
import pandas as pd


driver_list = pd.read_csv('/home/smsouthard/Data/train/driver_imgs_list.csv')

mypath = '/home/smsouthard/Data/train/'

driver_list['path'] = mypath \
                      + driver_list['classname'] \ 
                      + '/' \
                      + driver_list['img']




for

numpy.asarray(Image.open('').convert('L'))


