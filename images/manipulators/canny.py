import cv2
import numpy as np

img = cv2.imread('../scratch/original_3.jpg',0)
canny = cv2.Canny(img,100,200)

cv2.imwrite('../scratch/canny_3.jpg',canny)
