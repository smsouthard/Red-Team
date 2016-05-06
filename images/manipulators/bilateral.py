import cv2
import numpy as np

img = cv2.imread('../scratch/original_3.jpg')
bilateral = cv2.bilateralFilter(img,9,75,75)
cv2.imwrite('../scratch/bilateral_3.jpg',bilateral)

canny_bilateral = cv2.Canny(bilateral,100,200)
cv2.imwrite('../scratch/canny_bilateral_3.jpg',canny_bilateral)
