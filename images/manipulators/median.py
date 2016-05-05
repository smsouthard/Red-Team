import cv2
import numpy as np

img = cv2.imread('c0/img_7987.jpg')
median = cv2.medianBlur(img,5)
cv2.imwrite('../scratch/median.jpg',median)

canny_median = cv2.Canny(median,100,200)
cv2.imwrite('../scratch/canny_median.jpg',canny_median)
