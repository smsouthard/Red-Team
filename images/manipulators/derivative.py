import cv2
import numpy as np

img = cv2.imread('c0/img_7987.jpg',1)

laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

cv2.imwrite('scratch/laplacian.jpg',laplacian)
cv2.imwrite('scratch/sobelx.jpg',sobelx)
cv2.imwrite('scratch/sobely.jpg',sobely)
