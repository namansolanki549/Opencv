#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np


# In[7]:


img = cv2.imread('image_examples/obama.jpg')
kernel = np.ones((5,5), np.uint8)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (7, 7), 0)
img_canny = cv2.Canny(img, 150, 200)
img_dilated = cv2.dilate(img_canny, kernel, iterations = 1)
img_eroded = cv2.erode(img_canny, kernel, iterations = 1)
cv2.imshow('Test elephant image', img)
cv2.imshow('Gray elephant image', img_gray)
cv2.imshow('Blur elephant image', img_blur)
cv2.imshow('Canny elephant image', img_canny)
cv2.imshow('Dilated elephant image', img_dilated)
cv2.imshow('Eroded elephant image', img_eroded)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


imgResize = cv2.resize(img, (300, 500))
print(img.shape)
print(imgResize.shape)
cv2.imshow('Original Image', img)
cv2.imshow('Resized Image', imgResize)


# In[ ]:




