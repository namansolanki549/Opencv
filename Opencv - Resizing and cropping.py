#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2


# In[3]:


img = cv2.imread('image_examples/elephant.jpg')
print(img.shape)
imgResize = cv2.resize(img, (778, 519))
print(imgResize.shape)
cv2.imshow('Original', img)
cv2.imshow('Resized', imgResize)


# In[ ]:




