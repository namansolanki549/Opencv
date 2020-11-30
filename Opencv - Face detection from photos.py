#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np


# In[4]:


face_classifier = cv2.CascadeClassifier('Haarcascade/haarcascade_frontalface_default.xml')
image = cv2.imread('image_examples/obama.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(gray, 1.3, 5)
if faces is []:
    print('No faces found')
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (127, 0, 255), 2)
cv2.imshow('Face detection', image)
cv2.waitKey()
cv2.destroyAllWindows()


# In[15]:



eye_classifier = cv2.CascadeClassifier('Haarcascade/haarcascade_eye.xml')
image = cv2.imread('image_examples/alakh.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(gray, 1.3, 5)
if faces is ():
    print('No faces found')
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (127, 0, 255), 2)
    cv2.imshow('Face detection', image)
    cv2.waitKey(0)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    eyes = eye_classifier.detectMultiScale(roi_gray)
    for (ex, ey, eh, ew) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 255, 0), 2)
        cv2.imshow('img', image)
        cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




