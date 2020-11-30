#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
import numpy as np
import time


# In[9]:


car_classifier = cv2.CascadeClassifier('Haarcascade/haarcascade_car.xml')
cap = cv2.VideoCapture('image_examples/cars.avi')
while cap.isOpened():
    time.sleep(.05)
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_classifier.detectMultiScale(gray, 1.4, 2)
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.imshow('Cars', frame)
    if cv2.waitKey(1) == 13:
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:




