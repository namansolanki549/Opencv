#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np


# In[3]:


face_classifier = cv2.CascadeClassifier('Haarcascade/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('Haarcascade/haarcascade_eye.xml')
def detect(gray, frame):
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_classifier.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return frame
video_capture = cv2.VideoCapture(0)
while video_capture.isOpened():
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) == 13:
        break
video_capture.release()
cv2.destroyAllWindows()


# In[ ]:




