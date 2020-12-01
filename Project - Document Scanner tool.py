#!/usr/bin/env python
# coding: utf-8

# In[29]:


import cv2
import numpy as np


# In[39]:


widthImg=540
heightImg =640
 
img = cv2.imread('image_examples/document.jpg')


def preprocessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations = 2)
    imgThres = cv2.erode(imgDial, kernel, iterations = 1)
    return imgThres

def getWarp(img, biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    image_output = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    return image_output

def  reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    #print('add', add)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis = 1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    #print('myPointsNew', myPointsNew)
    return myPointsNew


def getContours(img):
    biggest, maxArea = np.array([]), 0
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>5000:
            
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 3)
    return biggest

img = cv2.resize(img,(widthImg,heightImg))
imgContour = img.copy()
imgThres = preprocessing(img)
biggest = getContours(imgThres)
warpedImage = getWarp(img, biggest)
cv2.imshow('Image', warpedImage)
cv2.waitKey(0)

cv2.destroyAllWindows()
    


# In[ ]:




