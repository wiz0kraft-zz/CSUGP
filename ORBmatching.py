#!/usr/bin/env python


import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('/home/wizkraft/Desktop/Faces/Logan/2.png')
img1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

img2 = cv2.imread('/home/wizkraft/Desktop/Faces/Logan/16.png')
img2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)


# Initiate ORB detector
orb = cv2.ORB_create()


# compute the descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
print(len(matches))

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], None,flags=2)
plt.imshow(img3),plt.show()