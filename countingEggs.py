# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 12:19:35 2022

@author: Emma Hanretty (hanre)
"""

#count number of eggs in "eggs.tif"
#reference: https://www.youtube.com/watch?v=rRcwuQGDFOA
##Note: I used the above youtube video as a guide but modified some of the values
## to get more accurate results for mosquito eggs.  The image is one from the drive 
## that I cropped.  From this program, the best images to analyze are the ones that:
    # - don't have any borders(eg. a circular border)
    # - have minimal background noise 
    # - have distinct contrast between darker and lighter colors
    # - HOWEVER they can still be color because it will get converted to black and white for analysis

import cv2
import numpy as np 
import matplotlib.pyplot as plt

#read the image
image = cv2.imread("eggs.tif")

#convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#blur image to avoid detecting noise
blur2 = cv2.medianBlur(gray,9) #blur = cv2.GaussianBlur(gray, (25,25), 0)
blur1 = cv2.medianBlur(blur2,9)
blur3 = cv2.medianBlur(blur1, 9)
blur = cv2.medianBlur(blur3,9)

cv2.imshow("blur", blur)


#use canny edge detection algo to detect edges
#input image, minimum threshold val, upper threshold val
canny = cv2.Canny(blur,50,120,3);

cv2.imshow("canny", canny)

#connect the edges to fill in the gaps
dilated = cv2.dilate(canny,(1,1), iterations=2)
cv2.imshow("dilated",dilated)

#calculate the contours
(cnt, heirarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
 
#convert image to rgb and draw contours onto it
rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb, cnt, -1, (0,255,0), 2)

cv2.imshow("contours", rgb)

print('Eggs in the image: ', len(cnt))

cv2.waitKey(0)
cv2.destroyAllWindows()