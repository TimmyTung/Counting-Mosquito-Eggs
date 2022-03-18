# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 09:39:24 2022

@author: hanre
"""

import cv2


def count(filePath, blurFlag):
    # read the image
    global dilated
    image = cv2.imread(filePath)
    cv2.imshow('original', image)
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if blurFlag == 'gaussian':
        # blur image to avoid detecting noise
        blur = cv2.GaussianBlur(gray, (11, 11), 0)

        # use canny edge detection algo to detect edges
        # input image, minimum threshold val, upper threshold val
        canny = cv2.Canny(blur, 0, 120, 3);

        # connect the edges to fill in the gaps
        dilated = cv2.dilate(canny,(1,1), iterations=1)

    elif blurFlag == 'median':
        # blur image to avoid detecting noise
        blur2 = cv2.medianBlur(gray, 9)
        blur1 = cv2.medianBlur(blur2, 9)
        blur3 = cv2.medianBlur(blur1, 9)
        blur = cv2.medianBlur(blur3, 9)

        # use canny edge detection algo to detect edges
        # input image, minimum threshold val, upper threshold val
        canny = cv2.Canny(blur, 50, 120, 3)

        # connect the edges to fill in the gaps
        dilated = cv2.dilate(canny,(1,1), iterations=2)

    # calculate the contours
    (cnt, heirarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # convert image to rgb and draw contours onto it
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)

    cv2.imshow("contours", rgb)

    print('Eggs in the image: ', len(cnt))
    cv2.waitKey(0);
    cv2.destroyAllWindows();