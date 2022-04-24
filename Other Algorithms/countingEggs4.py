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
from PIL import Image
import sys
from os.path import exists
#import imutils


def is_contour_bad(c):
    #approximate the contour
    #peri = cv2.arcLength(c, True)
    #approx = cv2.approxPolyDP(c, .02*peri, True)
    area = cv2.contourArea(c);
    #print(area)
    if area > 5000 or area < 150: 
        return True;
    else:
        return False;


def readfile():
    fileO = open("./anco/anco/filenames.txt", "r") #open file
    words = fileO.read().splitlines() #put lines into array
    fileO.close()
    return words
    
def main(filename):
    
    #read textfile name
    words = readfile()
    
    for w in words:
        print("/")
        #read the image
        file_exists = exists("./anco/anco/" + w)
        if file_exists:
           
            image = cv2.imread("./anco/anco/" + w)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            #1500:2750,500:2500
            #int(len(gray)*(.33)):int(len(gray)*(.61)), int(len(gray[0])*(.11)):int(len(gray[0]*.55))
            #gray = gray[int(len(gray[0])*(.25)):int(len(gray[0])*(.25)), int(len(gray)*(.25)):int(len(gray[0])*.25)]
            cv2.imwrite("gray.jpg", gray)
            width = len(gray)
            
            #print(width)
            
            
            ##remove white pixels from gray.jpg
            ##SOURCE/reference: https://answers.opencv.org/question/97416/replace-a-range-of-colors-with-a-specific-color-in-python/
            frame = cv2.imread("gray.jpg")
            #cv2.imshow('frame', frame)
            lower_black = np.array([0,0,0], dtype = "uint16")
            upper_black = np.array([65,65,65], dtype = "uint16")
            black_mask = cv2.inRange(frame,lower_black, upper_black)
            
            #cv2.imshow('black_mask', black_mask);
                
            
            #blur image to avoid detecting noise
            #blur2 = cv2.medianBlur(gray,9) #
            canny = cv2.GaussianBlur(black_mask, (9,9), 0)
            
                
            #blur3 = cv2.medianBlur(blur1, 9)
            #blur = cv2.medianBlur(blur3,9)
            
            #cv2.imshow("blur", blur)
            
            
            #use canny edge detection algo to detect edges
            #input image, minimum threshold val, upper threshold val
            #canny = cv2.Canny(blur,0,90,3);
            
            
            #cv2.imshow("canny", canny)
            
            #connect the edges to fill in the gaps
            #dilated = cv2.dilate(canny,(1,1), iterations=2)
            #erosion_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            #dilated = cv2.erode(canny, erosion_se, iterations=10)
            dilated = cv2.dilate(canny, (1,1), iterations=0)
            #cv2.imshow("dilated",dilated)
            
            #calculate the contours
            (cnt, heirarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            #only draw circle contours 
            rgb = cv2.cvtColor(gray,cv2.COLOR_BGR2RGB)
            count = 0
            for c in cnt:
                if not is_contour_bad(c):
                    peri = cv2.arcLength(c, True)
                    
                    approx = cv2.approxPolyDP(c, .04 * peri, True)
                    cv2.drawContours(rgb, [c], -1, (0,255,0), 3)
                    count += 1
                    
                    
                    #     cv2.imshow("contours", rgb)
            #cv2.imwrite("contours",rgb)
            with open('eggcounts.txt', 'a') as f:
                a = str(count) + "\n"
                f.write(a)
                f.close();
                
                cv2.waitKey(0)
                cv2.destroyAllWindows() 
                
                
if __name__ == "__main__":
    main(sys.argv[1:])