import numpy as np
import cv2
import math
import os
font = cv2.FONT_HERSHEY_TRIPLEX

FILENAME = 'stack5.jpg'

# SETUP - CREATE BLACK MASK AND FIND CONTOURS
lower_black = np.array([0, 0, 0], dtype="uint16")
upper_black = np.array([100, 100, 100], dtype="uint16")
img = cv2.imread(FILENAME)
img2 = img
tempImg = cv2.imread(FILENAME)
gray = cv2.cvtColor(tempImg, cv2.COLOR_RGB2GRAY)
cv2.imwrite("gray.jpg", gray)
frame = cv2.imread("gray.jpg")
try: 
    	os.remove("gray.jpg")
except: pass
black_mask = cv2.inRange(frame, lower_black, upper_black)
canny = black_mask
canny = cv2.GaussianBlur(black_mask, (11, 11), 0)
dilated = cv2.dilate(canny, (1, 1), iterations=1)
(contours, heirarchy) = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb, contours, -1, (0, 255, 0), 2)


# NEXT STEP - CALCULATE AREA
# assuming 4350 is the area of one egg, and count is number of eggs
count = 0;
for i in contours:
    area = cv2.contourArea(i)
    x,y,w,h = cv2.boundingRect(i)
    
    if(area > 4350):
        count += math.floor(area/4350)
        num = math.floor(area/4350)
        cv2.putText(rgb,str(num),(math.floor(x+w/2), math.floor(y+h/2)),font, 2, (255, 0, 0))
    elif(area > 3000):
        count += 1
        num = 1
        cv2.putText(rgb,str(num),(math.floor(x+w/2), math.floor(y+h/2)),font, 2, (255, 0, 0))
        
print("Number of eggs: " + str(count));
cv2.imwrite("final.jpg", rgb)
cv2.waitKey(0)
