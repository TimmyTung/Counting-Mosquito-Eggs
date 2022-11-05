import numpy as np
import cv2
import math
import os
import sys
import math

# function that rejects outliers from a numpy array and returns that array without outliers
# obtained from stack overflow: https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

font = cv2.FONT_HERSHEY_SIMPLEX
thickness = 7
color = (0,255,0)
displayBoundingBoxes = False
imgStart = 2
# for every file name in command line, find the number of eggs in the image
if ((str(sys.argv[1]) != "--stack") and (str(sys.argv[1]) != "--sheet") and (str(sys.argv[1]) != "--boxes")):
    print("Invalid command: Program requires -–stack or –-sheet flag in order to run.  Please see README.txt for more information.")
    exit(0);

if (str(sys.argv[1]) == "--boxes"):
    print("Invalid command: Program requires -–stack or -–sheet flag followed by optional -–boxes flag to run. Please see README.txt for more information.")
    exit(0);

if str(sys.argv[2]) == "--boxes": 
    imgStart = 3
    displayBoundingBoxes = True

for arg in range(imgStart, len(sys.argv)):
    # ensure file name exists and is imported corectly
    try:
        FILENAME = str(sys.argv[arg])
        img = cv2.imread(FILENAME)
        lower_black = np.array([0, 0, 0], dtype="uint16")
        upper_black = np.array([100, 100, 100], dtype="uint16")
        img2 = img
        tempImg = cv2.imread(FILENAME)
        gray = cv2.cvtColor(tempImg, cv2.COLOR_RGB2GRAY)
    except: 
        print("Invalid file name: " + str(sys.argv[arg]) + ", aborting program\n")
        exit(0)
    
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
    #cv2.drawContours(rgb, contours, -1, (0, 255, 0), 2)
    #cv2.imshow("final", rgb)
    #cv2.waitKey(0)
    if len(contours) == 0:
         print("No eggs found, please run another image.");
         exit(0)

    if str(sys.argv[1]) == "--sheet": #SHEET IMAGE ALGO
        # calculate average area of single egg
        contArray = []

        # get array of areas
        for i in contours:
            area = math.floor(cv2.contourArea(i))
            contArray.append(area)
        contArray.sort()
        contArray = np.array(contArray)
        contArray = reject_outliers(contArray)
        contArray = reject_outliers(contArray)
        contArray = reject_outliers(contArray)

	# assertion: average area is median of array without outliers
        EGG_AREA = np.median(contArray)

        # NEXT STEP - CALCULATE COUNT
        # assuming EGG_AREA is the area of one egg, and count is number of eggs
        count = 0;
        offset = 0.18;
        if EGG_AREA > 500 and EGG_AREA < 600:
            offset = .33;
        elif EGG_AREA > 400 and EGG_AREA < 450:
            offset = .27;
        elif EGG_AREA > 450 and EGG_AREA < 500:
            offset = .18;
        elif EGG_AREA < 400 and EGG_AREA > 300:
            offset = .335;
        elif EGG_AREA < 300 and EGG_AREA > 200:
            offset = .4     #untested
        elif EGG_AREA < 200 and EGG_AREA > 100:
            offset = .49    #untested
        elif EGG_AREA < 100:
            offset = .578
        elif EGG_AREA >= 1000:
            offset = .17
        #print(EGG_AREA)
        #print(offset)
        for i in contours:
            area = cv2.contourArea(i)
            x,y,w,h = cv2.boundingRect(i)
            if(area > EGG_AREA+(EGG_AREA*.4)):
                count += math.floor(area/(EGG_AREA-(EGG_AREA*offset)))
                if displayBoundingBoxes == True:
                    num = math.floor(area/(EGG_AREA-(EGG_AREA*offset)))
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(img,str(num),(math.floor(x+w/2)-4*thickness, math.floor(y+h/2)+4*thickness),font, 4, color, thickness)
            elif(area > EGG_AREA-(EGG_AREA*.4)):
                if displayBoundingBoxes == True:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                count += 1


    #STACK IMAGE ALGO
    elif str(sys.argv[1]) == "--stack":
         # NEXT STEP - CALCULATE AREA
        # assuming 4350 is the area of one egg, and count is number of eggs
        count = 0;
        for i in contours:
            area = cv2.contourArea(i)
            x,y,w,h = cv2.boundingRect(i)
            if(area > 4350):
                count += math.floor(area/4350)
                if displayBoundingBoxes == True:
                    num = math.floor(area/4350)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(img,str(num),(math.floor(x+w/2-thickness)-4*thickness, math.floor(y+h/2-thickness)+4*thickness),font, 4, color, thickness)
                elif(area > 3000):
                    if displayBoundingBoxes == True:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    count += 1
        
    print("Number of eggs in file " + str(sys.argv[arg]) + ": " + str(count));
    if displayBoundingBoxes == True:
        cv2.imwrite(str(sys.argv[arg]).split('.')[0] + "WithBoundingBoxes.jpg", img)
        

