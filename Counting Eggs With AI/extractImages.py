import difflib
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import math

font = cv2.FONT_HERSHEY_PLAIN
boxSizeChange = 0
blankImage = np.zeros((224, 224, 3), np.uint8)
blankImage[:, :] = (255, 0, 255)
l_img = blankImage.copy()
lower_black = np.array([0, 0, 0], dtype="uint16")
upper_black = np.array([85, 85, 85], dtype="uint16")
count = 0

#print(len(sys.argv))
#print(str(sys.argv))

for arg in range(1, len(sys.argv)):
    #print("Position: ", arg)
    #print("Getting image", sys.argv[arg])
    theImage = str(sys.argv[arg])
    # Reading image

    img2 = cv2.imread(theImage, cv2.IMREAD_COLOR)
    img3 = cv2.imread(theImage, cv2.IMREAD_COLOR)
    tempImg = cv2.imread(theImage)
    grey = cv2.cvtColor(tempImg, cv2.COLOR_RGB2GRAY)
    cv2.imwrite("grey.jpg", grey)
    frame = cv2.imread("grey.jpg")
    try: 
    	os.remove("grey.jpg")
    except: pass
    black_mask = cv2.inRange(frame, lower_black, upper_black)
    canny = cv2.GaussianBlur(black_mask, (11, 11), 0)
    dilated = cv2.dilate(canny, (1, 1), iterations=1)
    (contours, heirarchy) = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    cv2.drawContours(rgb, contours, -1, (0, 255, 0), 2)
    cv2.imwrite("contours.jpg", rgb)
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cropped_image = img2[y - boxSizeChange:y + h + boxSizeChange, x - boxSizeChange:x + w + boxSizeChange]
        if cropped_image is not None:
            count += 1
            changedNum = str(count)
            ratio = 1
            dim = (1, 1)
            area = cropped_image.shape[0] * cropped_image.shape[1]
            #print(area,count)
            if 1500 <= area:
                if cropped_image.shape[0] > cropped_image.shape[1]:
                    ratio = 224 / cropped_image.shape[0]
                    dim = (int(cropped_image.shape[1] * ratio), 224)
                else:
                    ratio = 224 / cropped_image.shape[1]
                    dim = (224, int(cropped_image.shape[0] * ratio))
                
                resizedImg = cv2.resize(cropped_image, dim, interpolation=cv2.INTER_AREA)
                l_img[0:resizedImg.shape[0], 0:resizedImg.shape[1]] = resizedImg
                l_img = cv2.cvtColor(l_img,cv2.COLOR_RGB2GRAY)
                if not os.path.exists(os.path.splitext(sys.argv[arg])[0] + 'Images'):
                    os.makedirs(os.path.splitext(sys.argv[arg])[0]+'Images')
                cv2.imwrite(os.path.splitext(sys.argv[arg])[0] + 'Images/Cropped Image' + changedNum + '.jpg', l_img)
                l_img90 = cv2.rotate(l_img,cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite(os.path.splitext(sys.argv[arg])[0] + 'Images/Cropped Image90' + changedNum + '.jpg', l_img90)
                l_img180 = cv2.rotate(l_img,cv2.ROTATE_180)
                cv2.imwrite(os.path.splitext(sys.argv[arg])[0] + 'Images/Cropped Image180' + changedNum + '.jpg', l_img180)
                l_img270 = cv2.rotate(l_img,cv2.ROTATE_90_COUNTERCLOCKWISE)
                cv2.imwrite(os.path.splitext(sys.argv[arg])[0] + 'Images/Cropped Image270' + changedNum + '.jpg', l_img270)
                l_imgFlipped = cv2.flip(l_img,0)
                cv2.imwrite(os.path.splitext(sys.argv[arg])[0] + 'Images/Cropped ImageFlipped' + changedNum + '.jpg', l_imgFlipped)
                l_img = blankImage.copy()
                cv2.rectangle(img3, (x - boxSizeChange, y - boxSizeChange), (x + w + boxSizeChange, y + h + boxSizeChange),(0, 255, 0), 2)
                cv2.putText(img3, changedNum, (x, y), font, 1, (255, 0, 0))

cv2.imwrite("Rectangle" + theImage, img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
