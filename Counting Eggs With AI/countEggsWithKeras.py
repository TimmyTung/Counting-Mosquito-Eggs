import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from PIL import Image, ImageOps
import difflib
import sys
import os
import math

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

model = load_model('keras_model.h5',compile=False)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
size = (224, 224)

font = cv2.FONT_HERSHEY_PLAIN
boxSizeChange = 0
blankImage = np.zeros((224, 224, 3), np.uint8)
blankImage[:, :] = (255, 0, 255)
l_img = blankImage.copy()
lower_black = np.array([0, 0, 0], dtype="uint16")
upper_black = np.array([85, 85, 85], dtype="uint16")
count = 0

for arg in range(1, len(sys.argv)):
  #print("Position: ", arg)
  #print("Getting image", sys.argv[arg])
  theImage = str(sys.argv[arg])
  # Reading image
  img2 = cv2.imread(theImage, cv2.IMREAD_COLOR)
  img3 = cv2.imread(theImage,cv2.IMREAD_COLOR)
  img = cv2.imread(theImage, cv2.IMREAD_GRAYSCALE)
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
  # assertion: average area is median of array
  EGG_AREA = np.median(contArray)/2

  count = 0
  eggCount = 0
  for c in contours :
    x,y,w,h = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    #print(area,EGG_AREA)
    if(area < EGG_AREA*4 and area > EGG_AREA):
      cropped_image = img2[y - boxSizeChange:y + h + boxSizeChange, x - boxSizeChange:x + w + boxSizeChange]
      if cropped_image is not None:
        count += 1
        ratio = 1
        dim = (1, 1)
        if cropped_image.shape[0] > cropped_image.shape[1]:
          ratio = 224 / cropped_image.shape[0]
          dim = (int(cropped_image.shape[1] * ratio), 224)
        else:
          ratio = 224 / cropped_image.shape[1]
          dim = (224, int(cropped_image.shape[0] * ratio))
        
        resizedImg = cv2.resize(cropped_image, dim, interpolation=cv2.INTER_AREA)
        l_img[0:resizedImg.shape[0], 0:resizedImg.shape[1]] = resizedImg
        testingAIImage = cv2.cvtColor(l_img, cv2.COLOR_BGR2RGB)
        testingAIImage = Image.fromarray(testingAIImage)
        image_array = np.asarray(testingAIImage)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)
        if prediction[0,0] > prediction[0,1] and prediction[0,0] > prediction[0,2]:
          #print("Single Egg")
          eggCount += 1
          cv2.putText(img3,"OneEgg"+ str(count),(x, y),font, 1, (255, 0, 0))
        elif prediction[0,1] > prediction[0,0] and prediction[0,1] > prediction[0,2]:
          #print("Two Eggs")
          eggCount += 2
          cv2.putText(img3,"TwoEgg" + str(count),(x, y),font, 1, (0, 255, 0))
        elif prediction[0,2] > prediction[0,0] and prediction[0,2] > prediction[0,1]:
          #print("Three Eggs")
          eggCount += 3
          cv2.putText(img3,"ThreeEgg" + str(count),(x, y),font, 1, (0, 0, 255))
        l_img = blankImage.copy()
        cv2.rectangle(img3, (x - boxSizeChange, y - boxSizeChange), (x + w + boxSizeChange, y + h + boxSizeChange), (0, 255, 0), 2)
    else:
      eggCount += math.floor(area/(EGG_AREA*2))
      num = math.floor(area/(EGG_AREA*2))
      cv2.putText(img3,str(num),(math.floor(x+w/2), math.floor(y+h/2)),font, 2, (255, 0, 0))
      cv2.rectangle(img3, (x - boxSizeChange, y - boxSizeChange), (x + w + boxSizeChange, y + h + boxSizeChange), (255, 0, 0), 2)
  cv2.imwrite("Rectangle" + theImage,img3)
  print("Total Eggs: " + str(eggCount))
  eggCount = 0
    
cv2.waitKey(0)
cv2.destroyAllWindows()

