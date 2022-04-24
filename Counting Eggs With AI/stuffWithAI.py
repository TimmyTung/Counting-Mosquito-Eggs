import difflib
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
import sys
import os

model = load_model('keras_model.h5', compile=False)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
currentArray = []
prevArray = []
boxSizeChange = 5
blankImage = np.zeros((224, 224, 3), np.uint8)
blankImage[:, :] = (255, 0, 255)
l_img = blankImage.copy()
lower_black = np.array([0, 0, 0], dtype="uint16")
upper_black = np.array([70, 70, 70], dtype="uint16")
count = 0
print(len(sys.argv))
print(str(sys.argv))

for arg in range(1, len(sys.argv)):
    eggCount = 0;
    # print("Position: ", arg)
    # print("Getting image", sys.argv[arg])
    theImage = str(sys.argv[arg])
    # Reading image
    font = cv2.FONT_HERSHEY_PLAIN
    img2 = cv2.imread(theImage, cv2.IMREAD_COLOR)
    img3 = cv2.imread(theImage, cv2.IMREAD_COLOR)
    tempImg = cv2.imread(theImage)
    gray = cv2.cvtColor(tempImg, cv2.COLOR_RGB2GRAY)
    cv2.imwrite("gray.jpg", gray)

    frame = cv2.imread("gray.jpg")
    try:
        os.remove("gray.jpg")
    except: pass
    black_mask = cv2.inRange(frame, lower_black, upper_black)
    canny = cv2.GaussianBlur(black_mask, (11, 11), 0)
    dilated = cv2.dilate(canny, (1, 1), iterations=1)

    # ret, threshold = cv2.threshold(dilated, 100, 255, cv2.THRESH_BINARY)

    (contours, heirarchy) = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    cv2.drawContours(rgb, contours, -1, (0, 255, 0), 2)
    #cv2.imwrite("contours.jpg", rgb)
    #print("Number of Controus: ", len(contours))
    totalArea = 0;
    for c in contours:
        totalArea += cv2.contourArea(c)
        
    #print("Total Area: ", totalArea)
    #averageArea = totalArea/len(contours)
    #print("Average Area: ", averageArea)
    for c in contours:

        x, y, w, h = cv2.boundingRect(c)
        cropped_image = img2[y - boxSizeChange:y + h + boxSizeChange, x - boxSizeChange:x + w + boxSizeChange]
        # cropped_image = img2[y:y + h, x:x + w]

        if cropped_image is not None:
            count += 1

            for i in range(0 - boxSizeChange, cropped_image.shape[0]):
                for v in range(0 - boxSizeChange, cropped_image.shape[1]):
                    currentArray.append(str(x + i) + "," + str(y + v))

            sm = difflib.SequenceMatcher(None, currentArray, prevArray)
            # sm2 = difflib.SequenceMatcher(None, prevArray, currentArray)
            # print(sm.ratio(), count)
            if sm.ratio() > .75:
                continue

            prevArray.clear()
            changedNum = str(count)
            ratio = 1
            dim = (1, 1)
            
            area = cropped_image.shape[0] * cropped_image.shape[1]
            #print(area, count)
            #print("lowerBound: ", averageArea * .5)
            #print("upperBound: ", averageArea * 3)
            #if averageArea * .5 <= area < averageArea * 3:
            if 2000 < area < 30000:
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
                # print(str(round(prediction[0, 0], 4)) + ", " + str(round(prediction[0, 1], 4)) + ", " + str(round(prediction[0, 2], 4)) + ", " + str(round(prediction[0, 3], 4)))
                if prediction[0, 0] > prediction[0, 1] and prediction[0, 0] > prediction[0, 2] and prediction[0, 0] > prediction[0, 3]:
                    # print("Single Egg")
                    eggCount += 1
                    cv2.putText(img3, "One Egg" + str(count), (x - boxSizeChange, y - boxSizeChange), font, 1, (255, 0, 0))
                elif prediction[0, 1] > prediction[0, 0] and prediction[0, 1] > prediction[0, 2] and prediction[0, 1] > prediction[0, 3]:
                    # print("Two Eggs")
                    eggCount += 2
                    cv2.putText(img3, "Two Eggs" + str(count), (x - boxSizeChange, y - boxSizeChange), font, 1, (255, 0, 255))
                elif prediction[0, 2] > prediction[0, 0] and prediction[0, 2] > prediction[0, 1] and prediction[0, 2] > prediction[0, 3]:
                    # print("Three Eggs")
                    eggCount += 3
                    cv2.putText(img3, "ThreeEggs" + str(count), (x - boxSizeChange, y - boxSizeChange), font, 1, (255, 255, 0))
                else:
                    # print("Other")
                    eggCount += 0
                    cv2.putText(img3, "Other" + str(count), (x - boxSizeChange, y - boxSizeChange), font, 1, (255, 255, 255))
                # cv2.imwrite('/home/woowat/Documents/Capstone Project/Testing image stuff/testingIMageCropping/Images/Cropped Image' + changedNum + ' .jpg', l_img)
                l_img = blankImage.copy()
                cv2.rectangle(img3, (x - boxSizeChange, y - boxSizeChange), (x + w + boxSizeChange, y + h + boxSizeChange), (0, 255, 0), 2)
                #cv2.putText(img3, changedNum, (x, y), font, 1, (255, 0, 0))

        for i in range(len(currentArray)):
            prevArray.append(currentArray[i])
        currentArray.clear()

    print(sys.argv[arg] + "Egg Count:", eggCount)
    cv2.imwrite("Rectangle" + theImage, img3)

cv2.waitKey(0)
cv2.destroyAllWindows()
