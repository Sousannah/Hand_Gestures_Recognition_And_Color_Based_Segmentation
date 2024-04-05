import cv2
import numpy as np
import math
from cvzone.ClassificationModule import Classifier

cap = cv2.VideoCapture(0)

correct = 0
Accuracy = 0
counter = 0

classifier = Classifier('keras_model_segmentation.h5', "labels_seg.txt")

offset = 20
imgSize = 300

labels = ['blank', 'fist', 'five', 'ok', 'thumbsdown', 'thumbsup']

while True:
    success, img = cap.read()
    imgOutput = img.copy()

    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the blue color in HSV
    lower_blue = np.array([101, 50, 38])
    upper_blue = np.array([110, 255, 255])

    # Create a mask to detect the blue color
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Sort the contours by area and find the largest one
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_contour = contours[0]

        # Get the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Check if the cropped image is valid (non-empty)
        if imgCrop.size != 0:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Create a mask for the blue color region
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

            # Find contours in the blue color mask
            blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create a black background
            blue_segmented = np.zeros_like(img)

            if len(blue_contours) > 0:
                # Sort the contours by area and find the largest one
                blue_contours = sorted(blue_contours, key=cv2.contourArea, reverse=True)
                largest_blue_contour = blue_contours[0]

                # Draw the largest blue contour on the segmented image
                cv2.drawContours(blue_segmented, [largest_blue_contour], -1, (0, 0, 255), -1)

            # Create a white mask of the same size as the segmented image
            white_mask = np.ones_like(blue_segmented) * 255

            # Use bitwise AND to set white mask's pixels to white wherever segmented image has blue pixels
            result = cv2.bitwise_and(white_mask, blue_segmented)

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                result = cv2.resize(result, (wCal, imgSize))  # Resize result to match imgWhite shape
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                result = cv2.resize(result, (imgSize, hCal))  # Resize result to match imgWhite shape

            cv2.imshow("Blue Segmentation", result)

            modelImg = imgWhite.copy()
            gray = cv2.cvtColor(modelImg, cv2.COLOR_BGR2GRAY)
            gray_inverted = cv2.bitwise_not(gray)
            contours, hierarchy = cv2.findContours(gray_inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            _, binary = cv2.threshold(gray_inverted, 100, 255, cv2.THRESH_BINARY)

            # Convert binary image to 3-channel grayscale
            binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

            prediction, index = classifier.getPrediction(binary_rgb, draw=False)
            print(prediction, index)

            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 4)

            print(Accuracy)
            cv2.imshow("ImageWhite", imgWhite)
            cv2.imshow("Segmentation", binary)
            cv2.imshow("ImageCrop", imgCrop)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(Accuracy)
# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()