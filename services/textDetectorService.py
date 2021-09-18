from easyocr import easyocr
from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
from spellchecker import SpellChecker

spell = SpellChecker()
net = cv2.dnn.readNet("models/frozen_east_text_detection.pb")
reader = easyocr.Reader(['en'])  #


def text_detector(image):

    orig = image
    (H, W) = image.shape[:2]

    whiteImage = np.zeros([320, 640, 3], dtype=np.uint8)
    whiteImage.fill(255)

    (newW, newH) = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)

    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)

    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):

        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < 0.5:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    boxes = non_max_suppression(np.array(rects), probs=confidences)

    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        boundary = 6

        text = orig[startY - boundary:endY + boundary, startX - boundary:endX + boundary]
        # Converting image to grayscale
        # text = cv2.cvtColor(text.astype(np.uint8), cv2.COLOR_BGR2GRAY)

        # adding the parts of cut images in to the blank image
        whiteImage[startY - boundary:endY + boundary, startX - boundary:endX + boundary] = text

    # convert the whole image to grayscale
    text = cv2.cvtColor(whiteImage.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    # Read the result from easy ocr
    result = reader.readtext(text)

    words = ''

    # Loop through the detected words and do spell correction
    if len(result) > 0:
        print(result)
        for i in result:
            if not has_numbers(i[1]): # Check if the word has numbers
                words += spell.correction(i[1]) + ', '  # If no number do the spell correction
            else:
                words += i[1] + ', '  # If there are number ignore spell correction
    else:
        print('no words')

    return words # Return the detected and spell corrected words


def has_numbers(input_string):
    return any(char.isdigit() for char in input_string)
