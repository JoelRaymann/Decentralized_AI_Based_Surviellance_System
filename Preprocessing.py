import colorsys
import imghdr
import os
import random
from keras import backend as K
from cv2 import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from FaceRecognitionUtilities import SearchPerson

def ReadClasses(classPath:str):
    '''Function to read class names for labelling
    
    Arguments:
        classPath {str} -- path to the class name file
    '''
    
    with open(classPath) as f:
        classNames = f.readlines()
    classNames = [c.strip() for c in classNames]
    return classNames

def ReadAnchors(anchorPath:str):
    '''Function to read and return all anchor boxes for YOLO
    
    Arguments:
        anchorPath {str} -- path to the anchor box detail file
    '''

    with open(anchorPath) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors

def GenerateColors(classNames:list) -> list:
    '''Function to generate a color to draw box for that
    particular class in the className list
    
    Arguments:
        classNames {list} -- list of all classes
    
    Returns:
        list -- list of all associated colors for the classes
    '''
    
    # Creating a Hue Saturation Value tuple unique for classnames
    hsvTuples = [(x/len(classNames), 1., 1.) for x in range(len(classNames))]
    
    # Generating colors
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsvTuples))
    # scaling RGB to 255 factor
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(10101) # Fixing a seed for consistent color
    random.shuffle(colors) # shuffle colors to decorrelate adjacent classes.
    random.seed(None) # Reseting back the seed
    return colors

def ScaleBoxes(boxes, imageShape):
    '''Scales the predicted boxes in order to be
    drawable on the image
    
    Arguments:
        boxes  -- all boxes predicted
        imageShape  -- shape to scale the image
    '''
    height = imageShape[0]
    width = imageShape[1]
    imageDims = K.stack([height, width, height, width])
    imageDims = K.reshape(imageDims, [1, 4])
    boxes = boxes * imageDims # Scaling up
    return boxes

def PreprocessImageWithPIL(imagePath:str, modelImageSize:tuple):
    '''Function to preprocess the entire image using PIL
    Library
    
    Arguments:
        imagePath {str} -- path of the image to process
        modelImageSize {tuple} -- tuple of size to resize
    '''
    
    # imageType = imghdr.what(imagePath)
    image = Image.open(imagePath)
    resizedImage = image.resize(tuple(reversed(modelImageSize)))
    imageData = np.array(resizedImage, dtype="float32")
    imageData /= 255
    imageData = np.expand_dims(imageData, 0) # Add Batch Dimensions
    return image, imageData

def PreprocessImageWithCV2(imagePath:str, modelImageSize:tuple):
    '''Function to process the image in CV2
    
    Arguments:
        imagePath {str} -- Path of the image to process
        modelImageSize {tuple} -- tuple of size to resize
    '''
    cv2image = cv2.imread(imagePath)
    resizedImage = cv2.resize(cv2image, tuple(reversed(modelImageSize)), interpolation = cv2.INTER_CUBIC)
    imageData = np.array(resizedImage, dtype = "float32")
    imageData /= 255.
    imageData = np.expand_dims(imageData, 0)
    return cv2image, imageData

def PreprocessImageHybrid(image, modelImageSize:tuple):
    '''Function to return scaled preprocessed image using 
    OpenCV2 and PIL
    
    Arguments:
        image {cv2 image} -- image data to resize
        modelImageSize {tuple} -- tuple of size to resize
    '''
    #Convert cv2 image from BGR to RGB
    # resizedImage = cv2.resize(image, dsize = tuple(modelImageSize), interpolation = cv2.INTER_LANCZOS4)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    resizedImage = image.resize(tuple(reversed(modelImageSize)))
    imageData = np.array(resizedImage, dtype="float32")
    imageData /= 255.
    imageData = np.expand_dims(imageData, 0)
    return image, imageData

def DrawBoxes(image, outScores, outBoxes, outClasses, classNames, colors, FRmodel, database):
    '''Function to draw bounding boxes on the image
    
    Arguments:
        image {Image} -- input image
        outScores {} -- Scores for the prediction
        outBoxes {} -- output boxes
        outClasses {list} -- output list of classes
        classNames {list} -- all class name list
        colors {list} -- all associated color to class name list
        FRmodel {model} -- Face Recognition model
        database {dict} -- known ppl to encoding
    '''
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    for i, c in reversed(list(enumerate(outClasses))):

        predictedClass = classNames[c]
        box = outBoxes[i]
        score = outScores[i]
        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32')) # To handle upper bound
        left = max(0, np.floor(left + 0.5).astype('int32')) 
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32')) # To handle lower bound
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

        # if predictedClass == "person":
        #     roi = image.crop((top, left, bottom, right))
        #     roi = cv2.cvtColor(np.asarray(roi), cv2.COLOR_RGB2BGR)
        #     _, identity = SearchPerson(roi, database, FRmodel)
        #     predictedClass += ": " + identity
            
        label = '{} {:.2f}'.format(predictedClass, score)

        draw = ImageDraw.Draw(image)
        labelSize = draw.textsize(label, font)

        print(label, (left, top), (right, bottom))

        if top - labelSize[1] >= 0:
            textOrgin = np.array([left, top - labelSize[1]])
        else:
            textOrgin = np.array([left, top + 1])
        
        # Drawing the bounding box
        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline = colors[c])
        draw.rectangle([tuple(textOrgin), tuple(textOrgin + labelSize)], fill = colors[c])
        draw.text(textOrgin, label, fill = (0, 0, 0), font = font)
        del draw