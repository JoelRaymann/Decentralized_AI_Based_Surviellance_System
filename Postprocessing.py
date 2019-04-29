import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import cv2
import time
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from Preprocessing import ReadClasses, ReadAnchors, GenerateColors, ScaleBoxes, PreprocessImageHybrid, DrawBoxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body

def YoloFilterBoxes(boxConfidence, boxes, boxClassProbs, threshold = .6):
    '''Filters YOLO boxes by thresholding on object and class confidence.
    
    Arguments:
        boxConfidence {tensor} -- tensor of shape (19, 19, 5, 1) showing prob. of whether there is a recognized item
        in the box or not
        boxes {tensor} -- tensor of shape (19, 19, 5, 4) consisting of shape of the box
        boxClassProbs {tensor} -- tensor of shape (19, 19, 5, 80) consisiting of all class prob. in that box

    Keyword Arguments:
        threshold {float} -- real value, if [ highest class probability score < threshold], then get rid of the 
        corresponding box (default: {0.6})

    Returns:
        scores -- tensor of shape (None,), containing the class probability score for selected boxes
        boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
        classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

    Dev Note: "None" is here because we don't know the exact number of selected boxes, as it depends on the threshold. 
    For example, the actual output size of scores would be (10,) if there are 10 boxes.    
    '''
    boxScores = boxConfidence * boxClassProbs # Compute Box Scores

    # Find the box_classes using max box_scores, keep track of the corresponding score
    boxClasses = K.argmax(boxScores, axis = -1)
    boxClassScores = K.max(boxScores, axis = -1)

    # Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    filteringMask = boxClassScores >= threshold

    # Apply the mask to scores, boxes, and classes
    scores = tf.boolean_mask(boxClassScores, filteringMask)
    boxes = tf.boolean_mask(boxes, filteringMask)
    classes = tf.boolean_mask(boxClasses, filteringMask)

    return scores, boxes, classes

def YoloNonMaxSuppression(scores, boxes, classes, maxBoxes = 10, iouThreshold = 0.5):
    '''Applies Non-max suppression (NMS) to set of boxes
    
    Arguments:
        scores {tensor} -- tensor of shape (None,), feed output of YoloFilterBoxes()
        boxes {tensor} -- tensor of shape (None, 4), output of YoloFilterBoxes() that have been scaled to the image size
        classes {tensor} -- tensor of shape (None,), output of YoloFilterBoxes()
    
    Keyword Arguments:
        maxBoxes {int} -- maximum number of predicted boxes you'd like (default: {10})
        iouThreshold {float} -- real value, "intersection over union" threshold used for NMS filtering (default: {0.5})

    Return:
        scores -- tensor of shape (, None), predicted score for each box
        boxes -- tensor of shape (4, None), predicted box coordinates
        classes -- tensor of shape (, None), predicted class for each box

    Dev Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    '''
    
    maxBoxesTensor = K.variable(maxBoxes, dtype = 'int32') # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([maxBoxesTensor])) # initializing variable maxBoxTensor with maxBoxes variable

    # Using tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    nmsIndices = tf.image.non_max_suppression(boxes, scores, maxBoxesTensor, iouThreshold)

    # Using K.gather() to select only the boxes, scores and classes mentioned in nmsIndices
    scores = K.gather(scores, nmsIndices)
    boxes = K.gather(boxes, nmsIndices)
    classes = K.gather(classes, nmsIndices)

    return scores, boxes, classes

def YoloEval(yoloOutputs, imageShape = (720. , 1280.), maxBoxes = 10, scoreThreshold = 0.6, iouThreshold = 0.5):
    '''Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.
    
    Arguments:
        yoloOutputs {tensors} -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    boxConfidence: tensor of shape (None, 19, 19, 5, 1)
                    boxXY: tensor of shape (None, 19, 19, 5, 2)
                    boxWH: tensor of shape (None, 19, 19, 5, 2)
                    boxClassProbs: tensor of shape (None, 19, 19, 5, 80)
    
    Keyword Arguments:
        imageShape {tensor} -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype) (default: {(720. , 1280.)})
        maxBoxes {int} -- maximum number of predicted boxes you'd like (default: {10})
        scoreThreshold {float} -- if [ highest class probability score < threshold], then get rid of the corresponding box (default: {0.6})
        iouThreshold {float} -- "intersection over union" threshold used for NMS filtering (default: {0.5})
    
    Return:
        scores -- tensor of shape (None, ), predicted score for each box
        boxes -- tensor of shape (None, 4), predicted box coordinates
        classes -- tensor of shape (None,), predicted class for each box    
    '''
    
    # Retrieve outputs of the YOLO model
    boxConfidence, boxXY, boxWH, boxClassProbs = yoloOutputs

    # convert boxes to be ready for filtering function
    boxes = yolo_boxes_to_corners(boxXY, boxWH)

    # Use YoloFilterBoxes() to filter out boxes, scores, and classes
    scores, boxes, classes = YoloFilterBoxes(boxConfidence, boxes, boxClassProbs,threshold = scoreThreshold)

    # Scale the box back to original image shape -- sed lyf!!!! only 480p???
    boxes = ScaleBoxes(boxes, imageShape)

    # Using Non-max suppression with the iouThreshold
    scores, boxes, classes = YoloNonMaxSuppression(scores, boxes, classes, maxBoxes, iouThreshold)

    return scores, boxes, classes


