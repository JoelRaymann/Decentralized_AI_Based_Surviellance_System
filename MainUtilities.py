import argparse
import os
import traceback
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
from cv2 import cv2
import time
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body
from Preprocessing import ReadAnchors, ReadClasses, PreprocessImageHybrid, ScaleBoxes, GenerateColors, DrawBoxes
from Postprocessing import YoloEval, YoloFilterBoxes, YoloNonMaxSuppression
from ExceptionHandler import RetryError

# Main Utility Functions
def PredictNetwork(sess, yoloModel, FRModel, database, image, classNames, scores, boxes, classes):
    '''Function to do prediction on the given image
    
    Arguments:
        sess {keras sess} -- Session for keras model
        yoloModel {model} -- Loaded model
        FRmodel {model} -- face recognition model
        database {dict} -- list of all encodings
        image {image} -- image to process
        classNames {list} -- list of all predictable class
        scores {tensor} -- tensor of prob. of every prediction
        boxes {tensor} -- tensor consisting of box info
        classes {tensor} -- tensor consisting of all predicted classes
    
    Return:
        cv2image -- Processed cv2 image
    '''
    try:
        image, imageData = PreprocessImageHybrid(image, modelImageSize = (608, 608))
        # Feed the image in model
        outScores, outBoxes, outClasses = sess.run([scores, boxes, classes], feed_dict = {yoloModel.input: imageData, K.learning_phase(): 0})
        # generate colors
        colors = GenerateColors(classNames)
        # Draw prediction box
        DrawBoxes(image, outScores, outBoxes, outClasses, classNames, colors, FRModel, database)
        # Convert back to cv2 image
        cv2image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        return cv2image

    except Exception as err:
        print("Fatal Error: ", err)
        traceback.print_exc()
        exit(1)

def PredictNodeCam(sess, yoloModel, FRmodel, database, classNames, scores, boxes, classes):
    '''Function to do realtime prediction in the master node using cv2 framework
    
    Arguments:
        sess {Keras session} -- the keras sess with the model loaded
        yoloModel {model} -- Loaded model
        FRmodel {model} -- face recognition model
        database {dict} -- list of all encodings
        classNames {list} -- list of all predictable class
        scores {tensor} -- tensor of prob. of every prediction
        boxes {tensor} -- tensor consisting of box info
        classes {tensor} -- tensor consisting of all predicted classes
    
    Return:
        outScores -- tensor of shape (None, ), scores of the predicted boxes
        outBoxes -- tensor of shape (None, 4), coordinates of the predicted boxes
        outClasses -- tensor of shape (None, ), class index of the predicted boxes
        
    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes. 
    '''
    # get local cams
    camera = cv2.VideoCapture(cv2.CAP_DSHOW)
    try:
        while(True):
            # Exiting mechanism
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt
            
            # Read video frame by frame
            status, image = camera.read()

            if not status:
                raise IOError
            
            # Preprocess the image with cv2 and Pillow
            image, imageData = PreprocessImageHybrid(image, modelImageSize = (608, 608))

            # Feed the image in model
            outScores, outBoxes, outClasses = sess.run([scores, boxes, classes], feed_dict = {yoloModel.input: imageData, K.learning_phase():0})

            # generate colors
            colors = GenerateColors(classNames)

            # Draw prediction box
            DrawBoxes(image, outScores, outBoxes, outClasses, classNames, colors, FRmodel, database)

            # Convert back to cv2 image
            cv2image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

            cv2.imshow("output", cv2image)
    
    except KeyboardInterrupt:
        print("[+] Releasing camera and shuting it down")
    
    except IOError:
        print("[+] Read Camera error")

    except Exception as err:
        print("[+] This is bad, we don't what error is this?!!")
        print("[+] Send us a mail to check it out")
        print("[+] You Faced the following error: ", err)
        check = str(input("[+] Do you want to print the traceback error? (Y/N): ")).lower()
        if check == "y":
            traceback.print_exc()
    
    finally:
        camera.release()
        cv2.destroyAllWindows()
    
    return outScores, outBoxes, outClasses

def ModelLoader(modelPath:str):
    '''Loads the Trained Keras Model
    
    Arguments:
        modelPath {str} -- file Path of the trained model

    Return:
        yoloModel -- Loaded yolo model
        status -- checking bool var of the loaded model
        retry -- checking bool var for retrying function
    '''
    try:
        # Load the model
        print("[+] Loading Trained Yolo Model")
        yoloModel = load_model(modelPath)
        status = True # For checking purposes
        retry = False # For retrying purposes

    except ValueError:
        print("[+] Invalid model file type please enter .h5 type file")
        status = False
        raise RetryError

    except ImportError:
        print("[+] Invalid file path, please check if file path exist")
        status = False
        raise RetryError
    
    except RetryError:
        check = input("[+] You Can Try again. Do you wish to?(Y/N): ").lower()
        if check == "y":
            retry = True

    except Exception as err:
        print("[+] This is bad, we don't what error is this?!!")
        print("[+] Send us a mail to check it out")
        print("[+] You Faced the following error: ", err)
        check = str(input("[+] Do you want to print the traceback error? (Y/N): ")).lower()
        if check == "y":
            traceback.print_exc()
        status = False
        raise RetryError

    finally:
        return yoloModel, status, retry
