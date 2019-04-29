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
from MainUtilities import PredictNetwork, PredictNodeCam, ModelLoader
from FaceRecognitionUtilities import LoadDatabase, LoadFaceModel, SearchPerson, TripleLoss

# Main App
if __name__ == "__main__":

    # Getting a session for Keras
    print("[+] Setting up Keras Session")
    try:
        sess = K.get_session()
    except Exception as err:
        print("[+] Session not acquired -- ERROR: ", err)
        traceback.print_exc()
        exit(1)
    
    # Getting Class Names and anchors
    classNames = ReadClasses("model_data/coco_classes.txt")
    anchors = ReadAnchors("model_data/yolo_anchors.txt")
    imageShape = (480. , 640.)
    

    # Loads the model
    while(True):
        yoloModel, status, retry = ModelLoader("model_data/yolo.h5")
        
        # check status and retry factors
        if status == False:
            if retry == False:
                print("[+] Quiting application after an exception")
                exit(1)
            else:
                print("[+] Reverting back to previous checkpoint")
                continue
        else:
            break
    
    print("[+] Model Loaded")
    print("[+] Setting up model")
    yoloOutputs = yolo_head(yoloModel.output, anchors, len(classNames))
    scores, boxes, classes = YoloEval(yoloOutputs, imageShape)
    
    # Load Face Recognition model
    # FRmodel, _ = LoadFaceModel()
    # print("[+] Loading Database")
    # database = LoadDatabase("Database", FRmodel)
    FRmodel = None
    database = None #LoadDatabase("Database", FRmodel)
    print("[+] Database loaded")
    print("[+] Starting base node cams")
    outScores, outBoxes, outClasses = PredictNodeCam(sess, yoloModel, FRmodel, database, classNames, scores, boxes, classes)

    print("[+] Gracefully Shutting Down")
    sess.close()