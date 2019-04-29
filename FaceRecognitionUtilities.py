from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
import tensorflow as tf
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import traceback
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
from fr_utils import *
from inception_blocks_v2 import *

def TripleLoss(ytrue, ypred, alpha = 0.2):
    '''
    Implementation of the triplet loss
    
    Arguments:
        ytrue {tensor} -- True labels
        ypred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    
    Keyword Arguments:
        alpha {float} -- hyperparam (default: {0.2})

    Returns:
        loss -- real number, value of the loss
    '''
    
    anchor, positive, negative = ypred[0], ypred[1], ypred[2]

    # Compute the distance of positive image pairs
    posDist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis = -1)
    # Compute the distance of the negative image pairs
    negDist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis = -1)
    # Subtracting both
    baseLoss = tf.add(tf.subtract(posDist, negDist), alpha)
    # summing all maximum of loss
    loss = tf.reduce_sum(tf.maximum(baseLoss, 0))
    return loss

def LoadDatabase(imageFolderPath:str, model) -> dict:
    '''
    Function that loads all recognised personals
    
    Arguments:
        imageFolderPath {str} -- Path of all registered user images
    
    Returns:
        database -- user name to encodings dictionary
    '''
    database = dict()
    for dbPath in os.listdir(imageFolderPath):
        name = dbPath
        encodingList = []
        for imagePath in os.listdir(os.path.join(imageFolderPath + "/" + dbPath)):
            encodingList.append(img_to_encoding(os.path.join(imageFolderPath + "/" + dbPath + "/" + imagePath), model))
        database[name] = encodingList
    return database

def LoadFaceModel():
    '''
    Function to load Face Model

    returns:
        FRModel -- Model of the loaded inception network
        status {bool} -- status of the model
    '''
    status = False
    FRmodel = None
    try:
        print("[+] Loading Face Recognition inception model")
        FRmodel = faceRecoModel(input_shape = (3, 96, 96))
        print("[+] Loading weights")
        FRmodel.compile(optimizer = 'adam', loss = TripleLoss, metrics = ['accuracy'])
        load_weights_from_FaceNet(FRmodel)
        print("[+] Loaded successfully")
        status = True
    
    except Exception as err:
        print("[+] Error: ", err)
        traceback.print_exc()
    
    finally:
        return FRmodel, status
    
def SearchPerson(image, database, model):
    '''
    Function to find who the person is
    
    Arguments:
        image {cv2 image} -- image of the person
        database {dict} -- encodings of all know ppl
        model {FRmodel} -- Face Recognition model

    returns:
        status {bool} -- status of detection
        identity {name} -- name of the person
    '''
    status = False
    encoding = ImageEncoding(image, model)
    minDist = 100
    identity = "unknown"
    for (name, enclist) in database.items():
        for enc in enclist:
            dist = np.linalg.norm(encoding - enc)
            if dist < minDist:
                minDist = dist
                identity = name
    if minDist > 0.6:
        identity = "unknown"
    else:
        status = True
    return status, identity

# Test DB
if __name__ == "__main__":
    model, _ = LoadFaceModel()
    database = LoadDatabase("Database", model)
    for name, enclist in database.items():
        print(name, " encoding list: ")
        for i in  enclist:
            print(i)