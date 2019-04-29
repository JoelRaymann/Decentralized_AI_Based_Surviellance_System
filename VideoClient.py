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
import time
import tensorflow as tf
from keras import backend as K
from flask import Flask, Response
from kafka import KafkaConsumer
from cv2 import cv2
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body
from Preprocessing import ReadAnchors, ReadClasses, PreprocessImageHybrid, ScaleBoxes, GenerateColors, DrawBoxes
from Postprocessing import YoloEval, YoloFilterBoxes, YoloNonMaxSuppression
from MainUtilities import PredictNetwork, PredictNodeCam, ModelLoader
from ExceptionHandler import RetryError
from FaceRecognitionUtilities import LoadDatabase, LoadFaceModel, SearchPerson
# Connect to kafka server
try:
    consumer = KafkaConsumer('Video-Relay', bootstrap_servers = ["localhost:9092"])

except Exception as err:
    print("[+] Connection Failed")
    print(err)
    traceback.print_exc()
    quit(0)
finally:
    pass

# Starting a mini flask node to get the images relay to use
app = Flask(__name__)

@app.route('/')
def index():
    '''return a multipart response
    '''
    return Response(KafkaStream(), mimetype = 'multipart/x-mixed-replace; boundary=frame')

def KafkaStream():
    '''
    Function to get the stream, predict and generate o/p
    '''
    # Getting a Keras Session
    print("[+] Setting up Keras Session")
    try:
        sess = K.get_session()
    except Exception as err:
        print("[+] Session not acquired -- ERROR: ", err)
        traceback.print_exc()
        exit(1)
    finally:
        pass

    # Getting Class names and anchors
    classNames = ReadClasses("model_data/coco_classes.txt")
    anchors = ReadAnchors("model_data/yolo_anchors.txt")
    imageShape = (480., 640.)

    # Loads the model
    while(True):
        yoloModel, status, retry = ModelLoader("model_data/yolo.h5")

        # check status and retry factors
        if status == False:
            if retry == False:
                print("[+] Quittting application after an exception")
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
    FRmodel, _ = LoadFaceModel()
    print("[+] Loading Database")
    database = LoadDatabase("Database", FRmodel)
    print("[+] Database loaded")
    for message in consumer:
        image = np.asarray(bytearray(message.value), dtype = "uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = PredictNetwork(sess, yoloModel, FRmodel, database, image, classNames, scores, boxes, classes)
        _, buffer = cv2.imencode(".png", image)
        yield (b'--frame\r\n'
        b'Content-Type: image/png\r\n\r\n' + buffer.tobytes() + b"\r\n\r\n"
        )
    sess.close()

if __name__ == '__main__':
    # Model Loading and preping

    

    # Load up the app
    try:
        app.run(host="localhost", debug=True)
    
    except KeyboardInterrupt:
        print("[+] Shutting down server")
    
    except Exception as err:
        print("[+] Unhandled Exception")
        print(err)
        traceback.print_exc()
    
    finally:
        consumer.close()