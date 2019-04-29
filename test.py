from cv2 import cv2 as cv
from PIL import ImageDraw, ImageFont, Image
import os
import sys
import numpy as np
import random
import time

first = True
camera = cv.VideoCapture(cv.CAP_DSHOW)
camera.set( cv.CAP_PROP_FRAME_HEIGHT, 608 )
camera.set( cv.CAP_PROP_FRAME_WIDTH, 608 )
h = camera.get( cv.CAP_PROP_FRAME_HEIGHT )
w = camera.get( cv.CAP_PROP_FRAME_WIDTH )

try:
    while True:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        status, img = camera.read()
        if first == True:
            width = np.size(img, 0) # width = 800
            height = np.size(img, 1)
            print(height, width)
            first = False
        if status:
            cv.imshow("cams", img)
    
except KeyboardInterrupt:
    print("[+] closing cams")
    
except Exception as err:
    print("Error: ", err)

finally:
    camera.release()
