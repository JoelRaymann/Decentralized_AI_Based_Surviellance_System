import face_recognition
from cv2 import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import traceback
import os

def SearchPersonFR(image, database, model):
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
    encodings = fr.face_encodings(image)
    identity = "unknown"
    for enc in encodings:
        for name, known_enc in database.items():
            match = fr.compare_faces([known_enc], enc)
            if match[0].any():
                identity = name
                status = True
    return status, identity
    
def LoadDatabaseFR(imageFolderPath:str, model) -> dict:
    '''
    Function that loads all recognised personals
    
    Arguments:
        imageFolderPath {str} -- Path of all registered user images
    
    Returns:
        database -- user name to encodings dictionary
    '''
    database = dict()
    for imagePath in os.listdir(imageFolderPath):
        name = imagePath.split(".")[0]
        image = fr.load_image_file(os.path.join(imageFolderPath + "/" + imagePath))
        database[name] = fr.face_encodings(image)
    for name, known_enc in database.items():
        print(name, known_enc)
    return database

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
        
        # Detect face location
        faceLocations = face_recognition.face_locations(image)

        # Convert image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

        thickness = (image.size[0] + image.size[1]) // 300
        label = "Face Detected"
        for (top, right, bottom, left) in faceLocations:

            draw = ImageDraw.Draw(image)
            labelSize = draw.textsize(label, font)

            top = max(0, np.floor(top + 0.5).astype('int32')) # To handle upper bound
            left = max(0, np.floor(left + 0.5).astype('int32')) 
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32')) # To handle lower bound
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            print(label, (left, top), (right, bottom))

            if top - labelSize[1] >= 0:
                textOrgin = np.array([left, top - labelSize[1]])
            else:
                textOrgin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline = (255, 0, 0))
            draw.rectangle([tuple(textOrgin), tuple(textOrgin + labelSize)], fill = (255, 0, 0))
            draw.text(textOrgin, label, fill = (0, 0, 0), font = font)
            del draw

        # Convert back to cv2 image
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

        cv2.imshow("output", image)

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