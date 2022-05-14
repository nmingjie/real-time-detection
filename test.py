import numpy as np
from keras.models import model_from_json
from PIL import ImageGrab
import cv2

while(True):
    printscreen_pil =  ImageGrab.grab()
    printscreen_numpy =   np.array(printscreen_pil.getdata(),dtype='uint8')\
    .reshape((printscreen_pil.size[1],printscreen_pil.size[0],3)) 
    #cv2.VideoCapture(printscreen_numpy)
    #cv2.imshow('window',printscreen_numpy)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break