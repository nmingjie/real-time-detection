import cv2
from simple_facerec import SimpleFacerec
import numpy as np
from PIL import ImageGrab
import pafy

from mss import mss

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Load Camera
#cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("C:/Users/MingJie/Desktop/real-time-detection/Video/elon.mp4")

# url = "https://www.youtube.com/watch?v=RWjn0o6mTuA&ab_channel=MonsterKongMarketing"
# video = pafy.new(url)
# best = video.getbest(preftype="mp4")
# cap = cv2.VideoCapture(best.url)

bounding_box = {'top': 195, 'left': 25, 'width': 890, 'height': 500}
sct = mss()



while True:
    # sct_img = sct.grab(bounding_box)
    # frame = np.array(sct_img)
    # start the screen capture
    printscreen_pil =  ImageGrab.grab()
    printscreen_numpy =   np.array(printscreen_pil.getdata(),dtype='uint8')\
    .reshape((printscreen_pil.size[1],printscreen_pil.size[0],3)) 
    frame = printscreen_numpy

    # #frame = cv2.imshow('window',printscreen_numpy)
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    #     break

    # ret, frame = cap.read()

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    small_frame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.75)
    cv2.imshow("Frame", small_frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

#cap.release()
cv2.destroyAllWindows()