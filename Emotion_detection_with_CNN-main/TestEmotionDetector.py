import cv2
import numpy as np
from keras.models import model_from_json
from PIL import ImageGrab
import pafy
import cv2

from threading import Thread
import time

from mss import mss

class ThreadedCamera(object):
    def __init__(self, src=0):
        url = src
        video = pafy.new(url)
        best = video.getbest(preftype="mp4")

        self.capture = cv2.VideoCapture(best.url)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
       
        # FPS = 1/X
        # X = desired FPS
        self.FPS = 1/200
        self.FPS_MS = int(self.FPS * 1000)
        
        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        
    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(self.FPS)
            
    def show_frame(self):
        cv2.imshow('frame', self.frame)
        cv2.waitKey(self.FPS_MS)

    def get_frame(self):
        return self.frame


emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

# start the webcam feed
#cap = cv2.VideoCapture(0)

# pass here your precorded video path
# you may download one from here : https://www.pexels.com/video/three-girls-laughing-5273028/

#cap = cv2.VideoCapture("C:/Users/MingJie/Desktop/real-time-detection/Video/test1.mp4")

# pass here your LIVE URL video path
# url = "https://www.youtube.com/watch?v=RWjn0o6mTuA&ab_channel=MonsterKongMarketing"
# video = pafy.new(url)
# best = video.getbest(preftype="mp4")
# cap = cv2.VideoCapture(best.url)

# faster way to process LIVE URL video path

src = "https://www.youtube.com/watch?v=rDq_rKWWNA8&ab_channel=FunLamb"
threaded_camera = ThreadedCamera(src)


#threaded cam, screen capture, fast
# while True:
#     try:
#         frame = threaded_camera.get_frame()
#         # frame = cv2.resize(frame, (1280, 720))
#         # frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

#         face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # detect faces available on camera
#         num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

#         # take each face available on the camera and Preprocess it
#         for (x, y, w, h) in num_faces:
#             cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
#             roi_gray_frame = gray_frame[y:y + h, x:x + w]
#             cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

#             # predict the emotions
#             emotion_prediction = emotion_model.predict(cropped_img)
#             maxindex = int(np.argmax(emotion_prediction))
#             cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#         # frame = cv2.resize(frame, (0, 0), fx=1/0.25, fy=1/0.25)
#         #frame = cv2.resize(frame, (1280, 720))
#         cv2.imshow('Justice for Depp', frame)
#         cv2.waitKey(threaded_camera.FPS_MS)

#     except AttributeError:
#         pass
        

#Capture screen
# normal screen, not full screen
# bounding_box = {'top': 175, 'left': 120, 'width': 1250, 'height': 700}
#side by side
bounding_box = {'top': 195, 'left': 25, 'width': 890, 'height': 500}
sct = mss()

while True:
    sct_img = sct.grab(bounding_box)
    frame = np.array(sct_img)
    #cv2.imshow('screen', np.array(sct_img))

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break

    # Find haar cascade to draw bounding box around face
    #frame = cv2.resize(frame, (1280, 720))
    # if not ret:
    #     break
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Justice for Depp', np.array(frame))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
#slow screen capture, obsolete
# while True:
#     # start the screen capture
#     printscreen_pil =  ImageGrab.grab()
#     printscreen_numpy =   np.array(printscreen_pil.getdata(),dtype='uint8')\
#     .reshape((printscreen_pil.size[1],printscreen_pil.size[0],3)) 
#     frame = printscreen_numpy
#     #frame = cv2.imshow('window',printscreen_numpy)
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break

#     # Find haar cascade to draw bounding box around face
#     #ret, frame = cap.read()
#     frame = cv2.resize(frame, (1280, 720))
#     # if not ret:
#     #     break
#     face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # detect faces available on camera
#     num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

#     # take each face available on the camera and Preprocess it
#     for (x, y, w, h) in num_faces:
#         cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
#         roi_gray_frame = gray_frame[y:y + h, x:x + w]
#         cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

#         # predict the emotions
#         emotion_prediction = emotion_model.predict(cropped_img)
#         maxindex = int(np.argmax(emotion_prediction))
#         cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#     cv2.imshow('Emotion Detection', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# #cap.release()
# cv2.destroyAllWindows()