import logging
from pynput.keyboard import Listener
import zmq
import json
import random
import sys
import os
import time
import cv2
import base64
import threading
import subprocess
import keyboard
from config import *


# sys.path.append("../../Projects/yolo_v4/darknet-master")

# os.chdir("/media/philippos/35b88023-1615-40ee-b7cb-374948324648/Projects/yolo_v4/darknet-master")


global puller

global pusher

global pusher_action_recognizer

global pusher_video

global pusher_object_detector


global pusher_image

global puller_object_detector

global puller_action_recognizer

global port_client_camera


global ip_Controller

global ip_action_recognizer

global ip_object_detector


global port_Camera_Controller



try:
    ip_Controller
except NameError:
    ip_Controller = 'localhost'

try:
    ip_action_recognizer
except NameError:
    ip_action_recognizer = 'localhost'

try:
    ip_object_detector
except NameError:
    ip_object_detector = 'localhost'



try:
    port_Camera
except NameError:
    port_Camera_Controller = '5558'



camera_port = 0





if(len(sys.argv) > 1):
    if(sys.argv[1] != None and sys.argv[1] != '-1'):

        port_Camera_Controller = sys.argv[1]

if(len(sys.argv) > 2):
    if(sys.argv[2] != None and sys.argv[2] != '-1'):

        ip_Controller = sys.argv[2]

if(len(sys.argv) > 3):
    if(sys.argv[3] != None and sys.argv[3] != '-1'):

        ip_action_recognizer = sys.argv[3]

if(len(sys.argv) > 4):
    if(sys.argv[4] != None and sys.argv[4] != '-1'):

        ip_object_detector = sys.argv[4]

if(len(sys.argv) > 5):
    if(sys.argv[5] != None and sys.argv[5] != '-1'):

        camera_port = sys.argv[5]


def initSockets():

    global puller

    global pusher

    global pusher_action_recognizer

    global pusher_video

    global pusher_object_detector

    global pusher_image

    global puller_object_detector


    global puller_action_recognizer

    global ip_Controller

    global ip_action_recognizer

    global ip_object_detector

    print("Initializing Sockets")

    context = zmq.Context()
    puller = context.socket(zmq.PULL)
    puller.connect("tcp://%s:%s" % (ip_Controller, port_Camera))

    pusher = context.socket(zmq.PUSH)
    pusher.bind("tcp://*:%s" % str(int(port_Camera)+10))

    pusher_action_recognizer = context.socket(zmq.PUSH)
    pusher_action_recognizer.bind("tcp://*:%s" % 6100)

    pusher_video = context.socket(zmq.PUB)

    pusher_video.connect("tcp://%s:%s" % (ip_action_recognizer, 6101))

    puller_action_recognizer = context.socket(zmq.PULL)
    puller_action_recognizer.connect(
        "tcp://%s:%s" % (ip_action_recognizer, 6102))

    pusher_object_detector = context.socket(zmq.PUSH)
    pusher_object_detector.bind("tcp://*:%s" % 6200)

    pusher_image = context.socket(zmq.PUB)

    pusher_image.connect("tcp://%s:%s" % (ip_object_detector, 6201))

    puller_object_detector = context.socket(zmq.PULL)
    puller_object_detector.connect("tcp://%s:%s" % (ip_object_detector, 6202))


jsonmsgTemplate = {
            'Sender': "Vision",
            'Source': "Vision",
            'Component': "-",
            'SessionId': "-",
            'Message': "",
        }





def sendSnapshot(msg_json):

     if(msg_json == None):

        sessionId = 'test'

     else:

        sessionId = msg_json["SessionId"]

     # video capture source camera (Here webcam of laptop)
     cap = cv2.VideoCapture(int(camera_port))
     img_name = "captured_image.png"
     img_path = os.path.abspath(img_name)
     while(True):

         ret, frame = cap.read()
         cv2.imshow('img1', frame)  # display the captured image
         if cv2.waitKey(1) & 0xFF == 32:  # save on pressing 'y'
            encoded, buffer = cv2.imencode('.jpg', frame)
            jpg_as_text = base64.b64encode(buffer)
            pusher_object_detector.send_string(
                "sending snapshot SessionId:%s" % sessionId)

            cv2.destroyAllWindows()
            break
     cap.release()

     # pusher_object_detector.send_string("snapshot sent")
     
     message = puller_object_detector.recv_string()
    
     if(message == 'waiting'):

        pusher_image.send(jpg_as_text)
     print("image sent")

     results = puller_object_detector.recv_json()
     return results, None



def recordVideo(sessionId):
    width, height = 360,240

    frames_per_second = 12.0
    video_type='mpeg'

   # cap = cv2.VideoCapture("%s")%camera_port
    cap = cv2.VideoCapture(int(camera_port))
    out = cv2.VideoWriter('%s.webm'%sessionId,  cv2.VideoWriter_fourcc(*'VP80'), 12.0, (640,480))

    stop_recording=False
    start_recording=False
    
    pusher_action_recognizer.send_string("start recording SessionId:%s"%sessionId)
    while True:
      
        if(stop_recording):

            break
        ret, frame = cap.read()
        cv2.imshow('frame',frame)
        key_press=cv2.waitKey(1)

        
        if key_press == 32:
            start_recording=True
          
            while(start_recording and not stop_recording):
                ret, frame = cap.read()
                cv2.imshow('frame',frame)
                encoded, buffer = cv2.imencode('.jpg', frame)
                jpg_as_text = base64.b64encode(buffer)
                pusher_video.send(jpg_as_text)
                out.write(frame)
                key_press=cv2.waitKey(1)   
                
                if key_press == 32 :
                    stop_recording=True
                    
                    break


    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("video sent")
    pusher_action_recognizer.send_string("stop recording")
   # encoded, buffer = cv2.imencode('.webm', frame)
   # jpg_as_text = base64.b64encode(buffer)

    #target = open('%s.webm'%sessionId, 'rb')

    #size = os.stat(target).st_size
    #file_to_sned = target.read(size)    
   # pusher_action_recognizer.send(target)
    results=    puller_action_recognizer.recv_json()
    return results


def mainThread():

    initSockets()

    while(True):

        
        msg_json = puller.recv_json()

      
        print(msg_json)
        if("Snapshot" in msg_json['Message']):

            vision_results, objectKnown = sendSnapshot(msg_json)
            pusher.send_json(vision_results)

        if("Record" in msg_json['Message']):
            vision_results = recordVideo(msg_json["SessionId"])

            
            pusher.send_json(vision_results)

        if("Capture" in msg_json['Message']):
            vision_results, objectKnown = capture()

       # msg_to_send=msg_json
       # msg_to_send['Message']=vision_results
       # msg_to_send['Source']="Vision"
       # print(msg_to_send)

       # pusher.send_json(msg_to_send)






x = threading.Thread(
            target=mainThread, args=())

x.start()






