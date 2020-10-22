import zmq
import json
import random
import sys

import threading
import time
import base64
import cv2
import numpy as np

from config import *


SENDER = "Sender:Controller"
SEPERATOR = ">-<"
SOURCE = "Source:Vision"
SESSIONID = "SessiondId:1"

global port_Controller

global port_KB

global port_Camera

global ip_client_Camera

global ip_client_UI

global ip_client_KB




try:
    port_Controller
except NameError:
    port_UI = '5556'


try:
    port_KB
except NameError:
    port_KB = '5557'


try:
    port_Camera
except NameError:
    port_Camera_Controller = '5558'



# socket.setsockopt(zmq.LINGER, 0)

try:
    ip_client_UI
except NameError:
    ip_client_UI = 'localhost'


try:
    ip_client_KB
except NameError:
    ip_client_KB = 'localhost'


try:
    ip_client_Camera
except NameError:
    ip_client_Camera = 'localhost'




ip0=ip_client_UI
ip1=ip_client_KB
ip2=ip_client_Camera





if(len(sys.argv)>1):
    if(sys.argv[1]!=None and sys.argv[1]!='-1'):

        ip0=sys.argv[1]

if(len(sys.argv)>2):
    if(sys.argv[2]!=None and sys.argv[2]!='-1'):

        ip1=sys.argv[2]


if(len(sys.argv)>3):
    if(sys.argv[3]!=None and sys.argv[3]!='-1'):

        ip2=sys.argv[3]






sub_port = 6000


#ips = ["139.91.183.118", "localhost", "localhost"]
ports=[port_UI,port_KB,port_Camera]

print(ports)

ips = [ip0, ip1, ip2]
partners = ["UI", "KB", "Vision"]


context = zmq.Context()

puller_KB=context.socket(zmq.PULL)
puller_UI=context.socket(zmq.PULL)
puller_Vision=context.socket(zmq.PULL)

pusher_KB=context.socket(zmq.PUSH)
pusher_UI=context.socket(zmq.PUSH)
pusher_Vision=context.socket(zmq.PUSH)


jsonmsgTemplate = {
            'Sender': "Controller",
            'Source': "-",
            'Component': "-",
            'SessionId': "-",
            'Message': "",
        }


jsonInitMessage = {
            'Sender': "Controller",
            'Source': "-",
            'Component': "-",
            'SessionId': "-",
            'Message': "Hello",
        }


jsonInitMessage2 = {
            'Sender': "Controller",
            'Source': "-",
            'Component': "-",
            'SessionId': "-",
            'Message': "Waiting for input",
        }


jsonInitMessage3 = {"Sender": "Controller",
"Source": "Vision",
"Component": "ObjectDetector",
"SessionId": 10,
"Message": [["bottle", 0.67, [["open", 0.5], ["empty", 0.32]]], ["phone", 0.47, [["connected", 0.5]]]]}


def getVision(json_message_dict):

    message = json_message_dict['Message']
    print(json_message_dict)
    object_known = True
    print(message)
    if('Snapshot' or 'Record Video' in message):

        msg_json = json.load(open('json_Controller_example.json'))

        scenario = 'A1'
    elif('Record Video' in message):

        msg_json = json.load(open('json_Controller_example2.json'))
        scenario = 'A2'
        object_known = False
    elif('Capture' in message):

        msg_json = json.load(open('json_Controller_example2.json'))
        scenario = 'B'

    return msg_json["Message"], object_known


def initSockets(ports, ips,puller_KB,puller_UI,puller_Vision,pusher_KB,pusher_UI,pusher_Vision):

    # socket.bind("tcp://139.91.183.118:%s" %port)

  
    puller_UI.connect("tcp://%s:%d" % (ips[0], int(ports[0])+10))
   


    pusher_UI.bind("tcp://*:%s" % (ports[0]))


    # send work

    # sockets[counter].send_json(jsonInitMessage)

    # msg_rcv_json = sockets[counter].recv_json()  # actual message
    # print(msg_rcv_json)
    print("Successfully connected to machine %s" % ports[0])



    puller_KB.connect("tcp://%s:%s" % (ips[1], ports[1]+10))
   

   
    pusher_KB.bind("tcp://*:%s" % (ports[1]))


    print("Successfully connected to machine %s" % ports[1])


    puller_Vision.connect("tcp://%s:%s" % (ips[2], ports[2]+10))
   

   
    pusher_Vision.bind("tcp://*:%s" % (ports[2]))


    print("Successfully connected to machine %s" % ports[1])



def UI_listener():



    
    


    while True:
        
        msg_from_UI = puller_UI.recv_json()

       

        if("Record" in msg_from_UI["Message"]):


              


            pusher_Vision.send_json(msg_from_UI)



        if("Snapshot" in msg_from_UI["Message"]):


              


            pusher_Vision.send_json(msg_from_UI)


            


def KB_listener():





    while True:

        msg_from_KB = puller_KB.recv_json()

        print(1111) 


        msg_to_send_UI2 = jsonmsgTemplate

       # msg_to_send_UI2['SessionId'] = task_id
        msg_to_send_UI2['Source'] = 'KB'
        msg_to_send_UI2['Message'] = msg_from_KB["Message"]

        #print(msg_from_KB)


        pusher_UI.send_json(msg_to_send_UI2) 



def Vision_listener():





    while True:

        msg_from_Vision = puller_Vision.recv_json()

        
        msg_from_Vision["Sender"]='Controller'

        pusher_UI.send_json(msg_from_Vision)


        if(msg_from_Vision["Component"]=="Object Detector"):
     
            pusher_KB.send_json(msg_from_Vision)


       



initSockets(ports, ips,puller_KB,puller_UI,puller_Vision,pusher_KB,pusher_UI,pusher_Vision)


x = threading.Thread(
    target=UI_listener, args=())

x.start()
       


x = threading.Thread(
    target=KB_listener, args=())

x.start()

x = threading.Thread(
    target=Vision_listener, args=())

x.start()
       

  





