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
import darknet
import re
import math
import multiprocessing 
import concurrent.futures


from queue import Queue

from darknet_images import *


global ip_client_vision


global puller

global puller_image

global pusher_action_vision

#ip_client_vision='localhost'


try:
    ip_client_vision
except NameError:
    ip_client_vision='localhost'






if(ip_client_vision is None and len(sys.argv)>1):
    if(sys.argv[1]!=None and sys.argv[1]!='-1'):

        ip_client_vision=sys.argv[1]







def initSockets():

    global puller

    global puller_image

    global pusher_action_vision

    context = zmq.Context()
    puller = context.socket(zmq.PULL)
    puller.connect("tcp://%s:%s" % (ip_client_vision,6200))
    
    print("Connected with Vision client at ip: %s at port %s for listening"%(ip_client_vision,6200)) 

    puller_image = context.socket(zmq.SUB)
    puller_image.bind("tcp://*:%s" % 6201)
    puller_image.setsockopt_string(zmq.SUBSCRIBE, str(""))

    

   # print("Connected with UI clinet at ip: %s at port %s for listening"%(6201))
    print("Opened port %s for listening"%(6201))



    pusher_action_vision = context.socket(zmq.PUSH)
    pusher_action_vision.bind("tcp://*:%s" % 6202)


    print("Opened port %s for message forwarding"%(6200)) 

jsonmsgTemplate = {
            'Sender': "Vision",
            'Source': "Vision",
            'Component': "Object Detector",
            'SessionId': "-",
            'Message': "",

}




q = Queue()
q2 = Queue()

def receiveInstructions(out_q):
    
    global SessionId

    while(True):
        message = puller.recv_string()

        if('sending snapshot' in message):

            m = re.search('(?<=SessionId:)\w+', message)
            SessionId = m.group(0)
            print(message)
            print(SessionId)
           
            out_q.put(1)

        if(message == 'snapshot sent'):
            print(message)
            out_q.put(0)




def makePredictionsLabels():

   

      

        

    pusher_action_vision.send_string("waiting")
    
    curdir = os.getcwd()

    # video capture source camera (Here webcam of laptop)
    
    frame = puller_image.recv_string()
    print("image received")
        
    img = base64.b64decode(frame)
    npimg = np.fromstring(img, dtype=np.uint8)
    snapshot = cv2.imdecode(npimg, 1)
    img_name = "%s.jpg" % SessionId
    print("saving as %s"%img_name)
    img_path = os.path.abspath(img_name)
   # print(img_path)
    cv2.imwrite('%s' % img_name, snapshot)

    thresh = 0.01
    thresh_2 = 0.05
    os.chdir(
     "/media/philippos/35b88023-1615-40ee-b7cb-374948324648/Projects/yolo_v4/darknet-master")
    network, class_names, class_colors = darknet.load_network(
    "yolov4-labels.cfg",
    "obj.data",
    "yolov4-labels_last.weights",
    1
    )
    
    image, detections = image_detection(
        img_path, network, class_names, class_colors, thresh
        )
    
    darknet.print_detections(detections, "store_true")
   
    darknet.free_network_ptr(network)

    return detections,img_path


def makePredictionsStates(label,img_path,bbox):
    
    thresh_2 = 0.05
   
  
   # while(int(free.group(0))<2*10**9):
     #   mem_info=cuda.current_context().get_memory_info()
      #  free=re.search('(?<=free=)\d+', str(mem_info))
      #  print("waiting for gpu to get released. free memory %s"%free.group(0))
    
    center_x = bbox[0]+bbox[2]/2
    center_y = bbox[1]-bbox[3]/2

    paths = glob.iglob("backup/%s/*" % label)

    stt_detc = []
    for path in paths:
         path = os.getcwd()+os.sep+path
         print(path)
         labels = glob.glob("%s/*.data" % path)
         weights = glob.glob(
             "%s/*yolo-obj_final.weights" % path)

         network, class_names, class_colors = darknet.load_network(
        "yolo-obj_test.cfg",
        labels[0],
        weights[0],
        1
        )

         image, detections2 = image_detection(
                img_path, network, class_names, class_colors, thresh_2
                )


        
         darknet.print_detections(
             detections2, "store_true")
         for label2, confidence2, bbox2 in detections2:

            center_x2 = bbox2[0]+bbox2[2]/2
            center_y2 = bbox2[1]-bbox2[3]/2
            
            if(math.sqrt((center_x2-center_x)**2+(center_y2-center_y)**2) > 50):
                    continue
            stt_detc.append(
                [label2, "%.2f" % (float(confidence2)/100), ])
         darknet.free_network_ptr(network)
    #cuda.current_context().trashing.clear()
    #cuda.current_context().reset()
        
    return stt_detc













def makePredictions(out_q):

    global SessionId
   
    while(True):

        

                    pusher_action_vision.send_string("waiting")
                    puller_image.setsockopt_string(zmq.SUBSCRIBE, str(""))
                    curdir = os.getcwd()

                    # video capture source camera (Here webcam of laptop)

                    frame = puller_image.recv_string()
                    print("image received")
                    
                    img = base64.b64decode(frame)
                    npimg = np.fromstring(img, dtype=np.uint8)
                    snapshot = cv2.imdecode(npimg, 1)
                    img_name = "%s.jpg" % SessionId
                    print("saving as %s"%img_name)
                    img_path = os.path.abspath(img_name)
                   # print(img_path)
                    cv2.imwrite('%s' % img_name, snapshot)

                    thresh = 0.01
                    thresh_2 = 0.05
                    os.chdir(
                     "/media/philippos/35b88023-1615-40ee-b7cb-374948324648/Projects/yolo_v4/darknet-master")
                    network, class_names, class_colors = darknet.load_network(
                    "yolov4-labels.cfg",
                    "obj.data",
                    "yolov4-labels_last.weights",
                    1
                    )

                    image, detections = image_detection(
                        img_path, network, class_names, class_colors, thresh
                        )
                   

                    darknet.print_detections(detections, "store_true")
                    rslt_list = []

                    for label, confidence, bbox in detections:

                        center_x = bbox[0]+bbox[2]/2
                        center_y = bbox[1]-bbox[3]/2

                        paths = glob.iglob("backup/%s/*" % label)

                        stt_detc = []
                        for path in paths:
                             path = os.getcwd()+os.sep+path
                             print(path)
                             labels = glob.glob("%s/*.data" % path)
                             weights = glob.glob(
                                 "%s/*yolo-obj_final.weights" % path)

                             network, class_names, class_colors = darknet.load_network(
                            "yolo-obj_test.cfg",
                            labels[0],
                            weights[0],
                            1
                            )

                             image, detections2 = image_detection(
                                    img_path, network, class_names, class_colors, thresh_2
                                    )
                             
                             darknet.print_detections(
                                 detections2, "store_true")
                             for label2, confidence2, bbox2 in detections2:

                                center_x2 = bbox2[0]+bbox2[2]/2
                                center_y2 = bbox2[1]-bbox2[3]/2
                                
                                if(math.sqrt((center_x2-center_x)**2+(center_y2-center_y)**2) > 50):
                                        continue
                                stt_detc.append(
                                    [label2, "%.2f" % (float(confidence2)/100), ])



                        rslt_list.append([label,"%.2f"%(float(confidence)/100),stt_detc])

                         




                      
                    os.chdir(curdir)
                    print(rslt_list)

                

                    jsonmsgTemplate["Message"]=rslt_list
                    pusher_action_vision.send_json(jsonmsgTemplate)
                    out_q.put(1)




def thread_Controller(in_q1):
   

 
    while(True):

       
        
        if(not in_q1.empty()):
            data=in_q1.get()
            in_q1.task_done()

            if(data == 0):
                continue
            elif(data == 1):          
                detections,img_path=makePredictionsLabels()

                rslt_list = []
                
                for label, confidence, bbox in detections:
                    
                    stt_detc=makePredictionsStates(label,img_path,bbox)

                    
                    rslt_list.append([label,"%.2f"%(float(confidence)/100),stt_detc])
                jsonmsgTemplate["Message"]=rslt_list
                pusher_action_vision.send_json(jsonmsgTemplate)
        
        


def threadTerminator(in_q1,in_q2):
    global stop_thread 
    while(True):
        
        if(not in_q1.empty()):
            data=in_q1.get()
            in_q1.task_done()

            print(data)
            if(data == 0):
                continue
            elif(data == 1):

                print(data)
                stop_thread=False
                x = threading.Thread(
                            target=thread_Controller, args=(in_q2,))

                x.start()
                in_q2.join()
                print(data)
                while(True):

                    if(not in_q2.empty()):

                        data=in_q2.get()
                        in_q2.task_done()
                        if(data==11):

                             print('killing thread')
                             stop_thread=True

                             break

#q2=Queue()

initSockets()

x = threading.Thread(
            target=receiveInstructions, args=(q,))

x.start()


x = threading.Thread(target=thread_Controller, args=(q,))

x.start()


q.join()
#q2.join()
                
