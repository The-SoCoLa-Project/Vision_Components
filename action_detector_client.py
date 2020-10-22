config_file_path = "./configs/pretrained/config_model1.json"
import os
import cv2
import sys
import importlib
import torch
import torchvision
import numpy as np
import zmq
import json
import threading
import io
import base64
import re

#from config import *

from queue import Queue



from data_parser import WebmDataset
from data_loader_av import VideoFolder

from models.multi_column import MultiColumn
from transforms_video import *

from utils import load_json_config, remove_module_from_checkpoint_state_dict
from pprint import pprint



global ip_Controller

global ip_client_Camera


global puller

global puller_video

global puller2

global puller2

global pusher_action_vision

#ip_client_vision='localhost'


try:
    ip_Controller
except NameError:
    ip_Controller='localhost'


try:
    ip_client_Camera
except NameError:
    ip_client_Camera='localhost'




if(ip_Controller is None and len(sys.argv)>1):
    if(sys.argv[1]!=None and sys.argv[1]!='-1'):

        ip_Controller=sys.argv[1]


if(ip_client_Camera is None and len(sys.argv)>2):
    if(sys.argv[2]!=None and sys.argv[2]!='-1'):

        ip_client_Camera=sys.argv[1]



jsonmsgTemplate = {
            'Sender': "Vision",
            'Source': "Vision",
            'Component': "Action Recognizer",
            'SessionId': "-",
            'Message': "",
        }


# Load config file
config = load_json_config(config_file_path)

# set column model
column_cnn_def = importlib.import_module("{}".format(config['conv_model']))
model_name = config["model_name"]

print("=> Name of the model -- {}".format(model_name))

# checkpoint path to a trained model
checkpoint_path = os.path.join("./", config["output_dir"], config["model_name"], "model_best.pth.tar")
print("=> Checkpoint path --> {}".format(checkpoint_path))


model = MultiColumn(config['num_classes'], column_cnn_def.Model, int(config["column_units"]))
model.eval()


print("=> loading checkpoint")
checkpoint = torch.load(checkpoint_path)
checkpoint['state_dict'] = remove_module_from_checkpoint_state_dict(
                                              checkpoint['state_dict'])
model.load_state_dict(checkpoint['state_dict'])
print("=> loaded checkpoint '{}' (epoch {})"
      .format(checkpoint_path, checkpoint['epoch']))



# Center crop videos during evaluation
transform_eval_pre = ComposeMix([
        [Scale(config['input_spatial_size']), "img"],
        [torchvision.transforms.ToPILImage(), "img"],
        [torchvision.transforms.CenterCrop(config['input_spatial_size']), "img"]
         ])

transform_post = ComposeMix([
        [torchvision.transforms.ToTensor(), "img"],
        [torchvision.transforms.Normalize(
                   mean=[0.485, 0.456, 0.406],  # default values for imagenet
                   std=[0.229, 0.224, 0.225]), "img"]
         ])

val_data = VideoFolder(root=config['data_folder'],
                       json_file_input=config['json_data_val'],
                       json_file_labels=config['json_file_labels'],
                       clip_size=config['clip_size'],
                       nclips=config['nclips_val'],
                       step_size=config['step_size_val'],
                       is_val=True,
                       transform_pre=transform_eval_pre,
                       transform_post=transform_post,
                       get_item_id=True,
                       )
dict_two_way = val_data.classes_dict


def initSockets():


    global ip_Controller

    global ip_client_Camera


    global puller

    global puller_video

    global puller2

    global puller2

    global pusher_action_vision

    context = zmq.Context()
    puller = context.socket(zmq.PULL)
    puller.connect("tcp://%s:%s" %(ip_client_Camera, 6100))


    puller_video = context.socket(zmq.SUB)
    puller_video.bind("tcp://*:%s" % 6101)



    pusher_action_vision = context.socket(zmq.PUSH)
    pusher_action_vision.bind("tcp://*:%s" % 6102)



    puller2 = context.socket(zmq.PULL)
    puller.connect("tcp://%s:%s" % (ip_Controller, 6300))






    pusher2 = context.socket(zmq.PUSH)
    pusher2.bind("tcp://*:%s" % 6300)






global SessionId

SessionId='test'


def receiveInstructions(out_q1,out_q2):
        global SessionId
   
  
        while(True):
            message = puller.recv_string()

            if('start recording' in message):
                
                m = re.search('(?<=SessionId:)\w+', message)
                SessionId=m.group(0)
               
                

                out_q1.put(0) 
                #out_q2.put(0)
               # out_q.put(stop_recording) 
            if(message=='stop recording'):
                
                out_q1.put(1) 
          




def recordVideo(in_q,outq):
    
    
    
        global SessionId
        out = cv2.VideoWriter('%s.webm'%SessionId,  cv2.VideoWriter_fourcc(*'VP80'), 12.0, (640,480))
        puller_video.setsockopt_string(zmq.SUBSCRIBE, str(''))
        while(True):
            if(not in_q.empty()):
                data = in_q.get()
                in_q.task_done()
                if(data==0):

                    
                    while(in_q.empty()):
                    
                   
                    
                   
                        frame = puller_video.recv_string()
                        img = base64.b64decode(frame)
                        npimg = np.fromstring(img, dtype=np.uint8)
                        source = cv2.imdecode(npimg, 1)
                        out.write(source)
                    #cv2.imshow("Stream", source)
                    #cv2.waitKey(1)
                    data = in_q.get()
                    in_q.task_done()
                    if(data==1):

                        outq.put(1)
                                                   
                


initSockets()
q1 = Queue() 
q2 = Queue() 
x = threading.Thread(
            target=receiveInstructions, args=(q1,q2,))

x.start()

 
x2 = threading.Thread(
            target=recordVideo, args=(q1,q2,))

x2.start()




from skvideo.io import FFmpegReader


def produce_Results(in_q):
    
    while(True):
        
        if(not in_q.empty()):
            data = in_q.get()
            in_q.task_done()
           
            if(data==1):
         
           
            #img = base64.b64decode(frame)
            #npimg = np.fromstring(img, dtype=np.uint8)
            #source = cv2.imdecode(npimg, 1)    
            #out.write(source)

            #video_path = base64.b64decode(msg)
            #npimg = np.fromstring(img, dtype=np.uint8)
            #source = cv2.imdecode(npimg, 1)

            #stop_recording=False
            #out.release()
            #cv2.destroyAllWindows()
            
                reader = FFmpegReader('test.webm', inputdict={}, outputdict={})

                imgs = []
                for img in reader.nextFrame():
                    imgs.append(img)
                    
                    
                imgs = transform_eval_pre(imgs)
                imgs = transform_post(imgs)
                #num_frames = len(imgs)
                #offset = 0
                #imgs = imgs[offset: num_frames_necessary + offset: self.step_size]
                #imgs=imgs[0]
                #print(imgs[0])
                data = torch.stack(imgs)
                data = data.permute(1, 0, 2, 3)
                input_data=data

                input_data = input_data.unsqueeze(0)


                if config['nclips_val'] > 1:
                    input_var = list(input_data.split(config['clip_size'], 2))
                    for idx, inp in enumerate(input_var):
                        input_var[idx] = torch.autograd.Variable(inp)
                else:
                    input_var = [torch.autograd.Variable(input_data)]


                input_var = [torch.autograd.Variable(input_data)]
                output = model(input_var).squeeze(0)
                output = torch.nn.functional.softmax(output, dim=0)
                # compute top5 predictions
                pred_prob, pred_top5 = output.data.topk(5)
                pred_prob = pred_prob.numpy()
                pred_top5 = pred_top5.numpy()

               # print("Id of the video sample = {}".format(item_id))
                #print("True label --> {} ({})".format(target, dict_two_way[target]))
                print("\nTop-5 Predictions:")

                results=[]

                for i, pred in enumerate(pred_top5):
                    print("Top {} :== {}. Prob := {:.2f}%".format(i + 1, dict_two_way[pred], pred_prob[i] * 100))
                    results.append([dict_two_way[pred],"%.2f"%(pred_prob[i])])

                jsonmsgTemplate["Message"]=results
                pusher_action_vision.send_json(jsonmsgTemplate)

               


x3 = threading.Thread(
            target=produce_Results, args=(q1,))

x3.start()

q1.join()
q2.join()

