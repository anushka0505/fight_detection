# For Youtube Download.
import io 
from pytube import YouTube
from IPython.display import HTML
from base64 import b64encode
from datetime import datetime

import os
import cv2
import time
import copy
import glob
import torch
import argparse
import statistics
import threading
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
# from moviepy.editor import *
import albumentations as A
from collections import deque
#from google.colab.patches import cv2_imshow

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATASET_DIR = '/content/Fight_Detection_From_Surveillance_Cameras-PyTorch_Project/dataset'
CLASSES_LIST = ['fight','noFight']
SEQUENCE_LENGTH = 16
predicted_class_name = ""

##used
# Define the transforms
def transform_():
    transform = A.Compose(
    [A.Resize(128, 171, always_apply=True),A.CenterCrop(112, 112, always_apply=True),
     A.Normalize(mean = [0.43216, 0.394666, 0.37645],std = [0.22803, 0.22145, 0.216989], always_apply=True)]
     )
    return transform

##used
def loadModel(modelPath):
  PATH=modelPath
  model_ft = torchvision.models.video.mc3_18(pretrained=True, progress=False)
  num_ftrs = model_ft.fc.in_features         #in_features
  model_ft.fc = torch.nn.Linear(num_ftrs, 2) #nn.Linear(in_features, out_features)
  model_ft.load_state_dict(torch.load(PATH,map_location=torch.device(device)))
  model_ft.to(device)
  model_ft.eval()
  return model_ft
##used
def PredTopKClass(k, clips, model):
  with torch.no_grad(): # we do not want to backprop any gradients

      input_frames = np.array(clips)
      
      # add an extra dimension        
      input_frames = np.expand_dims(input_frames, axis=0)

      # transpose to get [1, 3, num_clips, height, width]
      input_frames = np.transpose(input_frames, (0, 4, 1, 2, 3))

      # convert the frames to tensor
      input_frames = torch.tensor(input_frames, dtype=torch.float32)
      input_frames = input_frames.to(device)

      # forward pass to get the predictions
      outputs = model(input_frames)

      # get the prediction index
      soft_max = torch.nn.Softmax(dim=1)  
      probs = soft_max(outputs.data) 
      prob, indices = torch.topk(probs, k)

  Top_k = indices[0]
  Classes_nameTop_k=[CLASSES_LIST[item].strip() for item in Top_k]
  ProbTop_k=prob[0].tolist()
  ProbTop_k = [round(elem, 5) for elem in ProbTop_k]
  return Classes_nameTop_k[0]    #list(zip(Classes_nameTop_k,ProbTop_k))

##used in st
def PredTopKProb(k,clips,model):
  with torch.no_grad(): # we do not want to backprop any gradients

      input_frames = np.array(clips)
      
      # add an extra dimension        
      input_frames = np.expand_dims(input_frames, axis=0)

      # transpose to get [1, 3, num_clips, height, width]
      input_frames = np.transpose(input_frames, (0, 4, 1, 2, 3))

      # convert the frames to tensor
      input_frames = torch.tensor(input_frames, dtype=torch.float32)
      input_frames = input_frames.to(device)

      # forward pass to get the predictions
      outputs = model(input_frames)

      # get the prediction index
      soft_max = torch.nn.Softmax(dim=1)  
      probs = soft_max(outputs.data) 
      prob, indices = torch.topk(probs, k)

  Top_k = indices[0]
  Classes_nameTop_k=[CLASSES_LIST[item].strip() for item in Top_k]
  ProbTop_k=prob[0].tolist()
  ProbTop_k = [round(elem, 5) for elem in ProbTop_k]
  return list(zip(Classes_nameTop_k,ProbTop_k))

###used
def predict_on_video(video_file_path, output_folder_path, model, SEQUENCE_LENGTH,skip=2,showInfo=False):
    '''
    This function will perform action recognition on a video using the LRCN model.
    Args:
    video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
    output_file_path: The path where the ouput video with the predicted action being performed overlayed will be stored.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    '''

    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)

    # Get the width, height and fps of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_reader.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")

    # check if the output folder exits or not
    # if it does'nt, then make the folder
    alert_folder_check(output_folder_path)

    # output video path inside the output folder
    output_video_path = f"{output_folder_path}/Output_video.mp4"

    # Initialize the VideoWriter Object to store the output video in the disk.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 
                                   video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

    # Declare a queue to store video frames.
    frames_queue = deque(maxlen = SEQUENCE_LENGTH)
    transform= transform_()
    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    # Iterate until the video is accessed successfully.
    counter=0
    s_no = 1
    while video_reader.isOpened():

        # Read the frame.
        ok, frame = video_reader.read()
        
        # Check if frame is not read properly then break the loop.
        if not ok:
            break

        image = frame.copy()
        framee = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        framee = transform(image=framee)['image']
        if counter % skip==0:
          # Appending the pre-processed frame into the frames list.
          frames_queue.append(framee)
         
        # changing the predicted class name to blank before the prediction
        # this will make sure to only print the label on the first frame of the bunch
        # predicted_class_name = ''

        # Check if the number of frames in the queue are equal to the fixed sequence length.
        if len(frames_queue) == SEQUENCE_LENGTH:
            predicted_class_name= PredTopKClass(1,frames_queue, model)
            if showInfo:
                print(predicted_class_name)

            # checking if the bunch has "fight" as the predicted class 
            if predicted_class_name=="fight":

                # print the label on the last frame
                cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                # save the last frame where "fight" label is detected
                # and also add the timestamp and other info in the cvs file
                save_alert_image_csv(frame, s_no, output_folder_path)
            
            # reset the queue
            frames_queue = deque(maxlen = SEQUENCE_LENGTH)
    
        # Write predicted class name on top of the frame.
        if predicted_class_name=="fight":
            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # uncomment the below line if we want to print "no fight" label on the frames
        # else:
        #     cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        counter+=1
        
        # Write The frame into the disk using the VideoWriter Object.
        video_writer.write(frame)
        # time.sleep(2)
    if showInfo:
        print(f"Counter: {counter}")
    # Release the VideoCapture and VideoWriter objects.
    video_reader.release()
    video_writer.release()
##used
def alert_folder_check(path_):
    '''
    This function will perform a check if the folder path mentioned exists or not and if it does not the make the
    folder.
    Args:
    path_:  The path of the folder stored in the disk on where the output is supposed to be saved.
    '''

    # Check if the folder exists, and if not, create it
    if not os.path.exists(path_):
        os.makedirs(path_)
        print(f"Folder '{path_}' created.")
    else:
        print(f"Folder '{path_}' already exists.")
        # pass
##used
def save_alert_image_csv(frame, s_no, path_):
    '''
    This function will save the alert images in a folder and save the alert info in a csv file in the alert folder.
    Args:
    frame: The alert frame which on which alert is raised.
    s_no: The counter to serialise the alerts in the csv file.
    path_:  The path of the folder stored in the disk on where the output is supposed to be saved.
    '''

    # get the current time 
    now = datetime.now()
    timestamp = now.strftime("%Y-%B-%d_%H-%M-%S.%f")
    
    # Alert Image
    # image path
    image_path = f"{path_}/{timestamp}.jpg"

    # Save the image
    cv2.imwrite(image_path, frame)

    # CSV File
    # csv file path
    csv_file_path = f"{path_}/Report.csv"

    # column details to save
    # serial no, alert image name, time stamp and detection in a csv file
    columns = ["S_No", "Image_Name", "Time_stamp", "Feature"]

    try:
        # Try to read the existing CSV file into a DataFrame
        if os.path.isfile(csv_file_path):
            df = pd.read_csv(csv_file_path)
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        # If the file is not found, create a new DataFrame with the specified columns
        df = pd.DataFrame(columns=columns)

    # save the details in the csv column
    new_data = {"S_No": s_no, "Image_Name": timestamp, "Time_stamp": timestamp, "Feature": "Fight"}

    # increase the serial number counter
    s_no+=1

    # Convert new_data to a DataFrame
    new_data_df = pd.DataFrame([new_data])

    # Concatenate the new data with the existing DataFrame
    df = pd.concat([df, new_data_df], ignore_index=True)

    # Save the updated DataFrame back to the CSV file
    df.to_csv(csv_file_path, index=False)

##used in st
def streaming_framesInference(frames, model):
    clips = []
    transform = transform_()
    for frame in frames:
        image = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(image=frame)['image']

        # Append the normalized frame into the frames list
        clips.append(frame)
    first = PredTopKClass(1, clips, model)
    print(first)
    print(PredTopKProb(2, clips, model))
    return first
##used in st
def streaming_predict(frames, model):
    prediction = streaming_framesInference(frames, model)
    global predicted_class_name
    predicted_class_name = prediction

##used in st
def start_streaming(model,streamingPath):
    video = cv2.VideoCapture(streamingPath)
    l = []
    last_time = time.time() - 3
    while True:
        _, frame = video.read()
        if last_time+2.5 < time.time():
            l.append(frame)
        if len(l) == 16:
            last_time = time.time()
            x = threading.Thread(target=streaming_predict, args=(l,model))
            x.start()
            l = []
        if predicted_class_name == "fight":
            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        # else:
        #     cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("RTSP", frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
