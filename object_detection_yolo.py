# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

# Usage example:  python3 object_detection_yolo.py --video=run.mp4 --device 'cpu'
#                 python3 object_detection_yolo.py --video=run.mp4 --device 'gpu'
#                 python3 object_detection_yolo.py --image=office.jpg --device 'cpu'
#                 python3 object_detection_yolo.py --image=bird.jpg --device 'gpu'
import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import telebot
import datetime
import random
import bisect
import threading
import requests
import requests
import sqlite3
import sys
import MySQLdb
import psycopg2
import pyautogui
import win32gui
import time
from ftplib import FTP
import os
import calendar
print(calendar.day_abbr[datetime.date(2019, 2, 2).weekday()])
d_day_to_cat = {'Fri':'20211112', 'Sat':'20211113', 'Sun':'20211114', 'Mon': '20211115', 'Tue': '20211116', 'Wed': '20211117', 'Thu': '20211118'}
# Initialize the parameters
not_worked_time = 0
time0 = '0:0'
time1 = '0:0'
time0_flag = 0
time1_flag = 0
worked_time_v2 = 0
fl_newtime = 0
bot = telebot.TeleBot('2002045567:AAFBWxp3Fpxf9OdhRFm8HxCUMvAhuZVLwq4')
ch_id = ''
api_token = '2002045567:AAFBWxp3Fpxf9OdhRFm8HxCUMvAhuZVLwq4'
flag_of_sent_msg = 0
global ft
time_a = 0
time_t = 0
ftp = FTP('195.69.187.77', user='taxiuser', passwd='K2krJzFBEg9xxsz')


def send_telegram(text, tpe):
    token = '2002045567:AAFBWxp3Fpxf9OdhRFm8HxCUMvAhuZVLwq4'
    url = "https://api.telegram.org/bot"
    channel_id = "-1001668840613"
    url += token
    method = url + "/sendMessage"
    if tpe == 1:
        tx = 'Рабочее время: ' + str(text[0]) + ' часов, ' + str(text[1]) + ' минут.'
    elif tpe == 2:
        tx = 'Работал 1 человек' + str(text[0]) + ' часов, ' + str(text[1]) + ' минут.'
    else:
        tx = 'Работало более 1 человека' + str(text[0]) + ' часов, ' + str(text[1]) + ' минут.'
    r = requests.post(method, data={
         "chat_id": channel_id,
         "text": tx
          })

    if r.status_code != 200:
        raise Exception("post_text error")


def screenshot(window_title='InternetExplorer'):
    if window_title:
        hwnd = win32gui.FindWindow(None, window_title)
        if hwnd:
            win32gui.SetForegroundWindow(hwnd)
            x, y, x1, y1 = win32gui.GetClientRect(hwnd)
            x, y = win32gui.ClientToScreen(hwnd, (x, y))
            x1, y1 = win32gui.ClientToScreen(hwnd, (x1 - x, y1 - y))
            im = pyautogui.screenshot(region=(x, y, x1, y1))
            return im
        else:
            print('Window not found!')
    else:
        im = pyautogui.screenshot()
        return im


def gettime1(time):
    global time1
    global time1_flag
    global time0_flag
    global worked_time_v2
    time1 = time
    time1_flag = 1
    time0_flag = 0
    worked_time_v2 += 2
    print(worked_time_v2)


def gettime0(time):
    global time0
    global time1
    global time1_flag
    global time0_flag
    global not_worked_time
    if time0_flag == 0:
        time0 = time
        time0_flag = 1
        time1_flag = 0
        print(time0, time1)
        time_ms_1 = [int(i) for i in str(time1).split(':')[:2]]
        time_ms_0 = [int(i) for i in str(time0).split(':')[:2]]
        sum_time1 = (time_ms_1[0] * 60) + time_ms_1[1]
        sum_time0 = (time_ms_0[0] * 60) + time_ms_0[1]
        not_worked_time += sum_time0 - sum_time1
    print(str(time0).split(':'), str(time1).split(':'), not_worked_time)


def new_time_counter():
    global fl_newtime
    global worked_time_v2
    if fl_newtime == 0:
        worked_time_v2 += 2
        fl_newtime = 1


def time_alone():
    global time_a
    time_a += 2


def time_two():
    global time_t
    time_t += 2


def image_analis():
    confThreshold = 0.5  #Confidence threshold
    nmsThreshold = 0.4   #Non-maximum suppression threshold
    inpWidth = 416       #Width of network's input image
    inpHeight = 416      #Height of network's input image
    ms_label_list = []
    parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
    parser.add_argument('--device', default='cpu', help="Device to perform inference on 'cpu' or 'gpu'.")
    parser.add_argument('--image', help='office.jpg')
    parser.add_argument('--video', help='Path to video file.')
    args = parser.parse_args()
    # Load names of classes
    classesFile = "coco.names"
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    # Give the configuration and weight files for the model and load the network using them.
    modelConfiguration = "yolov3.cfg"
    modelWeights = "yolov3.weights"

    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

    if(args.device == 'cpu'):
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        print('Using CPU device.')
    elif(args.device == 'gpu'):
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        print('Using GPU device.')

    # Get the names of the output layers
    def getOutputsNames(net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Draw the predicted bounding box
    def drawPred(classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
        label = '%.2f' % conf
        # Get the label for the class name and its confidence
        if classes:
            assert(classId < len(classes))
            label = '%s:%s' % (classes[classId], label)
            ms_label_list.append(label.split(':')[0])
        #Display the label at the top of the bounding box
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

    # Remove the bounding boxes with low confidence using non-maxima suppression
    def postprocess(frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

    # Process inputs
    winName = 'Deep learning object detection in OpenCV'
    cv.namedWindow(winName, cv.WINDOW_NORMAL)

    outputFile = "yolo_out_py.avi"
    if (True):
        # Open the image file
        if not os.path.isfile('office.jpg'):
            print("Input image file ", args.image, " doesn't exist")
            sys.exit(1)
        cap = cv.VideoCapture('office.jpg')
        outputFile = 'off' + '_yolo_out_py.jpg'
    """elif (args.video):
        # Open the video file
        if not os.path.isfile(args.video):
            print("Input video file ", args.video, " doesn't exist")
            sys.exit(1)
        cap = cv.VideoCapture(args.video)
        outputFile = args.video[:-4]+'_yolo_out_py.avi'
    else:
        # Webcam input
        cap = cv.VideoCapture(0)"""

    # Get the video writer initialized to save the output video
    #if (not args.image):
    #    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

    while cv.waitKey(1) < 0:

        # get frame from the video
        hasFrame, frame = cap.read()

        # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing !!!")
            print("Output file is stored as ", outputFile)
            cv.waitKey(3000)
            # Release device
            cap.release()
            break

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))

        # Remove the bounding boxes with low confidence
        postprocess(frame, outs)

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # Write the frame with the detection boxes
        if (True):
            cv.imwrite(outputFile, frame.astype(np.uint8))
        else:
            vid_writer.write(frame.astype(np.uint8))

        cv.imshow(winName, frame)
    print(ms_label_list)
    is_two = (ms_label_list.count('person') >= 2)
    is_one = (1 == ms_label_list.count('person'))
    if is_one:
        return 1
    elif is_two:
        return 2
    return 0

image_analis()
b_time_sender = []
lst = ftp.nlst('20211114')
print('llll', len(lst))
while True:
    dt_now = datetime.datetime.now()
    m_d = str(dt_now).split(' ')[0].split('-')
    if ''.join(str(datetime.datetime.now().time()).split(':')[:2]) == '0801':
        flag_of_sent_msg = 0
    if ''.join(str(datetime.datetime.now().time()).split(':')[:2]) == '0800':
        print('вход')
        worked_time = 1440 - not_worked_time
        actual_time_hours = str(worked_time_v2 // 60)
        actual_time_minutes = str(worked_time_v2 % 60)
        time_a_hours = str(time_a // 60)
        time_a_minutes = str(time_a % 60)
        time_t_hours = str(time_t // 60)
        time_t_minutes = str(time_t % 60)
        b_time_sender = [actual_time_hours, actual_time_minutes]
        b_time_sender_a = [time_a_hours, time_a_minutes]
        b_time_sender_t = [time_t_hours, time_t_minutes]
        b_dop_time_sender = []
        if flag_of_sent_msg == 0:
            send_telegram(b_time_sender, 1)
            send_telegram(b_time_sender_a, 2)
            send_telegram(b_time_sender_t, 3)
        flag_of_sent_msg = 1
        not_worked_time = 0
        time0 = '0:0'
        time1 = '0:0'
        time0_flag = '0:0'
        time1_flag = '0:0'
        worked_time_v2 = 0
    if int(''.join(str(datetime.datetime.now().time()).split(':')[1])) % 2 == 1:
        fl_newtime = 0
    if int(''.join(str(datetime.datetime.now().time()).split(':')[1])) % 2 == 0:
        print(worked_time_v2)
        out = 'office.jpg'
        print("получил изображение")
        lst = ftp.nlst(d_day_to_cat[calendar.day_abbr[datetime.date(int(m_d[0]), int(m_d[1]), int(m_d[2])).weekday()]])
        print(lst[-1], d_day_to_cat[calendar.day_abbr[datetime.date(int(m_d[0]), int(m_d[1]), int(m_d[2])).weekday()]])
        with open(out, 'wb') as f:
            ftp.retrbinary('RETR ' + f'{lst[-1]}', f.write)
        result = image_analis()
        if result != 0:
            if result == 1:
                time_alone()
            if result == 2:
                time_two()
            print("попало в 1")
            new_time_counter()
        elif result == 0:
            print('прошло в 0')
            gettime0(datetime.datetime.now().time())
