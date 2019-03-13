# Person detection with alerts based on Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

import numpy as np
import tensorflow as tf
import cv2
import time
import sys
import os 
import msvcrt
import logging
import shutil
import config
from pushbullet import Pushbullet
from threading import Thread

class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt
        config = tf.ConfigProto(
                device_count = {'GPU': 1}
            )

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(config=config, graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})


        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()
        
# Class for the seperate thread that will grab frames from the camera (this is much faster than single threading)

class FrameGrab:

    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.cap.read()
    
    def start(self):
        Thread(target=self.get, daemon = True, args=()).start()
        return self
        
    # This is the part that actually grabs the frames in loop, I've kept an extra grab (without read) to pull one frame from buffer so we are always within 1-2 frames of most current. You may need to add more (or remove) the entry if youre using a diff fps or resolution (which could cause each frame grab to take longer)
    def get(self):
        while True:
            #self.cap.grab() On second thought, this is seeming to not be benificial when using multithreading, commenting out for now, will test further.
            (self.grabbed, self.frame) = self.cap.read()               
    
def analyzeframe(img, boxes, scores, classes, num):
    humandetected = 0
    for i in range(len(boxes)):
        if classes[i] == 1 and scores[i] > threshold:
            box = boxes[i]            
            yeval = int(box[2])
            y2eval = int(box[0])
            scoreint = str(round(scores[i], 3))
            # This is a filter to determine if any vertex of the bounding box for a detected person is within the part of the image I care about ie: not across the street (remember opencv uses rows, so top is 0)
            if (yeval > 150) or (y2eval > 150):
                # Draws the box, puts a confidence score under it, and alerts that we have detected a human
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(0,0,255),2)
                cv2.putText(img,scoreint,(box[1]+5,box[2]+25),cv2.FONT_HERSHEY_DUPLEX,0.7,(0,0,255),1,cv2.LINE_AA)
                humandetected = 1
            else: 
                # Still does the above, but no alert and its in white
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(211,211,211),2)
                cv2.putText(img,scoreint,(box[1]+5,box[2]+25),cv2.FONT_HERSHEY_DUPLEX,0.7,(211,211,211),1,cv2.LINE_AA)    
    return img, humandetected
    
def humanevent(img, timeelap, dirtime, pb, pbipcam_channel, timebetweenevents, pbenabled, url, yord):
    # Check to see how long its been since the last person detected, this avoids a new entry/notification for people hanging around. Tweak as needed
    if timeelap > timebetweenevents:
        # This process builds a directory structure based on time since epoch + driveway/frontyard and then writes the image into that directory as well as a php file to display the images. It also pushes the notification and writes to our log.
        dirtime = str(round(time.time()))
        os.mkdir('output/' + yord + dirtime)
        cv2.imwrite('output/' + yord + dirtime + '/' + dirtime + yord + '.jpg', img)
        shutil.copy2('showimgs.php', 'output/' + yord + dirtime)
        if pbenabled == 1:
            pb.push_link("Person Detected " + yord, url + yord + dirtime + "/showimgs.php", channel=pbipcam_channel)
        logging.warning('<a href="detectimgs/' + yord + dirtime + '/showimgs.php" target="_blank">Person detected ' + yord + '</a>') 
        lastlogtime = int(round(time.time()))
    else:
        # If were still detecting people, but its within the elapse interval, well write the frame to the same directory as before.
        linktime = str(round(time.time()))
        cv2.imwrite('output/' + yord + dirtime + '/' + linktime + yord + '.jpg', img)
        lastlogtime = int(round(time.time()))
    return lastlogtime, dirtime
    

if __name__ == "__main__":
    model_path = 'faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
    threshold = config.threshold
    timebetweenevents = config.timebetweenevents
    coloraftertime = config.coloraftertime
    odapi = DetectorAPI(path_to_ckpt=model_path)
    # Setup and spawn the capture threads
    capurl = config.capurl
    capdrive = FrameGrab(capurl).start()
    # Setup the logfile format to look pretty when displayed on a webpage
    logging.basicConfig(filename='output/detect.log', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    # Setup for pushbullet if enabled, you need to install the pushbullet py from here if you want to use this for push notifications - https://github.com/rbrcsk/pushbullet.py
    pbenabled = config.pbenabled
    if pbenabled == 1:
        pb = Pushbullet(config.pbapikey)
        pbipcam_channel = pb.get_channel(config.pbchannelname)
    # Set some initial values
    url = str(config.url)
    lastlogtimedr = int(round(time.time()))
    drtext = str("driveway")
    dirtimedr = "0"
    
    while True:
        start_time = time.time()
        # Capture frames
        img = capdrive.frame
        timestamp = time.ctime()
        # Feed them to tensorflow inference 
        boxes, scores, classes, num = odapi.processFrame(img)
        # Determine if a human was detected and draw a bounding box if so along with scores
        imgoutdrive, humandetectdrive = analyzeframe(img, boxes, scores, classes, num)   
        curtime = int(round(time.time()))
        timeelapdrive = curtime-lastlogtimedr
        # Add timestamps
        cv2.putText(imgoutdrive,timestamp,(20, 23),cv2.FONT_HERSHEY_DUPLEX,0.6,(211,211,211),1,cv2.LINE_AA)
        # If a person was detected in the frame, run a function that creates a display directory, saves the frame, sends us a push notification, and writes to the log.
        if humandetectdrive == 1:
            lastlogtimedr, dirtimedr = humanevent(imgoutdrive, timeelapdrive, dirtimedr, pb, pbipcam_channel, timebetweenevents, pbenabled, url, drtext)
        # Resize the images
        imgresizedr = cv2.resize(imgoutdrive,(375, 245))
        # Write the images to be served by an Apache server
        cv2.imwrite('output/RD/drivewaytmp.jpg', imgresizedr, [cv2.IMWRITE_JPEG_PROGRESSIVE, 1,cv2.IMWRITE_JPEG_QUALITY, 80])
        # Color the log background based on the time since last event
        if timeelapdrive < coloraftertime:
            colorf = open("output/color.txt", "w")
            colorf.write("#720808;")
        else:
            colorf = open("output/color.txt", "w")
            colorf.write("#1f1f1f;")
        print("Time since last driveway log:", timeelapdrive)
        sys.stdout.flush()
        end_time = time.time()
        # Here we determine how long its taken to process the frame(s) and then add some sleep to maintain the desired framerate (This helps with the output timing so the browser hits blank images MUCH less often) as well it reduces uneeded system load.
        processtime = end_time-start_time
        frametime = config.frametime-.025
        if processtime < frametime:
            sleeptime = frametime-processtime
            time.sleep(sleeptime)
        # Here we copy the images to a second file after the sleep. This provides redundancy incase the browser tries to grab the image while its being written, we can provide a second image location for "onerror". (The reason why you see the "-.010" above is to account for the copy time. You might need to tweak for your system.
        shutil.copyfile('output/RD/drivewaytmp.jpg', 'output/RD/driveway.jpg')
        final_time = time.time()
        print("Elapsed Time:", final_time-start_time)
