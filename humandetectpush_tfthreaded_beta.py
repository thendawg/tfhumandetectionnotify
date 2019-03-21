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
from queue import Queue, LifoQueue

class DetectorAPI:
    def __init__(self, model_path):
        self.path_to_ckpt = model_path
        config = tf.ConfigProto(
                device_count = {'GPU': 1}
            )
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.3
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
    
    def processFrame(self, q, q_img):
        qcount = 0
        while True:
            # Get a fresh frame each time processFrame executes then clear the queue after 20 iterations - this prevents too much memory from being consumed.
            self.image = q_img.get()
            q_img.task_done()
            if qcount > 20:
                q_img.task_done()
                q_img.mutex.acquire()
                q_img.queue.clear()
                q_img.all_tasks_done.notify_all()
                q_img.unfinished_tasks = 0
                q_img.mutex.release()
                qcount = 0
            else:
                qcount += 1
            # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(self.image, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_np_expanded})


            im_height, im_width,_ = self.image.shape
            boxes_list = [None for i in range(boxes.shape[1])]
            for i in range(boxes.shape[1]):
                boxes_list[i] = (int(boxes[0,i,0] * im_height),
                            int(boxes[0,i,1]*im_width),
                            int(boxes[0,i,2] * im_height),
                            int(boxes[0,i,3]*im_width))

            q.put((self.image, boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])))

    def close(self):
        self.sess.close()
        self.default_graph.close()
        
# Class for the seperate thread that will grab frames from the camera (this is much faster than single threading)

class FrameGrab:

    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.cap.read()
        
    # This is the part that actually grabs the frames in loop and puts them in queue
    def get(self, q_img):
        while True:
            (self.grabbed, self.frame) = self.cap.read()    
            q_img.put(self.frame)
                
            
    
def analyzeframe(img, boxes, scores, classes, num, hfilter, fsize):
    humandetected = 0
    alertpb = 0
    # For function to iterate through all detection boxes and draw those that are over the cutoff and correct class (1 - Human) 
    for i in range(len(boxes)):
        if classes[i] == 1 and scores[i] > threshold:
            box = boxes[i]            
            yeval = int(box[2])
            y2eval = int(box[0])
            scoreint = str(round(scores[i], 3))
            # *This is now set via config* This is a filter to determine if any vertex of the bounding box for a detected person is within the part of the image I care about ie: not across the street (remember opencv uses rows, so top is 0) Basically if you set this value to say 100, then at least some part of the detected object must be outside fo the top 100 rows/pixels in the image. 
            if (yeval > hfilter) or (y2eval > hfilter):
                # Draws the box, puts a confidence score under it, and alerts that we have detected a human
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(0,0,255),2)
                cv2.putText(img,scoreint,(box[1]+5,box[2]+25),cv2.FONT_HERSHEY_DUPLEX,fsize,(0,0,255),1,cv2.LINE_AA)
                humandetected = 1
            else: 
                # Still does the above, but no alert and its in white
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(211,211,211),2)
                cv2.putText(img,scoreint,(box[1]+5,box[2]+25),cv2.FONT_HERSHEY_DUPLEX,fsize,(211,211,211),1,cv2.LINE_AA)
    # A seperate for function to evaluate if scores of bounding boxes detected meet the second (higher) threshold for a pb notification.
    for i in range(len(boxes)):
        if classes[i] == 1 and scores[i] > thresholdpb:
            yeval = int(box[2])
            y2eval = int(box[0])
            if (yeval > hfilter) or (y2eval > hfilter):
                alertpb = 1                
    return img, humandetected, alertpb
    
def humanevent(img, timeelap, dirtime, pb, pbipcam_channel, timebetweenevents, pbenabled, url, maindir, alertpb, yord, timeelappb, lastpbtime):
    # Check to see how long its been since the last person detected, this avoids a new entry/notification for people hanging around. Tweak as needed       
    if timeelap > timebetweenevents:
        # This process builds a directory structure based on time since epoch + driveway/frontyard and then writes the image into that directory as well as a php file to display the images. It also pushes the notification and writes to our log.
        dirtime = str(round(time.time()))
        os.mkdir(maindir + yord + dirtime)
        cv2.imwrite(maindir + yord + dirtime + '/' + dirtime + yord + '.jpg', img)
        shutil.copy2('showimgs.php', maindir + yord + dirtime)
        logging.warning('<a href="' + url + yord + dirtime + '/showimgs.php" target="_blank">Person detected ' + yord + '</a>') 
        lastlogtime = int(round(time.time()))
    else:
        # If were still detecting people, but its within the elapse interval, well write the frame to the same directory as before.
        linktime = str(round(time.time()))
        cv2.imwrite(maindir + yord + dirtime + '/' + linktime + yord + '.jpg', img)
        lastlogtime = int(round(time.time()))
    # Seperately evaluate the PushBullet function, this requires the higher score threshold to be met and tracks the time since last pb push seperate from the last log entry.
    if (timeelappb > timebetweenevents) and (pbenabled == 1) and (alertpb == 1):
        pb.push_link("Person Detected " + yord, url + yord + dirtime + "/showimgs.php", channel=pbipcam_channel)
        lastpbtime = int(round(time.time()))
    elif (pbenabled == 1) and (alertpb == 1):
        lastpbtime = int(round(time.time()))
    return lastlogtime, dirtime, lastpbtime
    

if __name__ == "__main__":
    # Set some initial values
    model_path = str(config.model_path)
    threshold = config.threshold
    thresholdpb = config.thresholdpb
    timebetweenevents = config.timebetweenevents
    coloraftertime = config.coloraftertime
    hfilterdr = config.hfilterdr
    hfilteryd = config.hfilteryd
    maindir = str(config.maindir)
    livedir = str(config.livedir)
    fsize = config.fontsize
    hsizeout = config.imgwidth
    vsizeout = config.imgheight
    url = str(config.url)
    lastlogtimedr = int(round(time.time()))
    lastlogtimeyd = int(round(time.time()))
    lastpbtimedr = int(round(time.time()))
    lastpbtimeyd = int(round(time.time()))
    drtext = str(config.cam1text)
    ydtext = str(config.cam2text)
    dirtimedr = "0"
    dirtimeyd = "0"
    frametime = config.frametime-.020
    # Setup the logfile format to look pretty when displayed on a webpage
    logging.basicConfig(filename='output/detect.log', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    # Setup for pushbullet if enabled, you need to install the pushbullet py from here if you want to use this for push notifications - https://github.com/rbrcsk/pushbullet.py
    pbenabled = config.pbenabled
    if pbenabled == 1:
        pb = Pushbullet(config.pbapikey)
        pbipcam_channel = pb.get_channel(config.pbchannelname)
    # Setup the capture threads and queues
    q_d_img = LifoQueue()
    q_y_img = LifoQueue()
    capurl = config.capurl
    capurl2 = config.capurl2
    capdt = Thread(name='capdrivet', target=FrameGrab(capurl).get, daemon = True, args=(q_d_img,))
    capyt = Thread(name='capyardt', target=FrameGrab(capurl2).get, daemon = True, args=(q_y_img,))
    capdt.start()
    capyt.start()
    # Build a queue and start a seperate thread for each stream being analyzed - the maxsize of 1 for the queue insures that TF cant outrun the image processing/output. 
    q_d = Queue(maxsize=1)
    q_y = Queue(maxsize=1)
    drivet = Thread(name='adrivet', target=DetectorAPI(model_path).processFrame, daemon = True, args=(q_d, q_d_img))
    yardt = Thread(name='ayardt', target=DetectorAPI(model_path).processFrame, daemon = True, args=(q_y, q_y_img))
    drivet.start()
    yardt.start()
    
    while True:
        # Get timestamps, start of process time
        start_time = time.time()
        timestamp = time.ctime()
        # Retrieve TF results from queues
        img, boxes, scores, classes, num = q_d.get()
        q_d.task_done()
        img2, boxes2, scores2, classes2, num2 = q_y.get()
        q_y.task_done()
        # Determine if a human was detected and draw a bounding box if so along with scores
        imgoutdrive, humandetectdrive, alertpbdr = analyzeframe(img, boxes, scores, classes, num, hfilterdr, fsize)
        imgoutyard, humandetectyard, alertpbyd = analyzeframe(img2, boxes2, scores2, classes2, num2, hfilteryd, fsize)         
        # Determine time since last PB and log entry events
        curtime = int(round(time.time()))
        timeelapdrive = curtime-lastlogtimedr
        timeelapyard = curtime-lastlogtimeyd
        timeelappbdr = curtime-lastpbtimedr
        timeelappbyd = curtime-lastpbtimeyd
        # Add timestamps
        cv2.putText(imgoutdrive,timestamp,(20, 23),cv2.FONT_HERSHEY_DUPLEX,fsize,(211,211,211),1,cv2.LINE_AA)
        cv2.putText(imgoutyard,timestamp,(20, 23),cv2.FONT_HERSHEY_DUPLEX,fsize,(211,211,211),1,cv2.LINE_AA)
        # If a person was detected in the frame, run a function that creates a display directory, saves the frame, sends us a push notification, and writes to the log.
        if humandetectdrive == 1:
            lastlogtimedr, dirtimedr, lastpbtimedr = humanevent(imgoutdrive, timeelapdrive, dirtimedr, pb, pbipcam_channel, timebetweenevents, pbenabled, url, maindir, alertpbdr, drtext, timeelappbdr, lastpbtimedr)
        if humandetectyard == 1:
            lastlogtimeyd, dirtimeyd, lastpbtimeyd = humanevent(imgoutyard, timeelapyard, dirtimeyd, pb, pbipcam_channel, timebetweenevents, pbenabled, url, maindir, alertpbyd, ydtext, timeelappbyd, lastpbtimeyd)
        # Resize the images
        imgresizedr = cv2.resize(imgoutdrive,(hsizeout, vsizeout))
        imgresizeyd = cv2.resize(imgoutyard,(hsizeout, vsizeout))
        # Write the images to be served by an Apache server
        cv2.imwrite(livedir + drtext + 'tmp.jpg', imgresizedr, [cv2.IMWRITE_JPEG_PROGRESSIVE, 1,cv2.IMWRITE_JPEG_QUALITY, 80])
        cv2.imwrite(livedir + ydtext + 'tmp.jpg', imgresizeyd, [cv2.IMWRITE_JPEG_PROGRESSIVE, 1,cv2.IMWRITE_JPEG_QUALITY, 80])
        # Color the log background based on the time since last event
        if (timeelapdrive < coloraftertime) or (timeelapyard < coloraftertime):
            colorf = open(maindir + "color.txt", "w")
            colorf.write("#720808;")
        else:
            colorf = open(maindir + "color.txt", "w")
            colorf.write("#1f1f1f;")
        #For debugging - print("Time since last driveway log:", timeelapdrive)
        #For debugging - print("Time since last frontyard log:", timeelapyard)
        sys.stdout.flush()
        end_time = time.time()
        # Here we determine how long its taken to process the frame(s) and then add some sleep to maintain the desired framerate (This helps with the output timing so the browser hits blank images MUCH less often) as well it reduces uneeded system load.
        processtime = end_time-start_time
        if processtime < frametime:
            sleeptime = frametime-processtime
            time.sleep(sleeptime)
        # Here we copy the images to a second file after the sleep. This provides redundancy incase the browser tries to grab the image while its being written, we can provide a second image location for "onerror". (The reason why you see the "-.010" above is to account for the copy time. You might need to tweak for your system.
        shutil.copyfile(livedir + drtext + 'tmp.jpg', livedir + drtext + '.jpg')
        shutil.copyfile(livedir + ydtext + 'tmp.jpg', livedir + ydtext + '.jpg')
        final_time = time.time()
        print("Elapsed Time:", final_time-start_time)
