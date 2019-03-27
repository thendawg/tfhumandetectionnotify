# tfhumandetectionnotify
Its a tensorflow python script that detects people in realtime from rtsp stream and pushes notifications - I need to come up with a better name...

Preface: This is an ongoing project that started out solely for my use, so not everything has been commented/documented, although I believe at this point most everything is configurable via config.py.

Changelog -
3/13 - Everything that can be changed (afaik) now has been set to a variable that can be configured in config.py, if Im missing anything please let me know. Complete config and script for 1 cam has also been updated.

In the future, I plan to make number of cameras and the location's name a configurable option. This will come with a php interface for easy config in the future, for now, if youre using a single stream, overwrite humandetectpushfoldersthreaded.py and config.py with humandetectpushfoldersthreadedsingle.py and configsingle.py respectively.

**UPDATE 3/27 - A new method for realtime monitoring/web display**

I have played with flask a bit in the past, but after learning A TON a long the way of building this project, I am back to it. It seems it is the perfect solution for streaming the jpgs to browser with minimal latency, no caches being overwhelmed, no browser eating memory, and best of all, it minimizes load on the system running inference when no user is connected. To use this method, follow the directions below...

1) Install flask (pip install Flask)
2) Make sure index.html is in the templates folder which should be at the same level as humandetectlotsothreadspbtflask.py
3) Run humandetectlotsothreadspbtflask.py instead of the other py. 
4) Flask server will run on http://IP:5000 - feel free to edit the index.html to display the images however you like, I simply formatted it to replace the previous mlcamdisplay page so I can incorporate it into Organizr.

Keep in mind, this version is for 2 streams only at this time, and adds all of the improvements added into the other recent variations. Now that I feel the primary code is fairly optimized, Ill start working on a varation for multiple cams all configurable from a php page in a single package. For now, feel free to play with the py and make it suite whatever you like, it's fairly well documented :)

**THREADED TF BETA**

Currently this relies on the same config and web output files as the other revs so I havent branched it yet but will soon as this moves forward. This beta is only compatible with 2 stream sources and will use multiple threads for tensorflow, essentially cutting the image processing time in half. Ive been testing with a 3GB 1060, and although it gives me messages about vram potentially impacting performance, Ive restricted it all the way down to 1GB VRAM and have seen no actual performance impact. In this iteration Ive also dramatically improved the stream capture threading.

Future releases which are able to iterate any number of streams will be based off of this design, so I thought Id show what Im working on. As well, I plan to further thread the image processing (drawing boxes, scores, etc) as well as the img writing. This will all be in a packaged release at somepoint - hopefully with a better web interface as well!


**Screenshots**

Integrated into Organizrv2 - https://ibb.co/XjGqYrz

Sample output from a detection event - https://ibb.co/prd1K1D

**Prereqs**

You will need python3.5 plus a functioning tensorflow install - see here - https://www.tensorflow.org/install

If youre using modern hardware theres likely a binary of sorts for install, in my case I had to build from source, which can be, problematic, but tensorflow does have a decent community.

Use pip to install the prereqs and their dependencies -

numpy
cv2
msvcrt
logging
shutil
pushbullet (optional)

You will also need a model to run inference on. For the purpose of this README, we wont go into training your own model, but more info can be found in the tensorflow object detection API docs. Most of my testing has been done with pretrained models on the coco dataset available here - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md. Eventually I hope to have enough positive hit images (1000+) to retrain myself. For now, you can use any of these pre-trained COCO based models, as long as they return boxes, not masks. You will need to edit the path to the frozen inference graph in config.py. Im now testing faster_rcnn_resnet101_coco as it supposedly has a bit higher confidence rate and is still able to process 2 streams at 2fps with ease on my GTX1060.

**How it works...**

As may be obvious, this script was designed to run on one system that has a GPU installed and output the images to a directory so that they can be served by a webserver, in my case on a different machine. These directories are configurable in config.py

Current images are output to livedir (on my system this is a tmpfs on my NAS so disk isnt hit by constant writes) the mlcamdisplay.php will pull from these images to generate the real time viewing
When a person is detected, the resulting dir is output to maindir
The script auto copies a php file into the dir that will display all images and links it in the log.
At the root of web youll find 3 php display files that you can integrate into your Organizr dashboard like I did, or another page, the smaller log page has a text link to the full log, so you dont have to display it at all times.

If you have problems with the images flashing or refreshing during a write, the secondary img will help avoid this, but its not perfect, tweaking the refresh time in mlcamdisplay will yeild the absolute best results, as well you will need to change this is you wish to display at a diff fps.

**Update 3/21 - The Web Monitor/Log Display**

The web display pages now use the default directories used in the config. Previously, I had not realized that they were still populated with my custom directory used on my webserver. Now if all of the files are placed in the same base directory, using "output/" for the maindir with "output/RD" for live images, it will function without any additonal config. If you change the dir in the config.py, youll need to manually update the directories on the php pages - shouldnt be too hard to locate. Eventually this will all become dynamic.

The latest changes Ive made to the java on the display page seems to work perfect in Firefox/Edge to render the alt image if the primary is unavailable, however,  Chrome seems to have some major issues with the way Im doing this. Ill look into it at some point, but to be honest, the live image view is more for my purposes than what I want this project for long term (push notifications with images, which seems to be solid). As well, I know there must be a better way of doing this with node or something - see below...

I should add, the web frontend for displaying images is not my forte, and Im sure theres a better way of doing it, so if anyone has one, please feel free to contribute or contact me.

**Final words and recommendations**

Finally, you may wonder about the bat files. I personally run this from a windows system (as it has a GPU) and then use one of my linux servers to serve the pages. The bat files can be used in combination with forever to manage the process.

I highly recommend using forever as it will insure the process restarts if it fails (if youre using pushbullet, the occasional timeout can cause this, I should prob work on the error handling)

Forever - https://www.npmjs.com/package/forever

If you use forever as recommended, with the bat scripts, all logging will occur to output/RD, or you can change this in the bat.
processlog.log gets rotated on each startup keeping 1 
processerror.log is persistent

I recommend a cron job for the cleanup.sh, I run this on my webserver to clear any output directories/jpgs over 5 days. I also need to write something to truncate the detection log, but I havent gotten around to it as it doesnt really get THAT big in terms of filesize.

As well, Ive noticed running this for very long periods of time (days) can possibly contribute to increased frame latency (possibly more buffer issues?) But restarting the process nightly seems to resolve it, hence the restat .bat file. In windows, I setup a task schedule to run it nightly. Soon I will make .sh equivs of this for Linux :) You can automate with cron of course.
