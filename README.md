# tfhumandetectionnotify
Its a tensorflow python script that detects people in realtime from rtsp stream and pushes notifications - I need to come up with a better name...

Preface: This is an ongoing project that started out solely for my use, so not everything has been commented/documented, although I believe at this point most everything is configurable via the primary script.

Screenshots - 

Integrated into Organizrv2 - https://ibb.co/XjGqYrz

Sample output from a detection event - https://ibb.co/prd1K1D

First, the prereqs -

You will need python3 plus a functioning tensorflow install - see here - https://www.tensorflow.org/install

Use pip to install the prereqs and their dependencies -

numpy
cv2
msvcrt
logging
shutil
pushbullet (optional)

MOST of the things that need to be edited for your config are in config.py, however, one thing in specific is not, if youre using pushbullet to push notifications, you will need to edit the url it pushes in the primary py to match the url your webserver resides at. I personally use an address here that hits a reverse proxy with https and auth thats open to the web, so I can see the images when Im away from home.

**EDIT** The url is now set via the config file

As may be obvious, this script was designed to run on one system that has a GPU installed and output the images to a directory so that they can be served by a webserver, in my case on a different machine. Youre welcome to edit and move things as you desire, as mentioned, I plan to make this configurable in the future via the config, but here is how its setup to operate by default.

Current images are output to /output/RD/ (on my system this is a tmpfs on my NAS so disk isnt hit by constant writes) the mlcamdisplay will pull from these images to generate the real time viewing
When a person is detected, the resulting dir is output to /output/
The script auto copies a php file into the dir that will display all images and links it in the log.
At the root of web youll find 3 php display files that you can integrate into your Organizr dashboard like I did, or another page, the smaller log page has a text link to the full log, so you dont have to display it at all times.

If you ahve problems with the images flashing or refreshing during a write, the secondary img will help avoid this, but its not perfect, tweaking the refresh time in mlcamdisplay will yeild the absolute best results, as well you will need to change this is you wish to display at a diff fps.

Finally, you may wonder about the bat files. I personally run this from a windows system (as it has a GPU) and then use one of my linux servers to serve the pages. The bat files can be used in combination with forever to manage the process.

I highly recommend using forever as it will insure the process restarts if it fails (if youre using pushbullet, the occasional timeout can cause this, I should prob work on the error handling)

Forever - https://www.npmjs.com/package/forever

If you use forever as recommended, with the bat scripts, all logging will occur to output/RD
processlog.log gets rotated on each startup keeping 1 
processerror.log is persistent

I recommend a cron job for the cleanup.sh, I run this on my webserver to clear any output directories/jpgs over 5 days. I also need to write something to truncate the detection log, but I havent gotten around to it as it doesnt really get THAT big in terms of filesize.

As well, Ive noticed running this for very long periods of time (days) can possibly contribute to increased frame latency (possibly more buffer issues?) But restarting the process nightly seems to resolve it, hence the restat .bat file. In windows, I setup a task schedule to run it nightly. Soon I will make .sh equivs of this for Linux :) You can automate with cron of course.
