# tfhumandetectionnotify
Its a tensorflow python script that detects people in realtime from rtsp stream and pushes notifications - I need to come up with a better name...

Preface: This is an ongoing project that started out solely for my use, so not everything has been commented/documented, and a few things still need to be changed in the primary script rather than just in config (I just forgot to add variables for them as not needed in my config)

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

As may be obvious, this script was designed to run on one system that has a GPU installed and output the images to a directory so that they can be served by a webserver, in my case on a different machine. Youre welcome to edit and move things as you desire, as mentioned, I plan to make this configurable in the future via the config, but here is how its setup to operate by default.

Current images are output to /output/RD/ (on my system this is a tmpfs on my NAS so disk isnt hit by constant writes) the mlcamdisplay will pull from these images to generate the real time viewing
When a person is detected, the resulting dir is output to /output/
The script auto copies a php file into the dir that will display all images and links it in the log.
At the root of web youll find 3 php display files that you can integrate into your Organizr dashboard like I did, or another page, the smaller log page has a text link to the full log, so you dont have to display it at all times.

Finally, you may wonder about the bat files. I personally run this from a windows system (as it has a GPU) and then use one of my linux servers to serve the pages. The bat files can be used in combination with forever to manage the process.

I highly recommend using forever as it will insure the process restarts if it fails (if youre using pushbullet, the occasional timeout can cause this, I should prob work on the error handling)

Forever - https://www.npmjs.com/package/forever
