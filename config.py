# Path to the frozen graph that will be used for inference, see README for more info.
model_path = 'faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
# RTSP Stream URL's to capture video - I am using an mjpeg secondary stream that outputs 704x480 at 5fps. This seems to be a pretty standard substream option, if you use the main stream or something else, you may have to tweak other variables such as the text size for scores, the number of frames grabbed before return, etc.
capurl = "rtsp://username:password@ipaddress:554/cam/realmonitor?channel=1&subtype=1"
capurl2 = "rtsp://username:password@ipaddress:554/cam/realmonitor?channel=1&subtype=1"
# Score threshold for person to be detected
threshold = 0.86
# How long (in seconds) the detection log will stay red after a person is last detected.
coloraftertime = 45
# How long the script will wait (after the last detection event) before creating a new directory and log entry - this prevents us from getting a bunch of log entries if someone is persistently detected coming in and out of frame.
timebetweenevents = 60
# How long each loop should take, take 1/fps will give you this number, so for instance I target 2fps, so I use .500
frametime = .500
# The below entries are related to PushBullet, enabled is 0 or 1, the rest is self explanatory.
pbenabled = 0
pbapikey = 'apikey'
pbchannelname = 'channelname'
# URL to your server for pb links - this is the url to get to /output (you must include /output) 
url = 'https://someurl/output/'
# Horizontal filters as mentioned 0 is the top of the image, if you put something other than 0 here it will not detect persons whos bounding box is entirely outside (above) this value - read the main py to understand further.
hfilterdr = 0
hfilteryd = 0
# Variables below set the names for each camera that are used for the log entries, push notifications, and directory/image save names. This DOES NOT change the name output for the live imgs, this will come later.
cam1text =  'driveway'
cam2text = 'frontyard'
# These are the output directories, saved images to be served by your webserver for access via the above url are output to maindir, livedir is only used for the constantly refreshing images. The trailing slash is required so that if you wish to have everything in the same top level directory, just make this ''
maindir = 'output/'
livedir = 'output/RD/'
# Font size for use with timestamp and bounding box score, if using higher res, you will need to increase this.
fontsize = 0.65
# The script will resize the images before writing them out, you can set this value here, keep in mind, the larger this gets, the longer the *blocking* write operation gets which can be problematic.
imgwidth = 367
imgheight = 250