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