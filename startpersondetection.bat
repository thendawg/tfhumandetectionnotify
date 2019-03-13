del /f /q C:\mlapps\output\RD\lastprocesslog.log
move C:\mlapps\output\RD\processlog.log C:\mlapps\output\RD\lastprocesslog.log
del /f /q C:\mlapps\output\RD\processlog.log
forever -a -l C:\mlapps\output\RD\processlog.log -e C:\mlapps\output\RD\processerror.log start -c python humandetectpushfoldersthreaded.py