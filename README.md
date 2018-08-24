#SecurityPyCam

This projects turns your webcam into a security system for when there is motion.

As soon as there is movement in your environment, it starts recording and dumping 
that footage inside of a folder wherever you hold the script, titled "footage"

The filenames all have the date and time(24hr format) of when the recording 
started.

After 10 seconds of no movement, it will stop recording, until the next time.

All recordings are saved in the .avi format.




###TO RUN THIS SCRIPT
```
pip install imutils
```
also you will need OpenCV (called cv2 for python). I used this to get the OpenCV 
python3 wheel for Windows
[https://www.lfd.uci.edu/~gohlke/pythonlibs/](https://www.lfd.uci.edu/~gohlke/pythonlibs/)



####To make it automatically turn on

If you are using Windows, go to the Task Scheduler. From there, you can make a 
trigger when the computer is idle to trigger this script. I included a batch 
file to make triggering the python script easier this way.

For Linux (and possible OSX), you can use `xprintidle` to get the amount of time 
the computer has been idle, then form a script, polling that idle time every few 
secs. If its above a desired threshold, trigger the python script.


> Hope you all enjoy it!
