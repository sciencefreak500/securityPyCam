# import the necessary packages

#thank you pyimagesearch for providing a great template to base this from
#https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/

from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import os
import sys
from threading import Thread

countdownTriggered = False
text = "Unoccupied"
countdown = 0

def startCountdown():
    global countdownTriggered, text, countdown
    if countdownTriggered == False:
        countdownTriggered = True
    #print(countdown)
    if countdown > 0:
        time.sleep(1)
        countdown -= 1
        startCountdown()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())
 
# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
 
# otherwise, we are reading from a video file
else:
    vs = cv2.VideoCapture(args["video"])
 
# initialize the first frame in the video stream
firstFrame = None

frame = vs.read()
frame = frame if args.get("video", None) is None else frame[1]

if frame is None:
    print("something wrong with frame")
    sys.exit()
frame = imutils.resize(frame, width=640, height=480)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (21, 21), 0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = None

# loop over the frames of the video
while True:
    # grab the current frame and initialize the occupied/unoccupied
    # text
    firstFrame = gray
    frame = vs.read()
    frame = frame if args.get("video", None) is None else frame[1]

    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if frame is None:
        break

    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=640, height=480)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if countdown > 0:
        if video_writer == None:
            tstamp = "footage_" + str(datetime.datetime.now()).split('.')[0].replace(':','_') + ".avi"
            if os.name == 'nt':
                fullNamePath = 'footage\\' + tstamp
            else:
                fullNamePath = 'footage/' + tstamp
            video_writer = cv2.VideoWriter(fullNamePath, fourcc, 60, (640, 480))
        video_writer.write(frame)
        
    if countdown == 0:
        text = "Unoccupied"
        countdownTriggered = False;
        if video_writer != None:
            video_writer.release()
            video_writer = None


    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        continue
    
    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < args["min_area"]:
                continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        #(x, y, w, h) = cv2.boundingRect(c)
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Occupied"
        countdown = 10
        if countdownTriggered == False:
            t = Thread(target=startCountdown, args=())
            t.start()

        # draw the text and timestamp on the frame
    if text == "Occupied":        
        cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)

    # show the frame and record if the user presses a key
    cv2.imshow("Security Feed", frame)
    #cv2.imshow("Thresh", thresh)
    #cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()
sys.exit()
