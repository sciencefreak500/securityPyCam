import numpy as np
import cv2
from matplotlib import pyplot as plt

WindowName = "FaceDetect"
cv2.namedWindow(WindowName)
vc = cv2.VideoCapture(0)


MIN_MATCH_COUNT = 6

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

#load in whitelisted faces
michael = cv2.imread('whitelisted/michael.jpg',0)
#michael = cv2.cvtColor(michaelRef, cv2.COLOR_BGR2GRAY)
cv2.imshow('refface',michael)
# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()


if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
    cv2.imshow(WindowName, frame)

else:
    rval = False

while rval:
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        #frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        #eyes = eye_cascade.detectMultiScale(roi_gray)
        #for (ex,ey,ew,eh) in eyes:
            #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(michael,None)
        kp2, des2 = sift.detectAndCompute(roi_gray,None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 6)
        search_params = dict(checks = 33)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        #matches = flann.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32), 2)
        try:
            matches = flann.knnMatch(des1,des2,k=2)
        except:
            print('match failed')
            matches = []
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.61*n.distance:
                good.append(m)



        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            #h,w = michael.shape
            #pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            #dst = cv2.perspectiveTransform(pts,M)

            #img2 = cv2.polylines(roi_color,[np.int32(dst)],True,255,3, cv2.LINE_AA)
            cv2.putText(frame, "Michael Milord", (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        else:
            #print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
            matchesMask = None


        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
               singlePointColor = None,
               matchesMask = matchesMask, # draw only inliers
               flags = 2)

        img3 = cv2.drawMatches(michael,kp1,roi_gray,kp2,good,None,**draw_params)

                
        
        cv2.imshow('face',img3)
    cv2.imshow(WindowName,frame)
    

vc.release()
cv2.destroyAllWindows()
