import os
import cv2
from matplotlib import pyplot as plt
import numpy as np


import argparse
import datetime
import time
os.chdir("C:/Users/sabarna.hazra/Desktop/carnival POC/imutils-master")
import imutils
os.chdir("C:/Users/sabarna.hazra/Desktop/carnival POC")
os.getcwd()
import get_points

fourcc = cv2.cv.CV_FOURCC('i', 'Y', 'U', 'V')


def denoise(frame):
    #frame = cv2.medianBlur(frame,21)
    #frame = cv2.fastNlMeansDenoising(frame,None,(10.10),7,21)
    frame = cv2.GaussianBlur(frame,(21,21),0)
    
    return frame


camera = cv2.VideoCapture('GLORY MOB 3-08-2015.mp4')
camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,720)
camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,1280)
camera.set(0,100000)
(_, frame2) = camera.read()
#frame2 = imutils.resize(frame2, width=1280,height=720)
height,width = frame2.shape[:2]
out = cv2.VideoWriter('output.avi',fourcc,30, (1280,720),1)
#frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
#frame2 = cv2.cvtColor(frame2, cv2.COLOR_HSV2BGR)

gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)


# initialize the first frame in the video stream
firstFrame = gray2
# loop over the frames of the video
flag=0
f_no =0
while True:
                # grab the current frame and initialize the occupied/unoccupied
                # text
                (grabbed, frame) = camera.read()
                text = "Unoccupied"

                # if the frame could not be grabbed, then we have reached the end
                # of the video
                if not grabbed:
                                break

                # resize the frame, convert it to grayscale, and blur it
                #frame = imutils.resize(frame, width=1280,height=720)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #gray = cv2.GaussianBlur(gray, (21, 21), 0)
                gray = denoise(gray)
                # draw region of interst
                if flag==0 :
                    points = get_points.run(frame, multi=True)
                    flag=1
                # if the first frame is None, initialize it
                if firstFrame is None:
                                firstFrame = gray
                                continue
                # compute the absolute difference between the current frame and
                # first frame
                frameDelta = cv2.absdiff(firstFrame, gray)
                thresh = cv2.threshold(frameDelta, 20, 255, cv2.THRESH_BINARY)[1]

                # dilate the thresholded image to fill in holes, then find contours
                # on thresholded image
                thresh = cv2.dilate(thresh, None, iterations=2)
                (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
                count=0
                # loop over the contours
                for c in cnts:
                                # if the contour is too small, ignore it
                                if cv2.contourArea(c) < 60 :
                                               continue

                                # compute the bounding box for the contour, draw it on the frame,
                                # and update the text
                                (x, y, w, h) = cv2.boundingRect(c)
                                # checking if the point is right side of the line
                                if ((x + 0.5*w - points[0][0])*(points[0][3]-points[0][1]) - (y + 0.5*h -points[0][1])*(points[0][2]-points[0][0])) >= 0  :
                                #if first frame then do not draw box, instead write coordinates in a text file
                                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                    text = "Occupied"
                                    count += 1
                                # draw the text and timestamp on the frame
                cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                cv2.putText(frame,"Detected "+str(count),(10,540),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # show the frame and record if the user presses a key
                cv2.line(frame,(points[0][0],points[0][1]),(points[0][2],points[0][3]),(255,255,255),2)
                cv2.imshow("Security Feed", frame)
                #cv2.imwrite("images/"+str(f_no)+".jpg",frame)
                out.write(frame)
                key = cv2.waitKey(1) & 0xFF
                f_no += 1

                # if the `q` key is pressed, break from the loop
                if key == ord("q"):
                                break

# cleanup the camera and close any open windows
camera.release()
out.release()
cv2.destroyAllWindows()

        



