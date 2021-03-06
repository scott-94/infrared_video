import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.path as mplPath

import argparse
import datetime
import time


os.chdir("C:/Users/sabarna.hazra/Desktop/carnival POC/imutils-master")
import imutils
os.chdir("C:/Users/sabarna.hazra/Desktop/carnival POC")
os.getcwd()
import get_points_v3
import meanshift

fourcc = cv2.cv.CV_FOURCC('i', 'Y', 'U', 'V')


def denoise(frame):
    #frame = cv2.medianBlur(frame,21)
    #frame = cv2.fastNlMeansDenoising(frame,None,(10.10),7,21)
    frame = cv2.GaussianBlur(frame,(21,21),0)
    
    return frame


camera = cv2.VideoCapture('GLORY MOB 3-08-2015.mp4')
camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,720)
camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,1280)
camera.set(0,400000)
(_, frame2) = camera.read()
#frame2 = imutils.resize(frame2, width=1280,height=720)
height,width = frame2.shape[:2]
out = cv2.VideoWriter('output.avi',fourcc,30, (1280,720),1)
#frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
#frame2 = cv2.cvtColor(frame2, cv2.COLOR_HSV2BGR)

gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)
#file_r=open("loc2.txt","w")

# initialize the first frame in the video stream
firstFrame = gray2
cv2.bitwise_not(gray2,firstFrame)
#firstFrame = gray2
# loop over the frames of the video
flag=0
f_no =0
detect_flag=0
pt=[]
while True:
                # grab the current frame and initialize the occupied/unoccupied
                # text
                (grabbed, frame) = camera.read()
                text = "Unoccupied"
                cv2.bitwise_not(frame,frame)
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
                    points = get_points_v3.run(frame, multi=True)
                    flag=1
                    points2=np.array(points)
                
                bbPath = mplPath.Path(points2)
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
                                if (bbPath.contains_point((x + 0.5*w,y + 0.5*h)))  :
                                #if first frame then do not draw box, instead write coordinates in a text file
                                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                    text = "Occupied"
                                    pt.append((int(x + 0.5*w),int(y + 0.5*h)))
                                    #writing center point in a text file
                                    #file_r.write(str(int(x + 0.5*w))+","+(str(int(y + 0.5*h)))+",")
                                    #cv2.circle(frame, (int(x + 0.5*w),int(y + 0.5*h)), 1, (0,0,255),1)
                                    #cv2.floodFill(frame[y:y+h,x:x+w], mask, (x,y), 255)
                                    #meanshift.MS(x-5,y-5,w+10,h+10,frame,camera)
                                # draw the text and timestamp on the frame
                cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
#                cv2.putText(frame,"Detected "+str(count),(10,540),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#
#                # show the frame and record if the user presses a key
#                cv2.line(frame,(points[0][0],points[0][1]),(points[0][2],points[0][3]),(255,255,255),2)
                for pt1 in zip(points, points[1:]):
                    cv2.line(frame,pt1[0],pt1[1],(255,255,255),2)
                cv2.line(frame,points[-1],points[0],(255,255,255),2)
                cv2.imshow("Security Feed", frame)
                cv2.imwrite("images/"+str(f_no)+".jpg",frame)
#                out.write(frame)
                key = cv2.waitKey(1) & 0xFF
                f_no += 1

#                if the `q` key is pressed, break from the loop
                if key == ord("q"):
                                break

# cleanup the camera and close any open windows
#file_r.close()
for k in xrange(2, len(pt)):
		# if either of the tracked points are None, ignore
		# them
		if pt[k - 1] is None or pt[k] is None:
			continue
 
		# otherwise, compute the thickness of the line and
		# draw the connecting lines
		#thickness = int(np.sqrt(args["buffer"] / float(k + 1)) * 2.5)
		cv2.line(frame, (pt[k - 1][0],pt[k - 1][1]), (pt[k][0],pt[k][1]), (0, 0, 255), 1)
cv2.destroyAllWindows()
cv2.imshow("Trajectory", frame)
key = cv2.waitKey(1)
cv2.imwrite("images/"+str(f_no)+".jpg",frame)
if key == ord("q"):
    camera.release()
    cv2.destroyAllWindows()
# points=open("loc2.txt","r")
# all_points = points.read().split(',')

# for i in range(len(all_points)-1) :
    # all_points[i]=int(all_points[i])
    # #print str(all_points[i])+"    "+str(i)

# pts=[]    
# for j in range(0,len(all_points)-1,2):
                # x1=all_points[j]
                # y1=all_points[j+1]
                # pt=(x1,y1)
                # pts.append(pt)


# cv2.polylines(frame, np.int32([pts]), 1, (255,255,255))                
# cv2.imshow("Security Feed", frame)
# cv2.waitKey(0)

        



