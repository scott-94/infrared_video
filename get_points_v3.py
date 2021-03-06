# Import the required modules
import cv2
import argparse
import os
import numpy as np
os.chdir("C:/Users/sabarna.hazra/Desktop/carnival POC/imutils-master")
import imutils
os.chdir("C:/Users/sabarna.hazra/Desktop/carnival POC")



def run(im, multi=False):
    #im = imutils.resize(im, width=1280,height=720)
    flag=0
    im_disp = im.copy()
    im_draw = im.copy()
    window_name = "Draw line of interest."
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, im_draw)

    # List containing top-left and bottom-right to crop the image.
    pts_1 = []
    pts_2 = []

    rects = []
    run.mouse_down = False

    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
	    if multi == False and len(pts_2) == 1:
		print "WARN: Cannot select another object in SINGLE OBJECT TRACKING MODE."
		print "Delete the previously selected object using key `d` to mark a new location."
		return
            run.mouse_down = True
            pts_1.append((x, y))
            print pts_1
#        elif event == cv2.EVENT_LBUTTONUP and run.mouse_down == True:
#            run.mouse_down = False
#            pts_2.append((x, y))
#            print "Object selected at [{}, {}]".format(pts_1[-1], pts_2[-1])
#        elif event == cv2.EVENT_MOUSEMOVE and run.mouse_down == True:
#            im_draw = im.copy()
            #rect = (pts_1[-1],(x,pts_1[-1][1]),(x,y),(pts_1[-1][0],y))
#            box = cv2.cv.BoxPoints(rect)
#            box = np.int0(box)
#            cv2.drawContours(im_draw,[box],0,(0,0,255),2)
#            cv2.rectangle(im_draw, pts_1[-1], (x, y), (255,255,255),2)
#            cv2.imshow(window_name, im_draw)

    print "Press and release mouse around the object to be tracked. \n You can also select multiple objects."
    cv2.setMouseCallback(window_name, callback)

    print "Press key `p` to continue with the selected points."
    print "Press key `d` to discard the last object selected."
    print "Press key `q` to quit the program."
    
    while True:
        # Draw the rectangular boxes on the image
        window_name_2 = "Objects to be detected."
        #print pts_1[-1]
#        for pt1 in zip(pts_1, pts_1[1:]):
#            cv2.line(im_disp,pt1[0],pt1[1],(255,255,255),2)
#            if len(pts_1)>=4:
#                flag=1
            #cv2.line(im_disp,pts_1[-1],pts_1[0],(255,255,255),2)            
            #cv2.rectangle(im_disp, pt1, pt2, (255, 255, 255), 2)
        # Display the cropped images
        cv2.namedWindow(window_name_2, cv2.WINDOW_NORMAL)
#        if flag ==1 :
#            cv2.line(im_disp,pts_1[-1],pts_1[0],(255,255,255),2)
#        cv2.imshow(window_name_2, im_disp)
        key = cv2.waitKey(30)
        if key == ord('p'):
            # Press key `s` to return the selected points
            cv2.destroyAllWindows()
#            point= [(tl + br) for tl, br in zip(pts_1, pts_2)]
#            corrected_point=check_point(point)
            return pts_1
        elif key == ord('q'):
            # Press key `q` to quit the program
            print "Quitting without saving."
            exit()
        elif key == ord('d'):
            # Press ket `d` to delete the last rectangular region
            if run.mouse_down == False and pts_1:
                print "Object deleted at  [{}, {}]".format(pts_1[-1], pts_2[-1])
                pts_1.pop()
                pts_2.pop()
                im_disp = im.copy()
            else:
                print "No object to delete."    
    cv2.destroyAllWindows()
    point= [(tl + br) for tl, br in zip(pts_1, pts_2)]
    corrected_point=check_point(point)
    return corrected_point

def check_point(points):
    out=[]
    for point in points:
        #to find min and max x coordinates
        if point[0]<point[2]:
            minx=point[0]
            maxx=point[2]
        else:
            minx=point[2]
            maxx=point[0]
        #to find min and max y coordinates
        if point[1]<point[3]:
            miny=point[1]
            maxy=point[3]
        else:
            miny=point[3]
            maxy=point[1]
        out.append((minx,miny,maxx,maxy))

    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--imagepath", required=True, help="Path to image")

    args = vars(ap.parse_args())

    try:
        im = cv2.imread(args["imagepath"])
    except:
        print "Cannot read image and exiting."
        exit()
    points = run(im)
    print "Rectangular Regions Selected are -> ", points 
