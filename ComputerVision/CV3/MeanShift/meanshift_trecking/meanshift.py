import numpy as np
import cv2 as cv
import argparse
import glob
import time
import re
import os


parser = argparse.ArgumentParser()
parser.add_argument('--images_path', dest='images_path', type=str, help='path to image file')
parser.add_argument('--roi_path', dest='roi_path', type=str, help='path to roi image')
args = parser.parse_args()

#cap = cv.VideoCapture("/home/ana/ana/ucu/CV/homeworks/CV3/MeanShift/meanshift/slow_traffic_small.mp4")
#frame=cv.imread(bird2_pass)
#img2 = cv.rectangle(frame, (80,200), (150,290), 255,2)
#cv.imwrite('img2.jpg',img2)
#biker_glob='/home/ana/ana/ucu/CV/homeworks/CV3/MeanShift/meanshift/Biker/img/*.jpg'
#bird2_glob="/home/ana/ana/ucu/CV/homeworks/CV3/MeanShift/meanshift/Bird2/img/*.jpg"
i=0
files = glob.glob(args.images_path+'results/meanshift/*')
for f in files:
    os.remove(f)

for filename in sorted(glob.glob(args.images_path+"*.jpg")):
    frame = cv.imread(filename).astype(np.uint8)
    number = re.findall('(\d{3,5}).jpg', filename, re.I)[0] or i
    if i==0:
        # read the ROI for tracking
        roi = cv.imread(args.roi_path).astype(np.uint8)
        h, w, _ = roi.shape
        # find out initial location of window
        match = cv.matchTemplate(frame, roi, cv.TM_CCOEFF_NORMED)
        threshold = 0.8
        position = np.where(match >= threshold)
        point = list(zip(*position[::-1]))[0]
        x, y = point[0], point[1]
        track_window = (x, y, w, h)
        # ______________________________
        # Bird2 GRAY
        #hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        #mask = cv.inRange(hsv_roi, 0, 50.)
        # ______________________________
        # Other
        hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv_roi, np.array((0., 0., 0.)), np.array((180., 255., 255.)))
        # ______________________________
        roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
        i+=1
    else:
        # ______________________________
        # Bird2 GRAY
        #hsv = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # _______________________________
        # Other
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # _______________________________
        dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # apply camshift to get the new location
        ret, track_window = cv.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x,y,w,h = track_window
        img2 = cv.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv.imshow('img2',img2)
        cv.imwrite(args.images_path+'results/meanshift/'+str(number)+'.jpg', img2)
        time.sleep(0.001)
        k = cv.waitKey(30) & 0xff
        i+=1
        if k == 27:
            break

