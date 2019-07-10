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

i=0
files = glob.glob(args.images_path+'results_LK_twopoints/*')
for f in files:
    os.remove(f)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))

for filename in sorted(glob.glob(args.images_path+"*.jpg")):
    if i==0:
        old_frame = cv.imread(filename).astype(np.uint8)
        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
        number = re.findall('(\d{3,5}).jpg', filename, re.I)[0] or i
        # read the ROI for tracking
        roi = cv.imread(args.roi_path).astype(np.uint8)
        roi_gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        h, w, _ = roi.shape
        # find out initial location of window
        match = cv.matchTemplate(old_frame, roi, cv.TM_CCOEFF_NORMED)
        threshold = 0.8
        position = np.where(match >= threshold)
        point = list(zip(*position[::-1]))[0]
        x, y = point[0], point[1]
        track_window = (x, y, w, h)
        p0=np.float32(np.array([[x,y],[x,y+h],[x+w,y],[x+w,y+h]]))
        img2 = cv.rectangle(old_frame, (x, y), (x + w, y + h), 255, 2)
        cv.imwrite('frame_LK.jpg', img2)
        time.sleep(1)
        k = cv.waitKey(100) & 0xff
        if k == 27:
            break
        i+=1
    else:
        number = re.findall('(\d{3,5}).jpg', filename, re.I)[0] or i
        new_frame = cv.imread(filename).astype(np.uint8)
        frame_gray = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        x=p1[0][0]
        y=p1[0][1]
        z=p1[3][0]
        t=p1[3][1]
        img2 = cv.rectangle(new_frame, (x,y), (z,t), 255,2)
        cv.imwrite(args.images_path+'results_LK_twopoints/'+str(number)+'.jpg', img2)
        old_gray = frame_gray.copy()
        p0 = p1
        k = cv.waitKey(30) & 0xff
        i += 1
        if k == 27:
            break