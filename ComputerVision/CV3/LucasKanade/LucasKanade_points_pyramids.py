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
parser.add_argument('--max_level', dest='max_level', type=str, help='0-based maximal pyramid level number; ')
args = parser.parse_args()

i=0
files = glob.glob(args.images_path+'results_LK_pyramids/*')
for f in files:
    os.remove(f)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = args.max_level,
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
        p0 = cv.goodFeaturesToTrack(roi_gray, mask=None, **feature_params)
        p0+=[x,y]

        pict=old_frame.copy()
        for p in p0:
            pict = cv.circle(pict, (p[0][0], p[0][1]), 5, color[i].tolist(), -1)
        img2=pict
        cv.imwrite('frame.jpg', img2)
        time.sleep(1)
        k = cv.waitKey(100) & 0xff
        if k == 27:
            break
        i+=1
    else:
        number = re.findall('(\d{3,5}).jpg', filename, re.I)[0] or i
        new_frame = cv.imread(filename).astype(np.uint8)
        frame_gray = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)
        p1, st, err = cv.calcOpticalFlowPyrLK(prevImg=old_gray, nextImg=frame_gray, prevPts=p0, nextPts=None, **lk_params)
        p1=np.float32(np.array([[i]for i in p1[st==1]]))
        err=err[st==1]
        p1 = p1[err<50]
        err = err[err<50]
        p1 = p1[err > 0.01]
        pict = new_frame.copy()
        for p in p1:
            pict = cv.circle(pict, (p[0][0], p[0][1]), 5, color[1].tolist(), -1)
        img2=pict
        cv.imwrite(args.images_path+'results_LK_pyramids/'+str(number)+'.jpg', img2)
        old_gray = frame_gray.copy()
        p0 = p1
        k = cv.waitKey(30) & 0xff
        i += 1
        if k == 27:
            break
