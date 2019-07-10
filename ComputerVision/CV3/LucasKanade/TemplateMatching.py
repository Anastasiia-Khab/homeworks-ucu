import sys
from itertools import zip_longest as zip
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
parser.add_argument('--method', dest='method', type=int, help='method SSD:0 NCC:1 SAD:2')
args = parser.parse_args()

def normalize(img):
    mean=np.mean(img)
    return (img - mean) / np.sqrt(np.sum((img - mean) ** 2))



def templateMatch(image, template,method):
    result_tm=np.zeros((image.shape[0]-template.shape[0]+1,image.shape[1]-template.shape[1]+1))
    template_norm=normalize(template)
    if (method == 0):
        for i in range(image.shape[0]-template.shape[0]+1):
            for j in range(image.shape[1]-template.shape[1]+1):
                result_tm[i,j]=np.sum(
                    np.power(image[i:i+template.shape[0], j:j+template.shape[1]]-template,2))
    elif (method == 1):
        for i in range(image.shape[0] - template.shape[0] + 1):
            for j in range(image.shape[1] - template.shape[1] + 1):
                result_tm[i, j]=-np.sum(np.multiply(
                    normalize(image[i:i + template.shape[0], j:j + template.shape[1]]),
                    template_norm
                ))
    elif (method == 2):
        for i in range(image.shape[0] - template.shape[0] + 1):
            for j in range(image.shape[1] - template.shape[1] + 1):
                result_tm[i, j] = np.sum(
                    np.absolute(image[i:i + template.shape[0], j:j + template.shape[1]] - template))
    (y,x)=np.unravel_index(result_tm.argmin(), result_tm.shape)
    track_window = (x, y, template.shape[1], template.shape[0])
    return track_window

i=0
for filename in sorted(glob.glob(args.images_path+"*.jpg")):
    frame = cv.imread(filename).astype(np.uint8)
    number = re.findall('(\d{3,5}).jpg', filename, re.I)[0] or i
    if i==0:
            # read the ROI for tracking
        roi = cv.imread(args.roi_path).astype(np.uint8)
        h, w, _ = roi.shape
        (x2, y2, w2,h2) = templateMatch(frame, roi, args.method)
        img2 = cv.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), 255, 2)
        cv.imwrite( 'template_matching_SAD'+'.jpg', img2)
        i+=1
    else:
        break