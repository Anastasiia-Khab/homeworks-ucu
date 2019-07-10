import cv2 as cv

image = cv.imread("/home/ana/ana/ucu/CV/homeworks/CV3/MeanShift/meanshift/Bird2/img/0001.jpg")
part = image[240:280,100:140]
cv.imwrite("Bird2_roi.png", part)