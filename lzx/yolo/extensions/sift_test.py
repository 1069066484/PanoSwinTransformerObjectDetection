import cv2
import numpy as np


img = cv2.imread(r'E:\ori_disks\D\fduStudy\labZXD\repos\yolo5hw\example_data\1.png')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
print(help(cv2.SIFT_create))

kp = sift.detect(gray,None)#找到关键点
print(help(sift.detect))
print(cv2.KeyPoint.response)
for p in kp:
    print(p.pt, p.angle, p.octave, p.size, p.response)
img=cv2.drawKeypoints(gray,kp,img)#绘制关键点

cv2.imshow('sp',img)
cv2.waitKey(0)
