#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import glob
import math
from numpy import linalg

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('./calibraTablet/*.jpg')
i=1
for fname in images:
    img = cv2.imread(fname)
    # cv2.imshow('img',img)
    # cv2.waitKey(500)
    # wPercent = (1000/float(img.size[0]))  # Saca el porcentaje que representa el ancho fijado con respecto a la imagen actual
    # ejemplo, para una imagen de width de 800, queda 500/800, osea que corresponde 500 al 62.5 % de 800....
    # height = int((float(img.size[1]) * float(wPercent)))
    # img= cv2.resize((1000, height), Image.BILINEAR)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9,6), corners,ret)
        #cv2.imshow('img',img)
        print "imagen correcta #: " + str (i)
        i+=1
        #cv2.waitKey(500)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print str(mtx)
np.savez('camera_calibration.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)




# print "por aca ando"
# img = cv2.imread('2.jpg')
# h,  w = img.shape[:2]
# newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
# dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# print "ya casi"
# x,y,w,h = roi
# dst = dst[y:y+h, x:x+w]
# cv2.imwrite('calibresult1.png',dst)

img = cv2.imread('./calibraTablet/foto.jpg')
cv2.imshow('img',img)
cv2.waitKey(5000)
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
print "ya casi"
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult2.png',dst)
# cv2.imshow('Imagen corregida',dst)
# cv2.imshow('Imagen sin corregir',img)
# cv2.waitKey(0)
print "acabe"
cv2.destroyAllWindows()
