#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
import cv2

import math
import numpy as np
# hp_arr = np.array([10,20,3], np.float32)
# print str(hp_arr)
# base_p1 = np.ones(3, np.float32)
# matriz=np.matrix(base_p1, np.float32)
# print str(base_p1)
# print str(matriz)
# normal_pt = np.array([hp_arr[0]/hp_arr[2], hp_arr[1]/hp_arr[2]], np.float32)
# print str(normal_pt)
#
# if (normal_pt[1, 0] > 1):
#     print 'chepe'

#base = cv2.imread('1.jpg')  # lee primer imagen

#np.savez('fotobase.npz', base=base)
x=np.load('fotobase.npz')
base=x['base']
#print str(base)
#base2=cv2.cvtColor(base,cv2.COLOR_GRAY2BGRA)
#base=cv2s.fromarray(base)
cv2.imshow('Imagen sin corregir',base)
cv2.waitKey(0)
print "acabe"
cv2.destroyAllWindows()

# with np.load('camera_calibration.npz') as X:
#     mtx, dist, rvecs, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
#
# print 'mtx: '+ str(mtx)
# print 'dist' + str(dist)
# print 'rvecs'+ str(rvecs)
