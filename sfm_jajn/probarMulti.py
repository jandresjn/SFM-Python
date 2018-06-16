#!/usr/bin/python
# -*- coding: utf-8 -*-
 # SFM by Jorge Andres Jaramillo Neme ----> Thesis UAO
#...............................................................................................
import cv2
import numpy as np
import time

# import cv2
# t0 = time.time()
# img_a = cv2.imread("./mediaInput/5.png", cv2.IMREAD_COLOR)
# gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
# img_b = cv2.imread("./mediaInput/6.png", cv2.IMREAD_COLOR)
# gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
#
# surf=cv2.xfeatures2d.SURF_create(400)
#
#
# # find the keypoints and descriptors with SIFT
# kp1, des1 = surf.detectAndCompute(gray_a,None)
# kp2, des2 = surf.detectAndCompute(gray_b,None)
# t1 = time.time()
#
# img2_a = cv2.UMat(cv2.imread("./mediaInput/5.png", cv2.IMREAD_COLOR))
# imgUMat_a = cv2.UMat(img_a)
# gray2_a = cv2.cvtColor(imgUMat_a, cv2.COLOR_BGR2GRAY)
# img2_b = cv2.UMat(cv2.imread("./mediaInput/5.png", cv2.IMREAD_COLOR))
# imgUMat_b = cv2.UMat(img_b)
# gray2_b = cv2.cvtColor(imgUMat_b, cv2.COLOR_BGR2GRAY)
#
# surf2=cv2.xfeatures2d.SURF_create(400)
#
#
# # find the keypoints and descriptors with SIFT
# kp12, des12 = surf2.detectAndCompute(gray2_a,None)
# kp22, des22 = surf2.detectAndCompute(gray2_b,None)
#
# t2 = time.time()
#
# print "El tiempo estimado es : " + str(t1-t0) + " y el otro "+ str(t2-t1)
def cosa(arbol,perro):
    a=arbol+perro
    print a

def cosa2(arbol,perro):
    arbol=arbol
    perro=perro
    a=arbol+perro
    print a

cosa(1,2)
cosa2(1,2)
