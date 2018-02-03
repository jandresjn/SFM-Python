#!/usr/bin/python
# -*- coding: utf-8 -*-
# SFM by Jorge Andrés Jaramillo Neme ----> Thesis UAO
#...............................................................................................
import os
import sys
import cv2
import math
import glob
import numpy as np
from PIL import Image
from numpy import linalg
#-----------------------------------------------------------------------------------------------
#Para cargar imagenes en vez de un video...
#images = sorted(glob.glob('./TestImages/*.jpg'),key=lambda f: int(filter(str.isdigit, f)))
#print str(images)
#-----------------------------------------------------------------------------------------------

class sfm_neme:
    'Clase para aplicar SFM usando Opencv 3.2'
    #Próximas variables
    def __init__(self,videoPath,calibracionCamara):
        self.videoPath = videoPath
        self.calibracionCamara = calibracionCamara

    def importarCalibracionCamara(self):
        with np.load(self.calibracionCamara) as X:
            mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
            return mtx,dist,rvecs,tvecs

    def preProcessing(self,inputImage): #Preprocesa y rota frame.
        outputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
        outputImage = cv2.bilateralFilter(outputImage,-1,30,5);
        rows,cols = outputImage.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),-90,1)
        dst = cv2.warpAffine(outputImage,M,(cols,rows))
        return dst


    def sfmSolver(self):
        cap = cv2.VideoCapture(self.videoPath)
        while(cap.isOpened()):
            success, frame = cap.read()
            #print str(ret)
            if (success and (int(round(cap.get(1))) % 10 == 0 or int(round(cap.get(1)))==1)):
                # Efectua la lectura cada n frames, en este caso 10.
                dst = self.preProcessing(frame)
                print str(cap.get(1))
                cv2.imshow('frame',dst)
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            elif success == False:
                cv2.waitKey(0)
                break
        cap.release()
        cv2.destroyAllWindows()

mapeo = sfm_neme('./videoInput/1.mp4','./calibrateCamera/camera_calibration.npz')
# mtx,dist,rvecs,tvecs = mapeo.importarCalibracionCamara()
# print str(dist)
mapeo.sfmSolver()
