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
    # Objetos y variables globales, que pueden usarse en diferentes funciones.
    def __init__(self,videoPath,calibracionCamara):
        self.videoPath = videoPath
        self.calibracionCamara = calibracionCamara
        self.detector = cv2.xfeatures2d.SURF_create(400) # Lo defino general, por que lo utilizaré en 2 funciones.
        self.FLANN_INDEX_KDTREE = 1
        self.flann_params = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=5)
        self.matcher = cv2.FlannBasedMatcher(self.flann_params, {})
        self.matches= None
        self.ratio = 0.75 # Ratio para el filtro de matches...
        self.mtx,self.dist,self.rvecs,self.tvecs = None,None,None,None

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

    def filter_matches(self):
        print 'matches feos :' + str(len(self.matches))
        # Se encuentran los mejores matches según el ratio especificado y se agrupan en un array
        filtered_matches = []
        for m in self.matches:
            if len(m) == 2 and m[0].distance < m[1].distance * self.ratio:
                filtered_matches.append(m[0])
        return filtered_matches

    def imageDistance(self,matches_subset):
        # Se suma las distancias de todosl os matches ecuación
        sumDistance = 0.0
        for match in matches_subset:
            sumDistance += match.distance
        return sumDistance

    def featureMatching(self,inputImage1,inputImage2):
        base_features, base_descs = self.detector.detectAndCompute(inputImage1, None)
        next_features, next_descs = self.detector.detectAndCompute(inputImage2, None)
        self.matches = self.matcher.knnMatch(next_descs, trainDescriptors=base_descs, k=2)
        print "\t Match Count: ", len(self.matches)
        #print 'matche' + str(matches)
        matches_subset = self.filter_matches()
        print "\t Filtered Match Count: ", len(matches_subset)
        distance = self.imageDistance(matches_subset)
        print "\t Distance from Key Image: ", distance
        averagePointDistance = distance/float(len(matches_subset))
        print "\t Average Distance: ", averagePointDistance
        kp1 = []
        kp2 = []
        for match in matches_subset:
            kp1.append(base_features[match.trainIdx])  # Guarda los keyponts correspondiente y litrados de la primera imagen
            kp2.append(next_features[match.queryIdx])  # Guarda los keyponts correspondientes y filtrados de la segunda imagen
        p1 = np.array([k.pt for k in kp1], np.float32)  # Parecido al reshape del tutorial, lo guarde en un vector
        p2 = np.array([k.pt for k in kp2], np.float32)  # Parecido al reshape lo guarda en un vector
        return p1, p2


    def sfmSolver(self):
        self.mtx,self.dist,self.rvecs,self.tvecs=self.importarCalibracionCamara()
        cap = cv2.VideoCapture(self.videoPath)
        while(cap.isOpened()):
            success, frame = cap.read()
            #print str(ret)
            if (success and (int(round(cap.get(1))) % 5 == 0 or int(round(cap.get(1)))==1) and cap.get(1)<=5):# Añado para usar 2 imágenes.
                # Efectua la lectura cada n frames, en este caso 5.
                frameSiguiente = self.preProcessing(frame)
                if (cap.get(1) == 1):
                    frameActual = frameSiguiente
                kp1_filtrado , kp2_filtrado=self.featureMatching(frameActual,frameSiguiente)

                E,mask =cv2.findEssentialMat(kp1_filtrado,kp2_filtrado,self.mtx,cv2.RANSAC,0.999,1.0)
                points, R, t, mask = cv2.recoverPose(E, kp1_filtrado,kp2_filtrado)
                print 'retval: '
                print str(E)
                print 'mask: '
                print str(R)

                # print str(cap.get(1))
                cv2.imshow('frame',frameSiguiente)
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
