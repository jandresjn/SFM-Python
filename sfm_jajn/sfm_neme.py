#!/usr/bin/python
# -*- coding: utf-8 -*-
# SFM by Jorge Andrés Jaramillo Neme ----> Thesis UAO
#...............................................................................................
import cv2
import math
import glob
import numpy as np
from VtkPointCloud import VtkPointCloud

#----------------------------------- PENDIENTE DE INLCUIR --------------------------------------
#Para cargar imagenes en vez de un video...
#images = sorted(glob.glob('./TestImages/*.jpg'),key=lambda f: int(filter(str.isdigit, f)))
#print str(images)
#-----------------------------------------------------------------------------------------------

class sfm_neme:
    'Clase para aplicar SFM usando Opencv 3.2'
    # Objetos y variables globales, que pueden usarse en diferentes funciones.
    def __init__(self,mediaPath,calibracionCamara):
        self.mediaPath = mediaPath
        self.calibracionCamara = calibracionCamara
        self.detector = cv2.xfeatures2d.SURF_create(400) # Lo defino general, por que lo utilizaré en 2 funciones.
        self.FLANN_INDEX_KDTREE = 1
        self.flann_params = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=5)
        self.matcher = cv2.FlannBasedMatcher(self.flann_params, {})
        self.matches= None
        self.ratio = 0.75 # Ratio para el filtro de matches...
        self.mtx,self.dist,self.rvecs,self.tvecs = None,None,None,None


    def importarCalibracionCamara(self):
        if (type(self.calibracionCamara) is str):
            with np.load(self.calibracionCamara) as X:
                mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
                print 'Se seleccionó archivo .npz para la calibración'
                print 'mtx: '
                print mtx
                print 'dist: '+ str(dist)
                return mtx,dist
        else:
            print 'Se selecciona entrada K manual'
            focal=self.calibracionCamara[0]
            cx=self.calibracionCamara[1]
            cy=self.calibracionCamara[2]
            mtx=np.array([[focal,0,cx],[0,focal,cy],[0,0,1]],np.float32)
            dist=np.zeros((1,5))
            print 'mtx: '
            print mtx
            print 'dist: '+ str(dist)
            return mtx,dist

    def preProcessing(self,inputImage): #Preprocesa y rota frame.
        outputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
        # /¿outputImage = cv2.bilateralFilter(outputImage,-1,30,5);
        # rows,cols = outputImage.shape
        # M = cv2.getRotationMatrix2D((cols/2,rows/2),-90,1)
        # dst = cv2.warpAffine(outputImage,M,(cols,rows))
        return outputImage

    def filter_matches(self):
        print 'matches sin filtrar :' + str(len(self.matches))
        # Se encuentran los mejores matches según el ratio especificado y se agrupan en un array
        filtered_matches = []
        # for m in self.matches:
        #     if len(m) == 2 and m[0].distance < m[1].distance * self.ratio:
        #         filtered_matches.append(m[0])
        for m,n in self.matches:
            if m.distance < self.ratio*n.distance:
                filtered_matches.append(m)
        return filtered_matches

    def imageDistance(self,matches_subset):
        # Se suma las distancias de todos los matches ecuación
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
            kp1.append(base_features[match.trainIdx])# Lleno arreglo con sólo los keypoints comunes
            kp2.append(next_features[match.queryIdx])
        p1 = np.array([k.pt for k in kp1], np.float32) # Extraigo sólo las coordenadas(point) de los keypoint filtrados.
        p2 = np.array([k.pt for k in kp2], np.float32)
        return p1, p2,matches_subset,base_features,next_features

# Chequea que la triangulación esté bien teniendo en cuenta el determinante de R
    def CheckCoherentRotation(self,matR):
        EPS= 1E-7
        #print 'determinante: '+str(np.linalg.det(matR))
        if(np.fabs(np.linalg.det(matR))-1.0>EPS):
            print 'Error matriz inválida'
        print 'Matriz válida'

    def sfmSolver(self):
        self.mtx,self.dist=self.importarCalibracionCamara()
        images = sorted(glob.glob(self.mediaPath),key=lambda f: int(filter(str.isdigit, f)))
        # print str(images)
        print len(images)
        for index in range(len(images)):
            frame = cv2.imread(images[index])
            print 'imagen : '+ str(images[index])
            print 'index: ' + str(index)
            frameSiguiente = self.preProcessing(frame)

            if (index == 0): # Esto está temporal, mientras se le añade lo de múltiples vistas mas lo de bundle adjustment
                frameActual = frameSiguiente
            elif (index > 1 and index ):
                p1_filtrado, p2_filtrado,matches,base_features,next_features=self.featureMatching(frameActual,frameSiguiente)
                E,mask =cv2.findEssentialMat(p1_filtrado,p2_filtrado,self.mtx,cv2.RANSAC,0.999,1.0)
                points, R, t,mask= cv2.recoverPose(E, p1_filtrado,p2_filtrado)
                P2=np.array([[R[0,0],R[0,1], R[0,2], t[0]],[R[1,0],R[1,1], R[1,2], t[1]],[R[2,0],R[2,1], R[2,2], t[2]]],np.float32)
                tempEye=np.eye(3)
                P1=np.zeros((3,4))
                P1[:,:-1]=tempEye
                self.CheckCoherentRotation(R)

                p1_filtrado=np.expand_dims(p1_filtrado, axis=0)

                normp1=cv2.undistortPoints(p1_filtrado,self.mtx,self.dist)

                p2_filtrado=np.expand_dims(p2_filtrado, axis=0)
                normp2=cv2.undistortPoints(p2_filtrado,self.mtx,self.dist)

                point_4d_hom=cv2.triangulatePoints(P1, P2,normp1,normp2)
                point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1)) # Divide todo por el valor último del vector.
                puntos3d = point_4d[:3, :].T # Elimina el último valor del vector y acomoda a la matriz (n,3)
                puntosTotal=np.zeros((1,3))
                print 'shape puntos 3d : '+ str(puntos3d.shape)
                print 'shape puntos total: '+ str(puntosTotal.shape)
                puntosTotal=np.concatenate((puntosTotal,puntos3d))
                frameActual=frameSiguiente


#.........................GRAFICAR PUNTOS............................................

            if (index == (len(images)-1)):
                pointCloud = VtkPointCloud()
                
                for k in xrange(len(puntosTotal)):
                    point=puntosTotal[k,:3]
                    pointCloud.addPoint(point)

                print 'add point: ' + str(puntos3d[1,:3])
                pointCloud.renderPoints(pointCloud)

                cv2.imshow('frameActual',frameActual)
                cv2.imshow('frameSiguiente',frameSiguiente)
                cv2.waitKey(0)
                # if cv2.waitKey(500) & 0xFF == ord('q'):
                # break
                cv2.destroyAllWindows()
