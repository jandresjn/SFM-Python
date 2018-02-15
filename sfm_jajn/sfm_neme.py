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
        #outputImage = cv2.bilateralFilter(outputImage,-1,30,5);
        rows,cols = outputImage.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),-90,1)
        dst = cv2.warpAffine(outputImage,M,(cols,rows))
        return dst

    def filter_matches(self):
        print 'matches sin filtrar :' + str(len(self.matches))
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
        return p1, p2,matches_subset,base_features,next_features

# Checa que la triangulación esté bien teniendo en cuenta el determinante de R
    def CheckCoherentRotation(self,matR):
        EPS= 1E-7
        #print 'determinante: '+str(np.linalg.det(matR))
        if(np.fabs(np.linalg.det(matR))-1.0>EPS):
            print 'Error matriz inválida'
            return False
        print 'Matriz válida'
        return True
#
    def sfmSolver(self):
        self.mtx,self.dist,self.rvecs,self.tvecs=self.importarCalibracionCamara()
        cap = cv2.VideoCapture(self.videoPath)
        while(cap.isOpened()):
            success, frame = cap.read()
            #print str(ret)
            if (success and (int(round(cap.get(1))) % 30 == 0 or int(round(cap.get(1)))==1) and cap.get(1)<=30):# Añado para usar 2 imágenes.
                # Efectua la lectura cada n frames, en este caso 5.
                frameSiguiente = self.preProcessing(frame)
                if (cap.get(1) == 1): # Esto está temporal, mientras se le añade lo de múltiples vistas mas lo de bundle adjustment
                    frameActual = frameSiguiente
                else:
                    kp1_filtrado , kp2_filtrado,matches,base_features,next_features=self.featureMatching(frameActual,frameSiguiente)
                    E,mask =cv2.findEssentialMat(kp1_filtrado,kp2_filtrado,self.mtx,cv2.RANSAC,0.999,1.0)
                    points, R, t, mask = cv2.recoverPose(E, kp1_filtrado,kp2_filtrado)
                    P2=np.array([[R[0,0],R[0,1], R[0,2], t[0]],[R[1,0],R[1,1], R[1,2], t[1]],[R[2,0],R[2,1], R[2,2], t[2]]],np.float32)
                    tempEye=np.eye(3)
                    P1=np.zeros((3,4))
                    P1[:,:-1]=tempEye
                    resp=self.CheckCoherentRotation(R)
                    # kp1_filtrado_h=np.ones((len(kp1_filtrado),3),np.float32)
                    # kp1_filtrado_h[:,:-1]=kp1_filtrado
                    kp1_filtrado=np.expand_dims(kp1_filtrado, axis=0)
                    # print str(kp1_filtrado.shape)
                    # print str(kp1_filtrado)
                    normp1=cv2.undistortPoints(kp1_filtrado,self.mtx,self.dist)
                    # print 'shape_normp1'+ str(normp1.shape)
                    # normp1_homogeneo=np.ones((len(normp1[0]),3),np.float32)
                    # print 'shape_norm_homogeneo'+ str(normp1_homogeneo.shape)
                    # normp1_homogeneo[:,:-1] = normp1
                    # print 'K: '+ str(self.mtx)
                    # print 'dist: '+ str(self.dist)
                    # print 'normp1'
                    # print str(normp1)
                    kp2_filtrado=np.expand_dims(kp2_filtrado, axis=0)
                    normp2=cv2.undistortPoints(kp2_filtrado,self.mtx,self.dist)
                    # normp2_homogeneo=np.ones((len(normp2[0]),3),np.float32)
                    # normp2_homogeneo[:,:-1] = normp2
                    # print 'normp2'
                    # print str(normp2_homogeneo)
                    puntos3d=cv2.triangulatePoints(P1, P2,normp1,normp2)
                    puntos3d=np.squeeze(puntos3d).transpose()
                    print 'puntos3d'
                    print puntos3d
                    print 'shape puntos 3d : '+ str(puntos3d.shape)
                    #puntos3d=np.squeeze(puntos3d)
                    # eucliPoints=cv2.convertPointsFromHomogeneous(puntos3d.transpose())
                    # eucliPoints=np.squeeze(eucliPoints)
                    # print 'eucliPoints'
                    # print eucliPoints
                    # print 'eucliPoints'
                    # print eucliPoints[3,:]
                    # print 'lenght eucli: '+ str(len(eucliPoints))
                    # print 'shape_euclipoints: '+ str(eucliPoints.shape)
                    pointCloud = VtkPointCloud()
                    for k in xrange(len(puntos3d)):
                        #point = 20*(random.rand(3)-0.5)
                        point=puntos3d[k,:3]
                        pointCloud.addPoint(point)
                    pointCloud.renderPoints(pointCloud)

                    cv2.imshow('frameActual',frameActual)
                    cv2.imshow('frameSiguiente',frameSiguiente)
                    if cv2.waitKey(500) & 0xFF == ord('q'):
                        break
            elif success == False:
                cv2.waitKey(0)
                break
        cap.release()
        cv2.destroyAllWindows()
