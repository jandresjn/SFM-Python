#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
import cv2
import math
import glob
import numpy as np
<<<<<<< Updated upstream
from PIL import Image
from numpy import linalg

#-----------------------------------------------------------------------------------------------
#Para cargar imagenes en vez de un video...
#images = sorted(glob.glob('./TestImages/*.jpg'),key=lambda f: int(filter(str.isdigit, f)))
#print str(images)
#-----------------------------------------------------------------------------------------------
=======
from VtkPointCloud import VtkPointCloud
from imageUtilities import Imagen
>>>>>>> Stashed changes

class sfm_neme:
    'Clase para aplicar SFM usando Opencv 3.2'
    #Próximas variables
    def __init__(self,videoPath,calibracionCamara):
        self.videoPath = videoPath
        self.calibracionCamara = calibracionCamara
<<<<<<< Updated upstream
=======
        self.detector = cv2.xfeatures2d.SURF_create(400) # Lo defino general, por que lo utilizaré en 2 funciones.
        self.FLANN_INDEX_KDTREE = 1
        self.flann_params = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=5)
        self.matcher = cv2.FlannBasedMatcher(self.flann_params, {})
        self.matches= None
        self.ratio = 0.70 # Ratio para el filtro de matches...
        self.mtx,self.dist= None,None
        self.images_path= []
>>>>>>> Stashed changes

        self.arregloImagen=[]
        self.puntos3dTotal=np.empty((1,3))
    def importarCalibracionCamara(self):
<<<<<<< Updated upstream
        with np.load(self.calibracionCamara) as X:
            mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
            return mtx,dist,rvecs,tvecs

    def sfmSolver(self):
        cap = cv2.VideoCapture(self.videoPath)
        while(cap.isOpened()):
            success, frame = cap.read()
            #print str(ret)
            if (success and (int(round(cap.get(1))) % 10 == 0 or int(round(cap.get(1)))==1)):
                # Efectua la lectura cada n frames, en este caso 10.
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rows,cols = gray.shape
                M = cv2.getRotationMatrix2D((cols/2,rows/2),-90,1)
                dst = cv2.warpAffine(gray,M,(cols,rows))
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
mtx,dist,rvecs,tvecs = mapeo.importarCalibracionCamara()
print str(dist)
mapeo.sfmSolver()
=======
        if (type(self.calibracionCamara) is str):
            with np.load(self.calibracionCamara) as X:
                mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
                print 'Se seleccionó archivo .npz para la calibración'
                print 'mtx: '
                print mtx
                print 'dist: '+ str(dist)
                return mtx, dist
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
            return mtx, dist

    def preProcessing(self,inputImage): #Pre-procesa y rota frame.
        outputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
        # /¿outputImage = cv2.bilateralFilter(outputImage,-1,30,5);
        # rows,cols = outputImage.shape
        # M = cv2.getRotationMatrix2D((cols/2,rows/2),-90,1)
        # dst = cv2.warpAffine(outputImage,M,(cols,rows))
        return outputImage

    def filter_matches(self):
        print 'matches sin filtrar :' + str(len(self.matches))
        # Se encuentran los mejores matches según el ratio especificado y se agrupan en un array.
        filtered_matches = []
        # for m in self.matches:
        #     if len(m) == 2 and m[0].distance < m[1].distance * self.ratio:
        #         filtered_matches.append(m[0])
        for m,n in self.matches:
            if m.distance < self.ratio*n.distance:
                filtered_matches.append(m)
        return filtered_matches

    def imageDistance(self,matches_subset):
        # Se suma las distancias de todos los matches ecuación.
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
        matches_count = float(len(matches_subset))
        print "\t Filtered Match Count: ", matches_count
        distance = self.imageDistance(matches_subset)
        print "\t Distance from Key Image: ", distance
        averagePointDistance = distance/float(len(matches_subset))
        print "\t Average Distance: ", averagePointDistance
        kp1 = []
        kp2 = []
        for match in matches_subset:
            kp1.append(base_features[match.trainIdx])# Lleno arreglo con sólo los keypoints comunes.
            kp2.append(next_features[match.queryIdx])
        p1 = np.array([k.pt for k in kp1], np.float32) # Extraigo sólo las coordenadas(point) de los keypoint filtrados.
        p2 = np.array([k.pt for k in kp2], np.float32)
        return p1, p2,matches_subset,matches_count

# Chequea que la triangulación esté bien teniendo en cuenta el determinante de R.
    def CheckCoherentRotation(self,matR):
        EPS= 1E-7
        # print 'determinante: '+str(np.linalg.det(matR))
        if(np.fabs(np.linalg.det(matR))-1.0>EPS):
            print 'Error matriz inválida'
        print 'Matriz válida'
# Recibe las matrices de las cámaras y los puntos correspondientes de las imagenes, los cuales adecúa y ...
#... al final convierte los puntos homogéneos a euclídeos.
    def triangulateAndFind3dPoints(self,P1,P2,puntos1,puntos2):
        p1_filtrado=np.expand_dims(puntos1, axis=0)
        normp1=cv2.undistortPoints(p1_filtrado,self.mtx,self.dist)
        p2_filtrado=np.expand_dims(puntos2, axis=0)
        normp2=cv2.undistortPoints(p2_filtrado,self.mtx,self.dist)
        # Encuentra las coordenadas homogéneas del punto 3d relacionado a los puntos 2d
        point_4d_hom=cv2.triangulatePoints(P1, P2,normp1,normp2)
        point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1)) # Divide todo por el valor último del vector.
        puntos3d = point_4d[:3, :].T # Elimina el último valor del vector y acomoda a la matriz (n,3)
        return puntos3d

# Encuentra el mejor par de imágenes entre las primeras 10 para empezar la nube de puntos.
    def find2d3dPointsBase(self):
        img_base= cv2.imread(self.images_path[0])
        min_inliers_ratio=0
        index_winner=0
        p1_winner = None
        p2_winner = None
        # for index in range(len(self.images_path)-1):# Range corresponde con cuántas imagenes comparar después de la imagen base.
        for index in range(9):
            print "-------------------------INICIA-----------------------------------"
            img_actual=cv2.imread(self.images_path[index+1])
            p1_filtrado, p2_filtrado,matches_subset,matches_count=self.featureMatching(img_base,img_actual)
            M, mask = cv2.findHomography(p1_filtrado, p2_filtrado, cv2.RANSAC,5.0)
            mask_inliers= float(cv2.countNonZero(mask))
            mask_ratio= mask_inliers/matches_count
            print "index: " + str(index+1)
            print "Inliers ratio: " + str(mask_ratio)+ " Imagen: "+str(index+2)
            # CONDICIONES PARA EL MENOR INLIER, SE TIENE EN CUENTA EL INLIER RATIO Y LA CANTIDAD MÍNIMA DE MATCHES.
            if ((min_inliers_ratio > mask_ratio and matches_count > 120 ) or index == 0): # SFM toy lib usa 100.
                print  "mathces_count: " + str(matches_count)
                min_inliers_ratio=mask_ratio
                index_winner= index+1
                p1_winner=p1_filtrado
                p2_winner=p2_filtrado
            print "-------------------------ACABA-----------------------------------"
        print "La imagen con menos ratio inlier es imagen: "+ str(index_winner+1) + " e index: " + str(index_winner)
        E,mask =cv2.findEssentialMat(p1_winner,p2_winner,self.mtx,cv2.RANSAC,0.999,1.0)
        points, R, t,mask= cv2.recoverPose(E, p1_winner,p2_winner)
        P2=np.array([[R[0,0],R[0,1], R[0,2], t[0]],[R[1,0],R[1,1], R[1,2], t[1]],[R[2,0],R[2,1], R[2,2], t[2]]],np.float32)

        tempEye=np.eye(3)
        P1=np.zeros((3,4))
        P1[:,:-1]=tempEye
        self.CheckCoherentRotation(R)
        puntos3d = self.triangulateAndFind3dPoints(P1,P2,p1_winner,p2_winner)
# Hago set de los parámetros requeridos para la comparación a las imagenes iniciales:
        self.arregloImagen[index_winner].Pcam = P2
        self.arregloImagen[index_winner].p2dAsociados = p2_winner
        self.arregloImagen[index_winner].p3dAsociados = puntos3d
        self.arregloImagen[0].Pcam = P1
        self.arregloImagen[0].p2dAsociados=p1_winner
        self.arregloImagen[0].p3dAsociados=puntos3d
        return puntos3d

    def addView(self):
        for imageIndex in range(len(self.arregloImagen)-1):
        # for imageIndex in range(4):
            imgActualPos=imageIndex+1
            contador=1
            imagenActual = cv2.imread(self.images_path[imageIndex])
            bestInlierPoints1=None
            bestInlierPoints2=None
            bestInlierRatio = 0.0
            bestImgComparadaIndex=0
            bestM=None
            paqueteApto=False
            while (contador <= 10 and imgActualPos > 1 ):
                imgComparadaPos=imgActualPos-contador
                if ( imgComparadaPos != 0 ) : # La imagen 1  no puede compararse con anteriores...
                    print "imagenActual: " + str(imgActualPos) + " imagen comparada : "+ str(imgComparadaPos)
                    if (self.arregloImagen[imgComparadaPos-1].p3dAsociados is not None): # imgComparadaPos -1 es el index...
                        print "La imagen comparada apta es:  " + str(imgComparadaPos)
                        # Creo la variable imagenComparada, la cual es el imread() de la imagen a comparar que es apta por que tiene p3d asociados...
                        imagenComparada=cv2.imread(self.images_path[imgComparadaPos-1])
                        p1_filtrado, p2_filtrado,matches_subset,matches_count=self.featureMatching(imagenActual,imagenComparada)
                        M, mask = cv2.findHomography(p1_filtrado, p2_filtrado, cv2.RANSAC,5.0)
                        mask_inliers= float(cv2.countNonZero(mask))
                        mask_ratio= mask_inliers/matches_count
                        print "matches count: "+ str(matches_count)
                        print "Inliers ratio: " + str(mask_ratio)+ " Imagen Comparada: "+ str(imgComparadaPos)
                        if (mask_ratio > bestInlierRatio and matches_count > 20 ):
                            bestInlierRatio = mask_ratio
                            bestInlierPoints1=p1_filtrado
                            bestInlierPoints2=p2_filtrado
                            bestImgComparadaIndex =imgComparadaPos-1
                            bestM=M
                            paqueteApto = True
                            print "actualizo mejores"
                elif(imgComparadaPos <=0):
                    #contador = 11 # Si la imagen comparada es menor a la primera (no existe), se brinca el while...
                    break
                contador += 1
                print "aumento contador, contador Actual aumentado : "+ str(contador)
                # Teniendo los mejores puntos de la imagen ganadora, se procesa con la imagen actual para hallar los puntos 3d...
                if(((contador > 10 and paqueteApto == True) or imgComparadaPos <=1) and (self.arregloImagen[bestImgComparadaIndex].p3dAsociados is not None) and paqueteApto == True ):
                    print "Inicia ranqueamiento..."
                    print "se va a ransaquear con la mejor imagen : " + str(bestImgComparadaIndex+1)
                    bestM=np.linalg.inv(bestM)
                    p2dAsociadosAlineados= cv2.perspectiveTransform(np.array([self.arregloImagen[bestImgComparadaIndex].p2dAsociados]),bestM)
                    print "Se chequean shapes de entrada p3dAsociados : " + str(self.arregloImagen[bestImgComparadaIndex].p3dAsociados.shape) + " p2dAsociados: " +str(np.squeeze(p2dAsociadosAlineados).shape)
                    _,rvecs, tvecs, inliers = cv2.solvePnPRansac(self.arregloImagen[bestImgComparadaIndex].p3dAsociados,np.squeeze(p2dAsociadosAlineados), self.mtx, self.dist)
                    R = cv2.Rodrigues(rvecs.T)

                    print "ransac inliers: " + str(cv2.countNonZero(inliers))
                    print "shape p3d ransaqueados: " +str(self.arregloImagen[bestImgComparadaIndex].p3dAsociados.shape)
                    print "shape p2d ransaqueados: " +str(np.squeeze(p2dAsociadosAlineados).shape)
                    print "shape tvecs" + str(tvecs)
                    print "R  : " + str(R[0])
                    #self.CheckCoherentRotation(R[0]) pa saber si es válida la R
                    PcamBest=np.hstack((R[0],tvecs))
                    print "PcamBest: "
                    print PcamBest
                    self.arregloImagen[imageIndex].Pcam=PcamBest
                    self.arregloImagen[imageIndex].p2dAsociados=np.squeeze(p2dAsociadosAlineados)
                    print "chequeo dimensiones para concatenar  p2dAsociados: " + str(self.arregloImagen[bestImgComparadaIndex].p2dAsociados.shape)
                    print "chequeo dimensiones para concatenar  bestInlierPoints2: " + str(bestInlierPoints2.shape)
                    # self.arregloImagen[bestImgComparadaIndex].p2dAsociados=np.concatenate((self.arregloImagen[bestImgComparadaIndex].p2dAsociados,bestInlierPoints2))
                    # self.arregloImagen[bestImgComparadaIndex].p2dAsociados=np.concatenate((self.arregloImagen[bestImgComparadaIndex].p2dAsociados,bestInlierPoints2))
                    # puntos3d = self.triangulateAndFind3dPoints(PcamBest,self.arregloImagen[bestImgComparadaIndex].Pcam,bestInlierPoints1,bestInlierPoints2)
                    puntos3d = self.triangulateAndFind3dPoints(PcamBest,self.arregloImagen[bestImgComparadaIndex].Pcam,np.squeeze(p2dAsociadosAlineados),self.arregloImagen[bestImgComparadaIndex].p2dAsociados)
                    print "puntos3d encontrados" + str(len(puntos3d))
                    self.arregloImagen[imageIndex].p3dAsociados=puntos3d
                    # self.arregloImagen[bestImgComparadaIndex].p3dAsociados=np.concatenate((self.arregloImagen[bestImgComparadaIndex].p3dAsociados,puntos3d))
                    self.puntos3dTotal=np.concatenate((self.puntos3dTotal,puntos3d))
                    print "Se actualizaron datos del paquete"
                    paqueteApto=False
                    break



    def sfmSolver(self):
        self.mtx,self.dist=self.importarCalibracionCamara()
        self.images_path = sorted(glob.glob(self.mediaPath),key=lambda f: int(filter(str.isdigit, f)))
        for path in self.images_path:
            self.arregloImagen.append(Imagen(path))

        puntos3d=self.find2d3dPointsBase()
        self.puntos3dTotal=puntos3d
        print "puntos totales antes: " + str(puntos3d.shape)
        self.addView()
        print "puntos totales después: " + str(self.puntos3dTotal.shape)
         # Temporal
        pointCloud = VtkPointCloud()
        for k in xrange(len(self.puntos3dTotal)):
            point=self.puntos3dTotal[k,:3]
            pointCloud.addPoint(point)

        pointCloud.renderPoints(pointCloud)

#
#     def sfmSolver(self):
#         self.mtx,self.dist=self.importarCalibracionCamara()
#         self.images_path = sorted(glob.glob(self.mediaPath),key=lambda f: int(filter(str.isdigit, f)))
#         # print str(images_path)
#         print len(self.images_path)
#         for index in range(len(self.images_path)):
#             frame = cv2.imread(self.images_path[index])
#             print 'imagen : '+ str(self.images_path[index])
#             print 'index: ' + str(index)
#             frameSiguiente = self.preProcessing(frame)
#
#             if (index == 0): # Esto está temporal, mientras se le añade lo de múltiples vistas mas lo de bundle adjustment
#                 frameActual = frameSiguiente
#             elif (index >= 1 and index ):
#                 p1_filtrado, p2_filtrado,matches_subset=self.featureMatching(frameActual,frameSiguiente)
#                 E,mask =cv2.findEssentialMat(p1_filtrado,p2_filtrado,self.mtx,cv2.RANSAC,0.999,1.0)
#                 points, R, t,mask= cv2.recoverPose(E, p1_filtrado,p2_filtrado)
#                 P2=np.array([[R[0,0],R[0,1], R[0,2], t[0]],[R[1,0],R[1,1], R[1,2], t[1]],[R[2,0],R[2,1], R[2,2], t[2]]],np.float32)
#                 tempEye=np.eye(3)
#                 P1=np.zeros((3,4))
#                 P1[:,:-1]=tempEye
#                 self.CheckCoherentRotation(R)
#                 #Acopla las dimensiones de los arreglos a las requeridas por la función undistortPoints.
#                 p1_filtrado=np.expand_dims(p1_filtrado, axis=0)
#                 normp1=cv2.undistortPoints(p1_filtrado,self.mtx,self.dist)
#                 p2_filtrado=np.expand_dims(p2_filtrado, axis=0)
#                 normp2=cv2.undistortPoints(p2_filtrado,self.mtx,self.dist)
#                 #Encuentra las coordenadas homogéneas del punto 3d relacionado a los puntos 2d
#                 point_4d_hom=cv2.triangulatePoints(P1, P2,normp1,normp2)
#                 point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1)) # Divide todo por el valor último del vector.
#                 puntos3d = point_4d[:3, :].T # Elimina el último valor del vector y acomoda a la matriz (n,3)
#                 puntosTotal=np.zeros((1,3))
#                 print 'shape puntos 3d : '+ str(puntos3d.shape)
#                 print 'shape puntos total: '+ str(puntosTotal.shape)
#                 puntosTotal=np.concatenate((puntosTotal,puntos3d))
#                 #
#                 frameActual=frameSiguiente
#
# #.........................GRAFICAR PUNTOS............................................
#
#             if (index == (len(self.images_path)-1)):
#                 pointCloud = VtkPointCloud()
#
#                 for k in xrange(len(puntosTotal)):
#                     point=puntosTotal[k,:3]
#                     pointCloud.addPoint(point)
#
#                 print 'add point: ' + str(puntos3d[1,:3])
#                 pointCloud.renderPoints(pointCloud)
#
#                 cv2.imshow('frameActual',frameActual)
#                 cv2.imshow('frameSiguiente',frameSiguiente)
#                 cv2.waitKey(0)
#                 # if cv2.waitKey(500) & 0xFF == ord('q'):
#                 # break
#                 cv2.destroyAllWindows()
>>>>>>> Stashed changes
