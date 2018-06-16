#!/usr/bin/python
# -*- coding: utf-8 -*-
 # SFM by Jorge Andres Jaramillo Neme ----> Thesis UAO
#...............................................................................................
import cv2
import glob
import numpy as np
import sba
from pyquaternion import Quaternion
from VtkPointCloud import VtkPointCloud
from imageUtilities import Imagen
from pointCloud2Pmvs_jajn import pointCloud2Pmvs
import os
class sfm_neme:
    'Clase para aplicar SFM usando Opencv 3.4'
    def __init__(self,videoPath,calibracionCamara):
        self.mediaPath = videoPath
        self.calibracionCamara = calibracionCamara
        # self.detector = cv2.xfeatures2d.SURF_create(400) # Lo defino general, por que lo utilizaré en 2 funciones.
        self.detector = cv2.xfeatures2d.SIFT_create() # Lo defino general, por que lo utilizaré en 2 funciones.
        # self.detector=cv2.ORB_create(2000)
        # self.detector=cv2.AKAZE_create()
        # self.FLANN_INDEX_KDTREE = 1
        # self.FLANN_INDEX_LSH = 6 # ORB
        # self.flann_params = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=5)
        # self.index_params= dict(algorithm = self.FLANN_INDEX_LSH,table_number = 20,key_size = 12, multi_probe_level = 2) # ORB
        # self.search_params = dict(checks=60) # ORB
        # self.matcher = cv2.FlannBasedMatcher(self.flann_params, self.search_params)
        self.matcher=cv2.BFMatcher()
        # self.matcher= cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
        # self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # self.matcher = cv2.FlannBasedMatcher(self.flann_params, {})
        # self.matcher = cv2.FlannBasedMatcher(self.index_params, self.search_params)
        # self.matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
        self.matches= None
        self.ratio = 0.75 # RATIO DE FILTRO DE INLIERS
        self.images_path= []
        self.index_winner_base=None
        self.arregloImagen=[]
        self.puntos3dTotal=np.empty((1,3))
        self.puntos3dIndices=None
        self.MIN_REPROJECTION_ERROR = 5# IMPORTANTE. ERROR DE REPROJECCIÓN DE FILTRADO CUANDO TRIANGULA.
        self.n_cameras=0
        self.n_points=0
        self.camera_indices=None
        self.point_indices=None
        self.points_2d=None
        self.temporalParams=None
        self.pointParams=None
        self.camera_params=None
        self.mCameraPose=[]

    def importarCalibracionCamara(self):
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
            focal=self.calibracionCamara[0][0]
            cx=self.calibracionCamara[0][1]
            cy=self.calibracionCamara[0][2]
            mtx=np.array([[focal,0,cx],[0,focal,cy],[0,0,1]],np.float32)
            # dist=np.zeros((1,5))
            dist=np.asarray(self.calibracionCamara[1],np.float32)
            print 'mtx: '
            print mtx
            print 'dist: '+ str(dist)
            return mtx, dist

    def createMtx(self,focal,cx,cy):
        mtx=np.array([[focal,0,cx],[0,focal,cy],[0,0,1]],np.float32)
        return mtx

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
        # print self.matches
        for (m,n) in self.matches:
            if m.distance < self.ratio*n.distance:
                filtered_matches.append(m)

        # for m in self.matches:
        #     if len(m) == 2 and m[0].distance < m[1].distance * self.ratio:
        #         filtered_matches.append(m[0])
        return filtered_matches

    def imageDistance(self,matches_subset): # ESTE NO LO USO...
        # Se suma las distancias de todos los matches ecuación.
        sumDistance = 0.0
        for match in matches_subset:
            sumDistance += match.distance
        return sumDistance

    def extractFeatures(self):
        for index,path in enumerate(self.images_path):
            img_base= cv2.imread(path)
            img_base= self.preProcessing(img_base)
            base_features, base_descs = self.detector.detectAndCompute(img_base, None)
            self.arregloImagen[index].features=base_features
            self.arregloImagen[index].descriptors=base_descs
            print "se extrae features en index: " + str(index)

    def featureMatching(self,base_features,base_descs,next_features,next_descs):


        self.matches = self.matcher.knnMatch(base_descs,next_descs, k=2)
        print "\t Match Count: ", len(self.matches)
        matches_subset = self.filter_matches()
        matches_count = float(len(matches_subset))
        print "\t Filtered Match Count: ", matches_count

        kp1 = []
        kp2 = []
        for match in matches_subset:
            kp1.append(base_features[match.queryIdx])# Lleno arreglo con sólo los keypoints comunes.
            kp2.append(next_features[match.trainIdx])
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
    def triangulateAndFind3dPoints(self,P1,P2,puntos1,puntos2,index1,index2):
        mtx1=self.createMtx(self.camera_params[index1,0],self.camera_params[index1,1],self.camera_params[index1,2])
        mtx2=self.createMtx(self.camera_params[index2,0],self.camera_params[index2,1],self.camera_params[index2,2])

        p1_filtrado=np.expand_dims(puntos1, axis=0)
        normp1=cv2.undistortPoints(p1_filtrado,mtx1,self.dist) # VUELVO Y DEJO EL DIST GENERAL POR QUE NO LO VOY A USAR EN EL BUNDLE...

        p2_filtrado=np.expand_dims(puntos2, axis=0)
        normp2=cv2.undistortPoints(p2_filtrado,mtx2,self.dist)
        # Encuentra las coordenadas homogéneas del punto 3d relacionado a los puntos 2d
        point_4d_hom=cv2.triangulatePoints(P1, P2,normp1,normp2)
        point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1)) # Divide todo por el valor último del vector.
        puntos3d = point_4d[:3, :].T # Elimina el último valor del vector y acomoda a la matriz (n,3)
        print "puntos3d sin filtrar por reproyección: " +str(len(puntos3d))
        rvec1= cv2.Rodrigues(np.asarray(P1[:3,:3],np.float32))
        rvec1= rvec1[0]
        tvec1=P1[:,-1]
        print rvec1.shape
        print tvec1.shape
        print puntos3d.shape
        projected1=cv2.projectPoints(puntos3d,rvec1,tvec1,mtx1,self.dist)
        projected1=np.squeeze(projected1[0])
        print "projected1: " + str(projected1.shape)
        rvec2 = cv2.Rodrigues(np.asarray(P2[:3,:3],np.float32))
        rvec2= rvec2[0]
        tvec2=P2[:,-1]
        projected2=cv2.projectPoints(puntos3d,rvec2,tvec2,mtx2,self.dist)
        projected2=np.squeeze(projected2[0])
        print "projected2: " + str(projected2.shape)
        puntos3dFiltrados=[]
        puntos2dAsociados1=[]
        puntos2dAsociados2=[]

        print "shape puntos y normp: " +str(puntos1.shape)+"  "+str(normp1.shape)
        # FILTRA Y DEJA SÓLO LOS PUNTOS 2D Y 3D QUE CUMPLAN EL CRITERIO DE EL MIN REPROJECTION ERROR...
        for index,point3d in enumerate(puntos3d):
            # print np.linalg.norm(projected1[index]-puntos1[index])
            if (np.linalg.norm(projected1[index]-puntos1[index]) < self.MIN_REPROJECTION_ERROR and np.linalg.norm(projected2[index]-puntos2[index]) < self.MIN_REPROJECTION_ERROR ):
                puntos3dFiltrados.append(point3d)
                puntos2dAsociados1.append(np.squeeze(puntos1[index]))
                puntos2dAsociados2.append(np.squeeze(puntos2[index]))
        puntos3dFiltrados = np.asarray(puntos3dFiltrados,np.float32)
        puntos2dAsociados1 = np.asarray(puntos2dAsociados1,np.float32)
        puntos2dAsociados2 = np.asarray(puntos2dAsociados2,np.float32)
        print "puntos3d filtrados por reproyección: " + str(puntos3dFiltrados.shape)
        print "puntos2d filtrados por reproyección: " + str(puntos2dAsociados1.shape)
        return puntos3dFiltrados,puntos2dAsociados1,puntos2dAsociados2

# Encuentra el mejor par de imágenes entre las primeras n para empezar la nube de puntos.
    def find2d3dPointsBase(self):
        # img_base= cv2.imread(self.images_path[0])
        # img_base= self.preProcessing(img_base)
        min_inliers_ratio=0
        index_winner=0
        p1_winner = None
        p2_winner = None
        # for index in range(len(self.images_path)-1):# Range corresponde con cuántas imagenes comparar después de la imagen base.
        for index in range(3): # COMPARA CON LAS PRIMERAS 6 IMÁGENES PARA HALLAR EL PRIMER PAR.
            print "-------------------------INICIA-----------------------------------"
            # img_actual=cv2.imread(self.images_path[index+1])
            # img_actual=self.preProcessing(img_actual)
            p1_filtrado, p2_filtrado,matches_subset,matches_count=self.featureMatching(self.arregloImagen[0].features,self.arregloImagen[0].descriptors,self.arregloImagen[index+1].features,self.arregloImagen[index+1].descriptors)
            M, mask = cv2.findHomography(p1_filtrado, p2_filtrado, cv2.RANSAC,10.0)
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
        self.index_winner_base= index_winner
        mtx=self.createMtx(self.camera_params[index_winner,0],self.camera_params[index_winner,1],self.camera_params[index_winner,2])
        E,mask =cv2.findEssentialMat(p1_winner,p2_winner,mtx,cv2.RANSAC,0.999,1.0)
        points, R, t,mask= cv2.recoverPose(E, p1_winner,p2_winner,mtx)
        P2=np.array([[R[0,0],R[0,1], R[0,2], t[0]],[R[1,0],R[1,1], R[1,2], t[1]],[R[2,0],R[2,1], R[2,2], t[2]]],np.float32)
        tempEye=np.eye(3)
        P1=np.zeros((3,4))
        P1[:,:-1]=tempEye
        self.CheckCoherentRotation(R)

        puntos3d_filtrado,p1_winner_filtrado,p2_winner_filtrado = self.triangulateAndFind3dPoints(P1,P2,p1_winner,p2_winner,0,index_winner)
# LOURAKIS------------------------------------------------------
        totalFrames=np.zeros((len(puntos3d_filtrado),1))
        totalFrames.fill(2)
        frame0=np.zeros((len(puntos3d_filtrado),1))
        frame0.fill(0)
        frame_winner=np.zeros((len(puntos3d_filtrado),1))
        frame_winner.fill(index_winner)

        self.pointParams=np.concatenate((puntos3d_filtrado,totalFrames,frame0,p1_winner_filtrado,frame_winner,p2_winner_filtrado),axis=1)
        print "POIINTPARAMS SHAPE:"+ str(self.pointParams.shape)
        print self.pointParams[0]
        # CON DIST
        # self.camera_params[index_winner,14:]=P2[:,3]
        # self.camera_params[0,14:]=P1[:,3]
        # SIN DIST
        self.camera_params[index_winner,(14-5):]=P2[:,3] # ES LA TRASLACIÓN
        self.camera_params[0,(14-5):]=P1[:,3]

        # P2Quaternion=sba.quaternions.quatFromRotationMatrix(P2[:,:3])
        P2Quaternion=Quaternion(matrix=P2[:,:3])
        P2QuatVec=np.array((P2Quaternion[0],P2Quaternion[1],P2Quaternion[2],P2Quaternion[3]),np.float32)
        # P2QuatVec=P2Quaternion.asVector()

        # self.camera_params[index_winner,10:14]=P2QuatVec

        self.camera_params[index_winner,(10-5):(14-5)]=P2QuatVec

        P1Quaternion=Quaternion(matrix=P1[:,:3])
        P1QuatVec=np.array((P1Quaternion[0],P1Quaternion[1],P1Quaternion[2],P1Quaternion[3]),np.float32)
        # P1Quaternion=sba.quaternions.quatFromRotationMatrix(P1[:,:3])
        # P1QuatVec=P1Quaternion.asVector()
        # self.camera_params[0,10:14]=P1QuatVec
        self.camera_params[0,(10-5):(14-5)]=P1QuatVec #NO DIST
        print "quaterion: " + str(P2QuatVec)
#-------------------------------ACTUALIZO DATOS AJUSTADOS POR BUNDLE-----------------------
        self.arregloImagen[index_winner].Pcam = P2
        self.arregloImagen[index_winner].p2dAsociados = p2_winner_filtrado
        self.arregloImagen[index_winner].p3dAsociados = puntos3d_filtrado
        self.arregloImagen[0].Pcam = P1
        self.arregloImagen[0].p2dAsociados=p1_winner_filtrado
        self.arregloImagen[0].p3dAsociados=puntos3d_filtrado
#--------------------------------------------------------------------------------------------------

        self.puntos3dTotal=puntos3d_filtrado

    def findP2dAsociadosAlineados(self,p2dAsociados,p3dAsociados,imgPoints1,imgPoints2):
        p2dAsociadosAlineados = []
        p3dAsociadosAlineados = []
        IndexAlineados2d=[]
        IndexAlineados3d=[]
        for index3d, point3d in enumerate(p3dAsociados):
            for index2d, point2d in enumerate(imgPoints1):
                if(point2d[0] == p2dAsociados[index3d][0] and point2d[1] == p2dAsociados[index3d][1]):
                    p2dAsociadosAlineados.append(imgPoints2[index2d])
                    p3dAsociadosAlineados.append(p3dAsociados[index3d])
                    IndexAlineados2d.append(index2d)
                    IndexAlineados3d.append(index3d)
        print "shape p2dAsociadosalineados: " + str(np.asarray(p2dAsociadosAlineados,np.float32).shape)
        print "shape p3dAsociadosalineados: " + str(np.asarray(p3dAsociadosAlineados,np.float32).shape)
        return np.asarray(p2dAsociadosAlineados,np.float32),np.asarray(p3dAsociadosAlineados,np.float32),np.asarray(IndexAlineados3d,np.int16),np.asarray(IndexAlineados2d,np.int16)

    # def organizaParams(self): # Compara los puntos3dfiltrados actuales, con la imagen anterior o más anterior (diferente a la imgComparada).
    # # Si hay puntos 3d en común organiza pointParams de tal manera que agrega los índices,columnas y puntos 2d que se parecen.

    def addView(self):
        print "------------------------------ARRANCA AÑADE VISTA: ----------------------------------------------------------"
        for imageIndex in range(len(self.arregloImagen)):
        # for imageIndex in range(27):
            imgActualPos=imageIndex+1
            contador=1
            # imagenActual = cv2.imread(self.images_path[imageIndex])
            imagenComparadaTercera=None
            bestInlierPoints1=None
            bestInlierPoints2=None
            bestInlierRatio = 0.0
            bestImgComparadaIndex=0
            paqueteApto=False
            contadorBuffer=10
            while (contador <= contadorBuffer and imgActualPos > 1 and  imageIndex != self.index_winner_base ):
                imgComparadaPos=imgActualPos-contador
                if ( imgComparadaPos != 0 ) : # Compara hasta que haya 10 imagenes o no haya más.
                    print "imagenActual: " + str(imgActualPos) + " imagen comparada : "+ str(imgComparadaPos)
                    if (self.arregloImagen[imgComparadaPos-1].p3dAsociados is not None): # imgComparadaPos -1 es el index...
                        print "La imagen comparada es apta, es la :  " + str(imgComparadaPos)
                        # Creo la variable imagenComparada, la cual es el imread() de la imagen a comparar que es apta por que tiene p3d asociados...
                        # imagenComparada=cv2.imread(self.images_path[imgComparadaPos-1])
                        p1_filtrado, p2_filtrado,matches_subset,matches_count=self.featureMatching(self.arregloImagen[imageIndex].features,self.arregloImagen[imageIndex].descriptors,self.arregloImagen[imgComparadaPos-1].features,self.arregloImagen[imgComparadaPos-1].descriptors)
                        M, mask = cv2.findHomography(p1_filtrado, p2_filtrado, cv2.RANSAC,10.0)
                        mask_inliers= float(cv2.countNonZero(mask))
                        mask_ratio= mask_inliers/matches_count
                        print "matches count: "+ str(matches_count)
                        print "Inliers ratio: " + str(mask_ratio)+ " Imagen Comparada: "+ str(imgComparadaPos)
                        if (mask_ratio > bestInlierRatio and matches_count > 50):
                            bestInlierRatio = mask_ratio
                            bestInlierPoints1=p1_filtrado
                            bestInlierPoints2=p2_filtrado
                            bestImgComparadaIndex =imgComparadaPos-1
                            paqueteApto = True
                            print "actualizo mejores"
                elif(imgComparadaPos <=0):
                    #contador = 11 # Si la imagen comparada es menor a la primera (no existe), se brinca el while...
                    break
                contador += 1
                print "aumento contador, contador Actual aumentado : "+ str(contador)
                # Teniendo los mejores puntos de la imagen ganadora, se procesa con la imagen actual para hallar los puntos 3d...
                if((contador > contadorBuffer or imgComparadaPos <=1) and (self.arregloImagen[bestImgComparadaIndex].p3dAsociados is not None) and paqueteApto == True ):
                    print "INICIA SOLVEPNPRANSAC CON LA IMAGEN GANADORA:"
                    print "se va a ransaquear con la mejor imagen : " + str(bestImgComparadaIndex+1)
                    # Uso función para asociar los puntos 2d que corresponden con la nube de puntos de la img comparada...
                    p2dAsociadosAlineados,p3dAsociadosAlineados,indexComparada,indexActual=  self.findP2dAsociadosAlineados(self.arregloImagen[bestImgComparadaIndex].p2dAsociados,self.arregloImagen[bestImgComparadaIndex].p3dAsociados,bestInlierPoints2,bestInlierPoints1)
                    mtx=self.createMtx(self.camera_params[imageIndex,0],self.camera_params[imageIndex,1],self.camera_params[imageIndex,2])
                    # dist=self.camera_params[imageIndex,5:10]
                    _,rvecs, tvecs,inliers = cv2.solvePnPRansac(p3dAsociadosAlineados,p2dAsociadosAlineados, mtx,self.dist,flags=cv2.SOLVEPNP_ITERATIVE)
                    R = cv2.Rodrigues(rvecs.T)
                    print "shape p3d ransaqueados: " +str(self.arregloImagen[bestImgComparadaIndex].p3dAsociados.shape)
                    print "shape p2d ransaqueados: " +str(np.squeeze(p2dAsociadosAlineados).shape)
                    print "shape tvecs" + str(tvecs)
                    # print "R  : " + str(R)
                    #self.CheckCoherentRotation(R[0]) pa saber si es válida la R

                    PcamBest=np.hstack((R[0],tvecs))
                    print "PcamBest: "
                    print PcamBest
                    puntos3d_filtrados,bestInlierPoints1_filtrados,bestInlierPoints2_filtrados = self.triangulateAndFind3dPoints(PcamBest,self.arregloImagen[bestImgComparadaIndex].Pcam,bestInlierPoints1,bestInlierPoints2,imageIndex,bestImgComparadaIndex)
                    # puntos3d_filtrados=np.round(puntos3d_filtrados,3)

                    print "puntos3d_filtrados encontrados: " + str(len(puntos3d_filtrados))
                    print self.puntos3dTotal.shape
                    print puntos3d_filtrados.shape
                    self.puntos3dTotal=np.concatenate((self.puntos3dTotal,puntos3d_filtrados))
                    print self.puntos3dTotal.shape

# ------------------------BUNDLE ADJUSTMENT CONFIG LOURAKIS----------------

                    totalFrames=np.zeros((len(puntos3d_filtrados),1))
                    totalFrames.fill(2)
                    frameActual=np.zeros((len(puntos3d_filtrados),1))
                    frameActual.fill(imageIndex)
                    frame_winner=np.zeros((len(puntos3d_filtrados),1))
                    frame_winner.fill(bestImgComparadaIndex)

                    pointParamsActual=np.concatenate((puntos3d_filtrados,totalFrames,frameActual,bestInlierPoints1_filtrados,frame_winner,bestInlierPoints2_filtrados),axis=1)

                    self.pointParams=np.concatenate((self.pointParams,pointParamsActual))
                    self.camera_params[imageIndex,(14-5):]=PcamBest[:,3]



#------------------------------------------------------------------------------------------

                    P2Quaternion=Quaternion(matrix=PcamBest[:,:3])
                    P2QuatVec=np.array((P2Quaternion[0],P2Quaternion[1],P2Quaternion[2],P2Quaternion[3]),np.float32)

#--------
                    # P2Quaternion=sba.quaternions.quatFromRotationMatrix(PcamBest[:,:3])
                    # P2QuatVec=P2Quaternion.asVector()
                    self.camera_params[imageIndex,(10-5):(14-5)]=P2QuatVec
                    print self.camera_params[imageIndex,(10-5):(14-5)]
                    # self.camera_params[imageIndex,10:14]=P2QuatVec
                    # print self.camera_params[imageIndex,10:14]
#------------
#                     #
#                     print "-----------------BUNDLE LOURAKIS
# SBA MODIFICA INICIA-------------------------------------------------------------------------------


                    # print 'image index: ' + str(imageIndex)
                    # if (imageIndex <= self.index_winner_base ):
                    #     np.savetxt('camarasMegaTotales',self.camera_params[:self.index_winner_base+1,:],"%4.5f")
                    # else: np.savetxt('camarasMegaTotales',self.camera_params[:imageIndex+1,:],"%4.5f")
                    #
                    # cameras= sba.Cameras.fromTxt('camarasMegaTotales')
                    # np.savetxt("puntosTotales",self.pointParams,"%4.5f")
                    # points = sba.Points.fromTxt('puntosTotales',cameras.ncameras)
                    # options = sba.Options.fromInput(cameras,points)
                    # options.nccalib=sba.OPTS_FIX5_INTR
                    # newcams, newpts, info = sba.SparseBundleAdjust(cameras,points,options)
                    # self.puntos3dTotal=newpts.B
                    # newcams.toTxt('nuevascamaraspro')
                    # print "necams: "
                    # print newcams.camarray
                    # # self.camera_params=newcams.toDylan()
                    # nuevas_camera_params=np.genfromtxt('nuevascamaraspro')
                    # self.camera_params[:len(nuevas_camera_params),:] = nuevas_camera_params
                    #
                    # PcamBest[:,3]=self.camera_params[imageIndex,(14-5):]
                    #
                    # P1QuatVec=self.camera_params[imageIndex,(10-5):(14-5)]
                    # print P1QuatVec
                    # P1Quaternion=Quaternion(array=P1QuatVec)
                    # # P1Quaternion=sba.quaternions.quatFromArray(P1QuatVec)
                    # # PcamBest[:,:3]=P1Quaternion.asRotationMatrix()
                    # PcamBest[:,:3]=P1Quaternion.rotation_matrix
                    # print PcamBest[:,:3]
                    #
                    # puntos3d_filtrados=self.puntos3dTotal[-len(puntos3d_filtrados):,:]
                    #
                    # self.pointParams[:,:3]=self.puntos3dTotal
                    #
                    #
                    #



#SBA MODIFICA END-------------------------------------------------------------------------------
                    self.arregloImagen[imageIndex].Pcam=PcamBest
                    self.arregloImagen[imageIndex].p2dAsociados=bestInlierPoints1_filtrados
                    self.arregloImagen[bestImgComparadaIndex].p2dAsociados=np.concatenate((self.arregloImagen[bestImgComparadaIndex].p2dAsociados,bestInlierPoints2_filtrados))
                    self.arregloImagen[imageIndex].p3dAsociados=puntos3d_filtrados
                    self.arregloImagen[bestImgComparadaIndex].p3dAsociados=np.concatenate((self.arregloImagen[bestImgComparadaIndex].p3dAsociados,puntos3d_filtrados))



#-------------------------------------------------------------------------------------------------------

                    print "Se actualizaron datos del paquete"
                    paqueteApto=False
                    break
    # def probar3Fotos(self):





    def rotate(self,points, rot_vecs):
        """Rotate points by given rotation vectors.

        Rodrigues' rotation formula is used.
        """
        theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
        with np.errstate(invalid='ignore'):
            v = rot_vecs / theta
            v = np.nan_to_num(v)
        dot = np.sum(v * points , axis=1)[:, np.newaxis]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        return cos_theta * points + sin_theta * np.cross( v,points) + (1 - cos_theta) * dot * v
# ESTE PROJECT ES EL POR DEFECTO, EL CUAL SU PROJECCIÓN NO ES IGUAL A LA DE OPENCV Y POR ELLO NO FUNCIONA EL BUNDLE.
    def project(self,points, camera_params):
        """Convert 3-D points to 2-D by projecting onto images."""
        points_proj = self.rotate(points, camera_params[:, :3])
        points_proj += camera_params[:, 3:6]
        # print "points: "
        # print points.shape
        points_proj =points_proj[:, :2] / points_proj[:, 2, np.newaxis]
        # print "camera_params: "
        # print camera_params.shape
        f = camera_params[:, 6]
        k1 = camera_params[:, 7]
        k2 = camera_params[:, 8]
        n = np.sum(points_proj**2, axis=1)
        r = 1 + k1 * n + k2 * n**2
        points_proj *= (r * f)[:, np.newaxis]
        # print "3"
        # print points_proj.shape
        return points_proj
# INTENTO ADAPTAR LA FUNCIÓN PARA QUE USE LA FUNCIÓN DE OPENCV, SIN EMBARGO ES MUY LENTA ...
# POR QUE NO PUEDE HACERSE DIRECTAMENTE SOBRE TODOS LOS DATOS. PARA USARSE SÓLO COMENTAR UN FUN() Y DESCOMENTAR EL OTRO...
    def project2(self, points,camera_params):
        points_proj = []
        rvec = []
        tvec= []
        for index in range(len(points)): # SE PROYECTA POR CADA PUNTO, SABIENDO QUE PODRÍA PROYECTARSE POR GRUPOS DE PUNTOS.

            points3d=points[index]
            points3d=np.expand_dims(points3d, axis=0)
            # print "points3d shape: " + str(points3d.shape)
            rvec1=camera_params[index,:3]
            rvec1=np.expand_dims(rvec1, axis=1)
            rvec=rvec1
            # print "rvec shape: " + str(rvec1.shape)
            tvec1=camera_params[index,3:6]
            tvec=tvec1


            # print "tvec shape: " + str(tvec1.shape)
            projected1=cv2.projectPoints(points3d,rvec1,tvec1,self.mtx,self.dist)
            projected1=np.squeeze(projected1[0])
            points_proj.append(projected1)

        return points_proj



    def sfmSolver(self):
        import time
        # SE IMPORTAN PARÁMETROS E INICIALIZAN PARÁMETROS.
        self.mtx,self.dist=self.importarCalibracionCamara()
        self.images_path = sorted(glob.glob(self.mediaPath),key=lambda f: int(filter(str.isdigit, f)))
        # init_camera_params=np.array([self.mtx[0][0],self.mtx[0][2],self.mtx[1][2],1,0,self.dist[0],self.dist[1],self.dist[2],self.dist[3],self.dist[4],0,0,0,0,0,0,0])
        init_camera_params=np.array([self.mtx[0][0],self.mtx[0][2],self.mtx[1][2],1,0,0,0,0,0,0,0,0])

        print "INIT CAMERA PARAMS:"
        print init_camera_params
        # time.sleep(999)

        for path in self.images_path:
            imagC = cv2.imread(path,1)
            self.arregloImagen.append(Imagen(path,init_camera_params,imagC))
        self.n_cameras=len(self.arregloImagen)

        self.camera_params=np.tile(init_camera_params,(self.n_cameras,1))
        print "camera_params init: "+ str(self.camera_params.shape)
        # print self.camera_params[0]
        self.extractFeatures()
        print "extrajo features bn"
        #SE HALLA LOS PUNTOS BASE...
        self.find2d3dPointsBase()
        # SACO ESTOS PUNTOS PARA ACOMODAR EL PUNTO DE VISIÓN DE LA CÁMARA VIZ..
        punto3dMediana=np.median(self.puntos3dTotal,axis=0)
        print "mediana: " + str(punto3dMediana)
        self.addView() # AÑADO LAS VISTAS....


# SBA IMPORTANTE LOURAKIS-------------------------------------------------------------------
        np.savetxt('camarasMegaTotales',self.camera_params[:,:],"%4.5f")
        print self.camera_params[:,5:].shape
        # cameras= sba.Cameras.fromDylan(self.camera_params)
        cameras= sba.Cameras.fromTxt('camarasMegaTotales')
        # cameras.zeroLocalRotationEstimates()
        print "shapes Pont Params: "
        print self.pointParams.shape
        # np.random.shuffle(self.pointParams)

        np.savetxt("puntosTotales",self.pointParams,"%4.5f")
        points = sba.Points.fromTxt('puntosTotales',cameras.ncameras)
        options = sba.Options.fromInput(cameras,points)
        # options.motstruct = sba.OPTS_STRUCT

        # options.camera = sba.OPTS_CAMS_NODIST
        # options.nccalib=sba.OPTS_FIX2_INTR
        options.nccalib=sba.OPTS_FIX5_INTR
        # options.camera=sba.OPTS_NO_CAMS
        # options.rot0params = cameras.rot0params
        # print self.pointParams[:,:3]

        # self.orderParams(self.pointParams)
        newcams, newpts, info = sba.SparseBundleAdjust(cameras,points,options)
        self.puntos3dTotal=newpts.B
        newcams.toTxt('nuevascamaraspro')
        # nuevas_camera_params=np.genfromtxt('nuevascamaraspro')
        # self.camera_params[:len(nuevas_camera_params),:] = nuevas_camera_params

        self.mCameraPose=[None] * len(self.images_path)

        # P_final=np.zeros((3,4))
        # for index_path,path in enumerate(self.images_path):
        #
        #     P_final[:,3]=self.camera_params[index_path,(14-5):]
        #     P1QuatVec=self.camera_params[index_path,(10-5):(14-5)]
        #     # P1Quaternion=sba.quaternions.quatFromArray(P1QuatVec)
        #     # P_final[:,:3]=P1Quaternion.asRotationMatrix()
        #     P1Quaternion=Quaternion(array=P1QuatVec)
        #     P_final[:,:3]=P1Quaternion.rotation_matrix
        #     self.mCameraPose[index_path]=P_final
        #     print self.mCameraPose[index_path]

        rawC = np.genfromtxt('nuevascamaraspro')
        for idx in range(rawC.shape[0]):
            self.camera_params[idx,:]=rawC[idx,:]
            P1QuatVec = rawC[idx,5:9]
            q = Quaternion(rawC[idx,5:9])
            R = q.rotation_matrix

            trans=np.expand_dims(rawC[idx,9:12], axis=1)
            CameraPose=np.hstack((R,trans))
            self.mCameraPose[idx]=CameraPose

# P1Quaternion=sba.quaternions.quatFromArray(P1QuatVec)
# PcamBest[:,:3]=P1Quaternion.asRotationMatrix()
        #--------------PMVS--------------------------------
        toPmvs=pointCloud2Pmvs(self.mtx,self.arregloImagen,self.mCameraPose)
        toPmvs.bundles2Pmvs()
        #-----------------------------------------------------







        print "puntos totales después: " + str(self.puntos3dTotal.shape)

        self.n_points=len(self.puntos3dTotal)
        print "n_points: " + str(self.n_points)
        # import time

        os.chdir("../..")
        print self.puntos3dTotal.shape
        file=open("PointCloud.txt","w")
        #GRAFICO LOS PUNTOS USANDO VTK
        pointCloud = VtkPointCloud(1e8,punto3dMediana[0],punto3dMediana[1],punto3dMediana[2])
        for k in xrange(len(self.puntos3dTotal)):
            point=self.puntos3dTotal[k,:3]
            pointdim=np.expand_dims(point, axis=1)
            # point[1]=point[1]*-1
            # point[0]=point[0]*-1
            # point = point * -1
            pointCloud.addPoint(point)
            np.savetxt(file,pointdim.T,"%5.3f")

        pointCloud.renderPoints(pointCloud)
