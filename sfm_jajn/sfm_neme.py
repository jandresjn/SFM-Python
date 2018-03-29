#!/usr/bin/python
# -*- coding: utf-8 -*-
 # SFM by Jorge Andrés Jaramillo Neme ----> Thesis UAO
#...............................................................................................
import cv2
import glob
import numpy as np
import sba
from pyquaternion import Quaternion
from VtkPointCloud import VtkPointCloud
from imageUtilities import Imagen

class sfm_neme:
    'Clase para aplicar SFM usando Opencv 3.4'
    def __init__(self,videoPath,calibracionCamara):
        self.mediaPath = videoPath
        self.calibracionCamara = calibracionCamara
        self.detector = cv2.xfeatures2d.SURF_create(400) # Lo defino general, por que lo utilizaré en 2 funciones.
        # self.detectorOrb=cv2.ORB_create()
        # self.detectorAkaze=cv2.AKAZE_create()
        self.FLANN_INDEX_KDTREE = 1
        # self.FLANN_INDEX_LSH = 6 # ORB
        self.flann_params = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=5)
        # self.index_params= dict(algorithm = self.FLANN_INDEX_LSH,table_number = 20,key_size = 12, multi_probe_level = 2) # ORB
        self.matcher = cv2.FlannBasedMatcher(self.flann_params, {})
        # self.search_params = dict(checks=50) # ORB
        # self.matcher = cv2.FlannBasedMatcher(self.index_params, self.search_params)
        # self.matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
        self.matches= None
        self.ratio = 0.75 # RATIO DE FILTRO DE INLIERS
        self.images_path= []
        self.index_winner_base=None
        self.arregloImagen=[]
        self.puntos3dTotal=np.empty((1,3))
        self.puntos3dIndices=None
        self.MIN_REPROJECTION_ERROR = 10 # IMPORTANTE. ERROR DE REPROJECCIÓN DE FILTRADO CUANDO TRIANGULA.
        self.n_cameras=0
        self.n_points=0
        self.camera_indices=None
        self.point_indices=None
        self.points_2d=None
        self.temporalParams=None
        self.pointParams=None
        self.camera_params=None

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

    def featureMatching(self,inputImage1,inputImage2):
        base_features, base_descs = self.detector.detectAndCompute(inputImage1, None)
        next_features, next_descs = self.detector.detectAndCompute(inputImage2, None)
        #-------------ORB----------------
        # base_features, base_descs = self.detectorOrb.detectAndCompute(inputImage1, None)
        # next_features, next_descs = self.detectorOrb.detectAndCompute(inputImage2, None)
        #------------------------------
        # base_features, base_descs = self.detectorAkaze.detectAndCompute(inputImage1, None)
        # next_features, next_descs = self.detectorAkaze.detectAndCompute(inputImage2, None)
        self.matches = self.matcher.knnMatch(next_descs,base_descs, k=2)
        print "\t Match Count: ", len(self.matches)
        matches_subset = self.filter_matches()
        matches_count = float(len(matches_subset))
        print "\t Filtered Match Count: ", matches_count
        # distance = self.imageDistance(matches_subset)
        # print "\t Distance from Key Image: ", distance
        # averagePointDistance = distance/float(len(matches_subset))
        # print "\t Average Distance: ", averagePointDistance
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
    def triangulateAndFind3dPoints(self,P1,P2,puntos1,puntos2,index1,index2):
        mtx1=self.createMtx(self.camera_params[index1,0],self.camera_params[index1,1],self.camera_params[index1,2])
        mtx2=self.createMtx(self.camera_params[index2,0],self.camera_params[index2,1],self.camera_params[index2,2])
        dist1=self.camera_params[index1,5:10]
        print "dist1: shape"+ str(dist1.shape)
        print dist1
        p1_filtrado=np.expand_dims(puntos1, axis=0)
        normp1=cv2.undistortPoints(p1_filtrado,mtx1,dist1)
        dist2=self.camera_params[index2,5:10]
        print "dist2: shape"+ str(dist2.shape)
        print dist2
        p2_filtrado=np.expand_dims(puntos2, axis=0)
        normp2=cv2.undistortPoints(p2_filtrado,mtx2,dist2)
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
        projected1=cv2.projectPoints(puntos3d,rvec1,tvec1,mtx1,dist1)
        projected1=np.squeeze(projected1[0])
        print "projected1: " + str(projected1.shape)
        rvec2 = cv2.Rodrigues(np.asarray(P2[:3,:3],np.float32))
        rvec2= rvec2[0]
        tvec2=P2[:,-1]
        projected2=cv2.projectPoints(puntos3d,rvec2,tvec2,mtx2,dist2)
        projected2=np.squeeze(projected2[0])
        print "projected2: " + str(projected2.shape)
        puntos3dFiltrados=[]
        puntos2dAsociados1=[]
        puntos2dAsociados2=[]
        # normp1= np.squeeze(normp1)
        # normp2= np.squeeze(normp2)
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
        img_base= cv2.imread(self.images_path[0])
        img_base= self.preProcessing(img_base)
        min_inliers_ratio=0
        index_winner=0
        p1_winner = None
        p2_winner = None
        # for index in range(len(self.images_path)-1):# Range corresponde con cuántas imagenes comparar después de la imagen base.
        for index in range(4): # COMPARA CON LAS PRIMERAS 6 IMÁGENES PARA HALLAR EL PRIMER PAR.
            print "-------------------------INICIA-----------------------------------"
            img_actual=cv2.imread(self.images_path[index+1])
            img_actual=self.preProcessing(img_actual)
            p1_filtrado, p2_filtrado,matches_subset,matches_count=self.featureMatching(img_base,img_actual)
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
        #--------------------------------------OPTICAL FLOW--------------------------------------------------
        # imagen_ganadora=cv2.imread(self.images_path[index_winner])
        # imagen_ganadora=self.preProcessing(imagen_ganadora)
        #----------------------------------------------------------------------------------------------------
        tempEye=np.eye(3)
        P1=np.zeros((3,4))
        P1[:,:-1]=tempEye
        self.CheckCoherentRotation(R)
        # OPTICAL FLOW YA NO SE PUEDE VER POR QUE NO ESTÁ ACOMODADO PARA  MÚLTIPLES VISTAS.
        #-----------------------------------OPTICAL FLOW------------------------------------------------------
        # flow = cv2.calcOpticalFlowFarneback(img_base,imagen_ganadora, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # print "flow" + str(flow.shape)
        # vectorFlow=[]
        # lpoints=[]
        # rpoints=[]
        # ancho,largo = img_base.shape
        # print ancho
        # for y in range(0,ancho-1):
        #    for x in range(0,largo-1):
        #        vectorFlow= flow[y,x]
        #        # print vectorFlow
        #        # print vectorFlow.shape
        #        # np.insert(lpoints,1,[x,y],axis=0)
        #        if (abs(vectorFlow[0]) > 0.0001 or abs(vectorFlow[1])> 0.0001):
        #            lpoints.append([x,y])
        #            rpoints.append([x+vectorFlow[1],y+vectorFlow[0]])
        #        # print vectorFlow[0]
        # lpoints=np.asarray(lpoints,np.float32)
        # rpoints=np.asarray(rpoints,np.float32)
        # print "lpoints y rpoints shapes: "
        # print lpoints.shape
        # print rpoints.shape
        # print "pwinners shapes"
        # print p1_winner.shape
#---------------------------TEMPORAL POR SI SE DESEA VER CON OPTICAL FLOW---------------------------------------
        puntos3d_filtrado,p1_winner_filtrado,p2_winner_filtrado = self.triangulateAndFind3dPoints(P1,P2,p1_winner,p2_winner,0,index_winner)
        # puntos3d = self.triangulateAndFind3dPoints(P1,P2,lpoints,rpoints) # COMENTAR ESTE Y ACTIVAR EL OTRO PARA QUITAR OPTICAL....
# Hago set de los parámetros requeridos para la comparación a las imagenes iniciales:

#----------------BUNDLE ADJUSTMENT LOURAKIS --------------------------------

        totalFrames=np.zeros((len(puntos3d_filtrado),1))
        totalFrames.fill(2)
        frame0=np.zeros((len(puntos3d_filtrado),1))
        frame0.fill(0)
        frame_winner=np.zeros((len(puntos3d_filtrado),1))
        frame_winner.fill(index_winner)

        self.pointParams=np.concatenate((puntos3d_filtrado,totalFrames,frame0,p1_winner_filtrado,frame_winner,p2_winner_filtrado),axis=1)
        print "POIINTPARAMS SHAPE:"+ str(self.pointParams.shape)
        print self.pointParams[0]

        self.camera_params[index_winner,14:]=P2[:,3]
        self.camera_params[0,14:]=P1[:,3]

        P2Quaternion=sba.quaternions.quatFromRotationMatrix(P2[:,:3])
        # P2Quaternion=Quaternion(matrix=P2[:,:3])
        # P2QuatVec=np.array((P2Quaternion[0],P2Quaternion[1],P2Quaternion[2],P2Quaternion[3]),np.float32)
        P2QuatVec=P2Quaternion.asVector()
        self.camera_params[index_winner,10:14]=P2QuatVec
        
        # P1Quaternion=Quaternion(matrix=P1[:,:3])
        P1Quaternion=sba.quaternions.quatFromRotationMatrix(P1[:,:3])
        # P1QuatVec=np.array((P1Quaternion[0],P1Quaternion[1],P1Quaternion[2],P1Quaternion[3]),np.float32)
        P1QuatVec=P1Quaternion.asVector()
        self.camera_params[0,10:14]=P1QuatVec
        print "quaterion: " + str(P2QuatVec)

        print "-----------------BUNDLE LOURAKIS---------------------------------"

        # camera_params_base=
        cameras= sba.Cameras.fromDylan(self.camera_params)
        np.savetxt("puntosTotales",self.pointParams,"%4.5f")
        points = sba.Points.fromTxt('puntosTotales',cameras.ncameras)
        newcams, newpts, info = sba.SparseBundleAdjust(cameras,points)
        self.puntos3dTotal=newpts.B

        self.camera_params=newcams.toDylan()

        P1[:,3]=self.camera_params[0,14:]
        # P1QuatVec=self.camera_params[0,10:14]
        # P1Quaternion=Quaternion(array=P1QuatVec)
        # P1[:,:3]=P1Quaternion.rotation_matrix

        P1QuatVec=self.camera_params[0,10:14]
        P1Quaternion=sba.quaternions.quatFromArray(P1QuatVec)
        P1[:,:3]=P1Quaternion.asRotationMatrix()

        P2[:,3]=self.camera_params[index_winner,14:]
        # P2QuatVec=self.camera_params[index_winner,10:14]
        # P2Quaternion=Quaternion(array=P2QuatVec)
        # P2[:,:3]=P2Quaternion.rotation_matrix

        P2QuatVec=self.camera_params[index_winner,10:14]
        P2Quaternion=sba.quaternions.quatFromArray(P2QuatVec)
        P2[:,:3]=P2Quaternion.asRotationMatrix()

        # puntos3d_filtrados=self.puntos3dTotal[-len(puntos3d_filtrados):,:]
        self.pointParams[:,:3]=self.puntos3dTotal


        puntos3d_filtrado=self.puntos3dTotal
#-------------------------------ACTUALIZO DATOS AJUSTADOS POR BUNDLE-----------------------
        self.arregloImagen[index_winner].Pcam = P2
        self.arregloImagen[index_winner].p2dAsociados = p2_winner_filtrado
        self.arregloImagen[index_winner].p3dAsociados = puntos3d_filtrado
        self.arregloImagen[0].Pcam = P1
        self.arregloImagen[0].p2dAsociados=p1_winner_filtrado
        self.arregloImagen[0].p3dAsociados=puntos3d_filtrado
#--------------------------------------------------------------------------------------------------
        # Configuración para Bundle Adjustment...
        # camera_indices1=np.empty(len(puntos3d_filtrado))
        # camera_indices1.fill(0)
        # camera_indices2=np.empty(len(puntos3d_filtrado))
        # camera_indices2.fill(index_winner)
        # self.camera_indices=np.concatenate([camera_indices1,camera_indices2])
        # print "Camera_indices inicial: " + str(self.camera_indices.shape)
        # self.puntos3dIndices =np.arange(len(puntos3d_filtrado))
        # self.point_indices=np.concatenate([np.arange(len(puntos3d_filtrado)),np.arange(len(puntos3d_filtrado))])
        # print "Point_indices inicial: " + str(self.point_indices.shape)
        # self.points_2d=np.concatenate((p1_winner_filtrado,p2_winner_filtrado),axis=0)
        # print "Point_2d Inicial: " + str(self.points_2d.shape)
        return puntos3d_filtrado

    def findP2dAsociadosAlineados(self,p2dAsociados,p3dAsociados,imgPoints1,imgPoints2):
        p2dAsociadosAlineados = []
        p3dAsociadosAlineados = []
        for index3d, point3d in enumerate(p3dAsociados):
            for index2d, point2d in enumerate(imgPoints1):
                if(point2d[0] == p2dAsociados[index3d][0] and point2d[1] == p2dAsociados[index3d][1]):
                    p2dAsociadosAlineados.append(imgPoints2[index2d])
                    p3dAsociadosAlineados.append(p3dAsociados[index3d])
        print "shape p2dAsociadosalineados: " + str(np.asarray(p2dAsociadosAlineados,np.float32).shape)
        print "shape p3dAsociadosalineados: " + str(np.asarray(p3dAsociadosAlineados,np.float32).shape)
        return np.asarray(p2dAsociadosAlineados,np.float32),np.asarray(p3dAsociadosAlineados,np.float32)

    def addView(self):
        print "------------------------------ARRANCA AÑADE VISTA: ----------------------------------------------------------"
        for imageIndex in range(len(self.arregloImagen)):
        # for imageIndex in range(27):
            imgActualPos=imageIndex+1
            contador=1
            imagenActual = cv2.imread(self.images_path[imageIndex])
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
                        imagenComparada=cv2.imread(self.images_path[imgComparadaPos-1])
                        p1_filtrado, p2_filtrado,matches_subset,matches_count=self.featureMatching(imagenActual,imagenComparada)
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
                    p2dAsociadosAlineados,p3dAsociadosAlineados=  self.findP2dAsociadosAlineados(self.arregloImagen[bestImgComparadaIndex].p2dAsociados,self.arregloImagen[bestImgComparadaIndex].p3dAsociados,bestInlierPoints2,bestInlierPoints1)
                    mtx=self.createMtx(self.camera_params[imageIndex,0],self.camera_params[imageIndex,1],self.camera_params[imageIndex,2])
                    dist=self.camera_params[imageIndex,5:10]
                    _,rvecs, tvecs, inliers = cv2.solvePnPRansac(p3dAsociadosAlineados,p2dAsociadosAlineados, mtx,dist)
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
                    print "puntos3d_filtrados encontrados: " + str(len(puntos3d_filtrados))
                    self.puntos3dTotal=np.concatenate((self.puntos3dTotal,puntos3d_filtrados))
                    # print "chequeo dimensiones para concatenar  p2dAsociados: " + str(self.arregloImagen[bestImgComparadaIndex].p2dAsociados.shape)
                    # print "chequeo dimensiones para concatenar  bestInlierPoints2_filtrados: " + str(bestInlierPoints2_filtrados.shape)
                    # print "shape p2dAsociado Actualizado: "+ str(self.arregloImagen[bestImgComparadaIndex].p2dAsociados.shape)
                    # print "shape p3dAsociado Actualizado: "+ str(self.arregloImagen[bestImgComparadaIndex].p3dAsociados.shape)
# ------------------------BUNDLE ADJUSTMENT CONFIG LOURAKIS----------------
                    totalFrames=np.zeros((len(puntos3d_filtrados),1))
                    totalFrames.fill(2)
                    frameActual=np.zeros((len(puntos3d_filtrados),1))
                    frameActual.fill(imageIndex)
                    frame_winner=np.zeros((len(puntos3d_filtrados),1))
                    frame_winner.fill(bestImgComparadaIndex)

                    pointParamsActual=np.concatenate((puntos3d_filtrados,totalFrames,frameActual,bestInlierPoints1_filtrados,frame_winner,bestInlierPoints2_filtrados),axis=1)

                    self.pointParams=np.concatenate((self.pointParams,pointParamsActual))
                    self.camera_params[imageIndex,14:]=PcamBest[:,3]





                    # P2Quaternion=Quaternion(matrix=PcamBest[:,:3])
                    # P2QuatVec=np.array((P2Quaternion[0],P2Quaternion[1],P2Quaternion[2],P2Quaternion[3]),np.float32)
                    P2Quaternion=sba.quaternions.quatFromRotationMatrix(PcamBest[:,:3])
                    P2QuatVec=P2Quaternion.asVector()
                    self.camera_params[imageIndex,10:14]=P2QuatVec
                    print self.camera_params[imageIndex,10:14]

                    print "-----------------BUNDLE LOURAKIS---------------------------------"

                    # camera_params_base=
                    cameras= sba.Cameras.fromDylan(self.camera_params)
                    np.savetxt("puntosTotales",self.pointParams,"%4.5f")
                    points = sba.Points.fromTxt('puntosTotales',cameras.ncameras)
                    newcams, newpts, info = sba.SparseBundleAdjust(cameras,points)
                    self.puntos3dTotal=newpts.B
                    # print "shape self camera params antes: "
                    # print self.camera_params
                    # print "shape self camera params después: "
                    self.camera_params=newcams.toDylan()
                    # print self.camera_params
                    # P2[:,3]=self.camera_params[index_winner,14:]
                    PcamBest[:,3]=self.camera_params[imageIndex,14:]


                    # P2QuatVec=self.camera_params[index_winner,10:14]
                    # P2Quaternion=sba.quaternions.quatFromArray(P2QuatVec)
                    # P2[:,:3]=P2Quaternion.asRotationMatrix()

                    P1QuatVec=self.camera_params[imageIndex,10:14]
                    print P1QuatVec
                    # P1Quaternion=Quaternion(array=P1QuatVec)
                    P1Quaternion=sba.quaternions.quatFromArray(P1QuatVec)
                    PcamBest[:,:3]=P1Quaternion.asRotationMatrix()
                    # PcamBest[:,:3]=P1Quaternion.rotation_matrix

                    puntos3d_filtrados=self.puntos3dTotal[-len(puntos3d_filtrados):,:]

                    self.pointParams[:,:3]=self.puntos3dTotal

                    self.arregloImagen[imageIndex].Pcam=PcamBest
                    self.arregloImagen[imageIndex].p2dAsociados=bestInlierPoints1_filtrados
                    self.arregloImagen[bestImgComparadaIndex].p2dAsociados=np.concatenate((self.arregloImagen[bestImgComparadaIndex].p2dAsociados,bestInlierPoints2_filtrados))
                    self.arregloImagen[imageIndex].p3dAsociados=puntos3d_filtrados
                    self.arregloImagen[bestImgComparadaIndex].p3dAsociados=np.concatenate((self.arregloImagen[bestImgComparadaIndex].p3dAsociados,puntos3d_filtrados))



                    # Configuración para Bundle Adjustment...
                    # camera_indices1=np.empty(len(puntos3d_filtrados))
                    # camera_indices1.fill(bestImgComparadaIndex)
                    # camera_indices2=np.empty(len(puntos3d_filtrados))
                    # camera_indices2.fill(imageIndex)
                    # self.camera_indices=np.concatenate([self.camera_indices,camera_indices1,camera_indices2])
                    # arreglo=np.arange(np.amax(self.puntos3dIndices)+1,np.amax(self.puntos3dIndices)+len(puntos3d_filtrados)+1)
                    # self.puntos3dIndices=np.concatenate([self.puntos3dIndices,arreglo])
                    # self.point_indices=np.concatenate([self.point_indices,arreglo,arreglo])
                    # self.points_2d=np.concatenate((self.points_2d,bestInlierPoints2_filtrados,bestInlierPoints1_filtrados),axis=0)
                    # FIN ACTUALIZACIÓN PARA BUNDLE
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

#DESCOMENTAR ESTE FUN Y COMENTAR EL OTRO, SI SE QUIERE USAR PROJECT2, ES CASI LO MISMO PERO SE AJUSTA dimensiones
# Y COSAS PARA QUE PROJECT2 FUNCIONE, LA DIFERENCIA ES EL NP.SQUEEZE...
#----------------------------------------------------------------

    # def fun(self,params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    #     """Compute residuals.
    #     `params` contains camera parameters and 3-D coordinates.
    #     """
    #     camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    #     points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    #     points_proj = self.project2(points_3d[point_indices.astype(int)], camera_params[camera_indices.astype(int)])
    #     points_proj = np.squeeze(points_proj)
    #     # print "points_proj shape: "
    #     # print points_proj.shape
    #     # print "camera params fun: "
    #     # print camera_params[camera_indices.astype(int)].shape
    #     return (points_proj - points_2d).ravel()

#-----------------------------------------------------------------------------------------------


    def fun(self,params, n_cameras, n_points, camera_indices, point_indices, points_2d):
        """Compute residuals.
        `params` contains camera parameters and 3-D coordinates.
        """

        camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
        points_3d = params[n_cameras * 9:].reshape((n_points, 3))
        points_proj = self.project(points_3d[point_indices.astype(int)], camera_params[camera_indices.astype(int)])
        # print "point shape : "+ str(point_3d.shape) + "camera params shape: "+ str(camera_params.shape)
        return (points_proj - points_2d).ravel()

    def bundle_adjustment_sparsity(self,n_cameras, n_points, camera_indices, point_indices):
        from scipy.sparse import lil_matrix
        m = camera_indices.size * 2
        n = n_cameras * 9 + n_points * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(camera_indices.size)
        for s in range(9):
            A[2 * i, camera_indices * 9 + s] = 1
            A[2 * i + 1, camera_indices * 9 + s] = 1

        for s in range(3):
            A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
            A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

        return A

    def sfmSolver(self):
        import time
        # SE IMPORTAN PARÁMETROS E INICIALIZAN PARÁMETROS.
        self.mtx,self.dist=self.importarCalibracionCamara()
        self.images_path = sorted(glob.glob(self.mediaPath),key=lambda f: int(filter(str.isdigit, f)))
        init_camera_params=np.array([self.mtx[0][0],self.mtx[0][2],self.mtx[1][2],1,0,self.dist[0],self.dist[1],self.dist[2],self.dist[3],self.dist[4],0,0,0,0,0,0,0])
        print "INIT CAMERA PARAMS:"
        print init_camera_params
        # time.sleep(999)

        for path in self.images_path:
            self.arregloImagen.append(Imagen(path,init_camera_params))
        self.n_cameras=len(self.arregloImagen)

        self.camera_params=np.tile(init_camera_params,(self.n_cameras,1))
        print "camera_params init: "+ str(self.camera_params.shape)
        # print self.camera_params[0]
        #SE HALLA LOS PUNTOS BASE...
        puntos3d=self.find2d3dPointsBase()
        # SACO ESTOS PUNTOS PARA ACOMODAR EL PUNTO DE VISIÓN DE LA CÁMARA VIZ..
        punto3dMediana=np.median(puntos3d,axis=0)
        print "mediana: " + str(punto3dMediana)
        # self.puntos3dTotal=puntos3d
        self.addView() # AÑADO LAS VISTAS....
        # print "POIINTPARAMS SHAPE:"+ str(self.pointParams.shape)
        # print self.pointParams[1200]
        # print self.camera_params




        # print "n_cameras: " + str(self.n_cameras)
        # print "Camera_indices final: " + str(self.camera_indices.shape)
        # print "Point_indices final: " + str(self.point_indices.shape)
        # print "Point_2d final: " + str(self.points_2d.shape)
        print "puntos totales después: " + str(self.puntos3dTotal.shape)
        # COJO LAS POSES Y LES APLICO RODRIGUES PARA ACOMODAR LOS DATOS IGUAL QUE LOS USADOS EN BUNDLE ADUSTMENT.

        # for idx,cameraParams in enumerate(self.camera_params):
        #     rvec = cv2.Rodrigues(np.asarray(self.arregloImagen[idx].Pcam[:3,:3],np.float32))
        #     rvec= np.squeeze(rvec[0])
        #     # print "rvec: " + str(rvec)
        #     tvec=self.arregloImagen[idx].Pcam[:,-1]
        #     # print "tvec: "+ str(tvec)
        #     cameraParams[:3]=rvec
        #     cameraParams[3:6]=tvec


            # print "cameraParams: " + str(cameraParams)
            # print "shape rvec: " + str(rvec.shape) + " shape tvec: " + str(tvec.T.shape)
        # print self.camera_params[4]

        # print np.median(puntos3d,axis=0)
         # Temporal
        self.n_points=len(self.puntos3dTotal)
        print "n_points: " + str(self.n_points)
        # import time

#         # a[np.argsort(a[:,1])]
#         # ESTO LO ACTIVO PARA REORDENAR TODOS LOS DATOS EN FUNCIÓN LOS POINT INDICES.
#         # ---------------------------------------------------------------------
#         self.camera_indices=np.expand_dims(self.camera_indices,axis=1)
#         print "camera_indices shape: " + str(self.camera_indices.shape)
#         self.point_indices=np.expand_dims(self.point_indices,axis=1)
#         print "point_indices shape: " + str(self.point_indices.shape)
#         print "point_2d shape: " + str(self.points_2d.shape)
#         self.temporalParams=np.concatenate((self.camera_indices,self.point_indices,self.points_2d),axis=1)
#         print "temporal params shape : " + str(self.temporalParams.shape)
#         self.temporalParams=self.temporalParams[np.argsort(self.temporalParams[:,1])]
#         # print self.temporalParams
#         self.camera_indices= self.temporalParams[:,0]
#         self.point_indices= self.temporalParams[:,1]
#         self.points_2d=self.temporalParams[:,2:]
#         #---------------------------------------------------------
# # ----------------INICIA EL BUNDLE ADJUSTMENT EL CUAL TIENE UN ALGORITMO ENFOCADO A GRANDES DATOS Y POR LO TANTO ANALIZA
# # TODAS LAS IMAGENES CON SUS RESPECTIVOS ÍNDICES DE UNA...
#         import matplotlib.pyplot as plt
#         x0 = np.hstack((self.camera_params.ravel(), self.puntos3dTotal.ravel()))
#         f0 = self.fun(x0, self.n_cameras, self.n_points, self.camera_indices, self.point_indices, self.points_2d)
#         f0= np.asarray(f0,np.float32)
#         plt.figure(1)
#         plt.plot(f0)
#         #DESCOMENTAR PARA QUE MUESTRE LA GRÁFICA DE REPROYECCIÓN DE ERROR SIN TENER QUE ESPERAR EL BUNDLE
#         plt.show()
#         time.sleep(999)
#         A = self.bundle_adjustment_sparsity(self.n_cameras, self.n_points, self.camera_indices, self.point_indices)
#         import time
#         from scipy.optimize import least_squares
#         t0 = time.time()
#         res = least_squares(self.fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
#                             args=(self.n_cameras, self.n_points, self.camera_indices, self.point_indices, self.points_2d))
#         t1 = time.time()
#         print res.x.shape
#         print("Optimization took {0:.0f} seconds".format(t1 - t0))
#         plt.figure(2)
#         plt.plot(res.fun)
#
#         print "xo shape: " + str(x0.shape)+ "x shape: "+ str(res.x.shape)
#         plt.show() # ESTE SHOW MUESTRA AMBAS GRÁFICAS.....
#         cam_pam_size=self.camera_params.ravel().size
#         print cam_pam_size
#         self.puntos3dTotal=res.x[cam_pam_size:].reshape((self.n_points, 3))

        print self.puntos3dTotal.shape
        #GRAFICO LOS PUNTOS USANDO VTK
        pointCloud = VtkPointCloud(1e8,punto3dMediana[0],punto3dMediana[1],punto3dMediana[2])
        for k in xrange(len(self.puntos3dTotal)):
            point=self.puntos3dTotal[k,:3]
            # point[1]=point[1]*-1
            # point[0]=point[0]*-1
            # point = point * -1
            pointCloud.addPoint(point)

        pointCloud.renderPoints(pointCloud)
