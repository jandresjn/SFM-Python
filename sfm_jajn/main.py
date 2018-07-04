#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SFM by Jorge Andrés Jaramillo Neme ----> Thesis UAO
#...............................................................................

from sfm_neme import sfm_neme
import numpy as np

# INSTRUCCIONES DE USO:
# Especificar path de dataset de fotos, tener en cuenta los nombres de las fotos por números.
# Se usa *.formato (i.e *.jpg)
#-----------------------------------------------------------------------------------------------
# Para los datos de calibración de cámara, se puede especificar el path del archivo de calibración .npz
# o también se puede insertar manualmente los datos intrínsecos de la cámara [[f,cx,cy],[k1,k2,t1,t2,k3]]



# mapeo = sfm_neme('./videoInput/1.mp4','./calibrateCamera/camera_calibration.npz')
# mapeo = sfm_neme('./mediaInput/templeSet/*.png','./calibrateCamera/camera_calibration.npz')
# mapeo = sfm_neme('./mediaInput/templeSet/*.png',[800,400,225])
# mapeo = sfm_neme('./mediaInput/templeSet/*.png',[[1520.4,302.32,246.87],[0.0,0.0,0.0,0.0,0.0]])
# mapeo = sfm_neme('./mediaInput/templeSet/*.png',[[800.0,400.0,225.0],[0,0,0,0,0]])
# mapeo = sfm_neme('./mediaInput/crazyhorse/*.JPG',[[1.5204000000000001e+003, 3.0231999999999999e+002,2.4687000000000000e+002],[0.0,0.0,0.0,0.0,0.0]])
# mapeo = sfm_neme('./mediaInput/crazyhorse/*.JPG',[[2500.0, 384.0,512.0],[0.0,0.0,0.0,0.0,0.0]])
# mapeo = sfm_neme('./mediaInput/vaca/*.jpg',[[2000, 384.0,512.0],[0.0,0.0,0.0,0.0,0.0]])
# mapeo = sfm_neme('./mediaInput/crazyhorse/*.JPG',[[2000.0, 512.0 ,384.0],[0.0,0.0,0.0,0.0,0.0]])
mapeo = sfm_neme('./mediaInput/OldTownHall/*.jpg',[[1000.0, 1280.0 ,960.0],[0.0,0.0,0.0,0.0,0.0]])
# mapeo = sfm_neme('./mediaInput/chepeSet/*.png','./calibrateCamera/camera_calibration.npz')
# mapeo = sfm_neme('./mediaInput/bonsai/*.jpg','./calibrateCamera/camera_calibration.npz')
# mapeo = sfm_neme('./mediaInput/bonsai/*.jpg',[[1100, 650,650],[0.0,0.0,0.0,0.0,0.0]])
mapeo.sfmSolver()
print 'Fin'
