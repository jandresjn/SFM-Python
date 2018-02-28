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
# o también se puede insertar manualmente los datos intrínsecos de la cámara [[f,cx,cy],[k1,k2,k3,k4,k5]]




# mapeo = sfm_neme('./videoInput/1.mp4','./calibrateCamera/camera_calibration.npz')
# mapeo = sfm_neme('./mediaInput/templeSet/*.png','./calibrateCamera/camera_calibration.npz')
# mapeo = sfm_neme('./mediaInput/templeSet/*.png',[800,400,225])
mapeo = sfm_neme('./mediaInput/templeSet2/*.png',[[1520.4,302.32,246.87],[0,0,0,0,0]])
# mapeo = sfm_neme('./mediaInput/templeSet/*.png',[[800.0,400.0,225.0],[0,0,0,0,0]])
# mapeo = sfm_neme('./mediaInput/templeSet3/*.jpg',[[6644.158796,974.051090,708.302330],[-0.152046, -0.050096, -0.001488, -0.000074, 0.000000]])
# mapeo = sfm_neme('./mediaInput/chepeSet/*.png','./calibrateCamera/camera_calibration.npz')
mapeo.sfmSolver()
print 'Fin'
