#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SFM by Jorge Andrés Jaramillo Neme ----> Thesis UAO
#...............................................................................

from sfm_neme import sfm_neme
import numpy

# INSTRUCCIONES DE USO:
# Especificar path de dataset de fotos, tener en cuenta los nombres de las fotos por números.
# Se usa *.formato (i.e *.jpg)
#-----------------------------------------------------------------------------------------------
# Para los datos de calibración de cámara, se puede especificar el path del archivo de calibración .npz
# o también se puede insertar manualmente los datos intrínsecos de la cámara [f,cx,cy]




# mapeo = sfm_neme('./videoInput/1.mp4','./calibrateCamera/camera_calibration.npz')
# mapeo = sfm_neme('./mediaInput/templeSet/*.png','./calibrateCamera/camera_calibration.npz')
# mapeo = sfm_neme('./mediaInput/templeSet/*.png',[800,400,225])
mapeo = sfm_neme('./mediaInput/templeSet/*.png',[1520.4,302.32,246.87])
mapeo.sfmSolver()
print 'Fin'
