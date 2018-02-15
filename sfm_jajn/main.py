#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SFM by Jorge Andrés Jaramillo Neme ----> Thesis UAO
#...............................................................................

from sfm_neme import sfm_neme
import numpy

# INSTRUCCIONES DE USO:
# Especificar la ruta del video y el archivo de calibración. La adición de parámetros de cámara individuales y
# paquetes de fotos se adicionará en el siguiente commit...

mapeo = sfm_neme('./videoInput/1.mp4','./calibrateCamera/camera_calibration.npz')
mapeo.sfmSolver()
print 'Fin'
