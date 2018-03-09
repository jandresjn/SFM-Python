#!/usr/bin/python
# -*- coding: utf-8 -*-

class Imagen:
    'Contiene información importante para su comparación en SFM'

    def __init__(self,path,init_camera_params):
        self.path = path
        self.p2dAsociados = None
        self.p3dAsociados = None
        self.Pcam = None
        self.index = None
        self.camera_params=init_camera_params
