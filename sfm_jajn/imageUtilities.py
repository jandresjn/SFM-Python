#!/usr/bin/python
# -*- coding: utf-8 -*-

class Imagen:
    'Contiene información importante para su comparación en SFM'

    def __init__(self,path,init_camera_params,imagColor):
        self.path = path
        self.p2dAsociados = None
        self.p3dAsociados = None
        self.Pcam = None
        self.index = None
        self.camera_params=init_camera_params
        self.features=None
        self.descriptors=None
        self.imagColor=imagColor

class Punto3d:
    'Contiene la información que relacióna individualmente un punto 3d'

    def __init__(self):
        pass
