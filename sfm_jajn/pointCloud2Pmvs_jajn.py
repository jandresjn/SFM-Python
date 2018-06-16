#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np


class pointCloud2Pmvs:
    def __init__(self,K,ImagesPoses,cameraPoses):
        self.K=K
        self.ImagesPoses=ImagesPoses
        self.cameraPoses=cameraPoses

    def bundles2Pmvs(self):

        K_mat=np.array(self.K,np.float32)

        self.currentDir = os.getcwd()
        self.dirOut=self.currentDir+"/outPMVS"
        os.mkdir("outPMVS")
        os.chdir(self.dirOut)
        os.mkdir("models")
        os.mkdir("txt")
        os.mkdir("visualize")

        optionsFile = open("options.txt", "w")
        optionsFile.writelines("timages  -1 0 %s \n" % (len(self.ImagesPoses)-1))
        optionsFile.writelines("oimages 0 \n")
        optionsFile.writelines("level 1 \n")
        optionsFile.close()
        i=0
        for img in self.ImagesPoses:
            P=np.zeros((3,4))
            P=self.cameraPoses[i]
            P = np.array(self.cameraPoses[i],np.float32)
            # P = np.array(self.ImagesPoses[i].Pcam,np.float32)
            P=np.matmul(K_mat,P)


            os.chdir(self.dirOut+"/visualize")
            cv2.imwrite("%04d.jpg"%i,img.imagColor)

            os.chdir(self.dirOut+"/txt")
            txtFile = open("%04d.txt"%i, "w")
            txtFile.writelines("CONTOUR \n")

            for j in range(3):
                for k in range(4):
                    txtFile.writelines("%s "%P[j,k])
                txtFile.writelines("\n")
            txtFile.close()
            i+=1
