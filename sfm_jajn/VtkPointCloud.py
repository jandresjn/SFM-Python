#!/usr/bin/python
# -*- coding: utf-8 -*-
import vtk

class VtkPointCloud:
    'Clase para visualizar nubes de puntos  usando VTK'
    def __init__(self, maxNumPoints,fPointX,fPointY,fPointZ):
        colors = vtk.vtkNamedColors()

        # Set the colors.
        colors.SetColor("AzimuthArrowColor", [255, 77, 77, 255])
        colors.SetColor("ElevationArrowColor", [77, 255, 77, 255])
        colors.SetColor("RollArrowColor", [255, 255, 77, 255])
        colors.SetColor("SpikeColor", [255, 77, 255, 255])
        colors.SetColor("BkgColor", [26, 51, 102, 255])

        self.maxNumPoints = maxNumPoints
        self.vtkPolyData = vtk.vtkPolyData()
        self.clearPoints()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)
        mapper.SetScalarVisibility(0)
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)
        self.vtkActor.GetProperty().SetColor(1.0,1.0,1.0)
        self.vtkActor.GetProperty().SetPointSize(2)
        self.camera = vtk.vtkCamera()
        self.camera.SetPosition(0, 0,-10)
        # self.camera.SetFocalPoint(-0.33183324,-0.13742721,-9.25939941)
        self.camera.SetFocalPoint(fPointX,fPointY,fPointZ)

        self.camCS = vtk.vtkConeSource()
        self.camCS.SetHeight(1.5)
        self.camCS.SetResolution(12)
        self.camCS.SetRadius(0.4)

        self.camCBS = vtk.vtkCubeSource()
        self.camCBS.SetXLength(1.5)
        self.camCBS.SetZLength(0.8)
        self.camCBS.SetCenter(0.4, 0, 0)


        self.camAPD = vtk.vtkAppendPolyData()
        self.camAPD.AddInputConnection(self.camCBS.GetOutputPort())
        self.camAPD.AddInputConnection(self.camCS.GetOutputPort())

        self.transform = vtk.vtkTransform()
        self.transform.RotateWXYZ(0,90,1,0)
        self.transformFilter=vtk.vtkTransformPolyDataFilter()
        self.transformFilter.SetTransform(self.transform)
        self.transformFilter.SetInputConnection(self.camAPD.GetOutputPort())
        # self.transformFilter.SetInputConnection(self.camCS.GetOutputPort())
        self.transformFilter.Update()


        self.camMapper = vtk.vtkPolyDataMapper()
        self.camMapper.SetInputConnection(self.transformFilter.GetOutputPort())
        self.camActor = vtk.vtkLODActor()
        self.camActor.SetMapper(self.camMapper)
        self.camActor.SetScale(0.1, 0.1, 0.1)


        # self.camActor.SetFocalPoint(fPointX,fPointY,fPointZ)

    def addPoint(self, point):
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)
        else:
            r = random.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])
        self.vtkCells.Modified()
        self.vtkPoints.Modified()

    def clearPoints(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)

    def renderPoints(self, pointCloudObj):
        # Renderer
        renderer = vtk.vtkRenderer()
        renderer.AddActor(pointCloudObj.vtkActor)
        renderer.AddActor(self.camActor)
        renderer.GradientBackgroundOn();
        renderer.SetBackground(0.8, 0.9, 0.9)
        renderer.SetBackground2(0,0,1);
        transform = vtk.vtkTransform()
        transform.Translate(0.0, 0.0, 0.0)

        axes = vtk.vtkAxesActor()
        #  The axes are positioned with a user transform
        axes.SetUserTransform(transform)
        renderer.AddActor(axes)
        # renderer.ResetCamera()
        renderer.SetActiveCamera(self.camera)
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)
        renderWindow.Render()
        renderWindowInteractor.Start()
