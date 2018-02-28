#!/usr/bin/python
# -*- coding: utf-8 -*-
import vtk

class VtkPointCloud:
    'Clase para visualizar nubes de puntos  usando VTK'
    def __init__(self, maxNumPoints,fPointX,fPointY,fPointZ):
        self.maxNumPoints = maxNumPoints
        self.vtkPolyData = vtk.vtkPolyData()
        self.clearPoints()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)
        mapper.SetScalarVisibility(0)
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)
        self.vtkActor.GetProperty().SetColor(1,1,1)
        self.camera = vtk.vtkCamera()
        self.camera.SetPosition(0, 0,-10)
        # self.camera.SetFocalPoint(-0.33183324,-0.13742721,-9.25939941)
        self.camera.SetFocalPoint(fPointX,fPointY,fPointZ)
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
        renderer.SetBackground(0.0, 0.0, 0.0)
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
