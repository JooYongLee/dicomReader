import pydicom

import os
import numpy
from matplotlib import pyplot, cm
import vtk

PathDicom = "C:/Users/jurag/Downloads/CTData/CTData"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))
# print(lstFilesDCM)


dcmReader = vtk.vtkDICOMImageReader()
dcmReader.SetDirectoryName(PathDicom)
dcmReader.Update()
print(dcmReader)

compositeOpacity = vtk.vtkPiecewiseFunction()
compositeOpacity.AddPoint(-3024, 0)
compositeOpacity.AddPoint(-155.41, 0)
compositeOpacity.AddPoint(217.64, 0.68)
compositeOpacity.AddPoint(419.74, 0.83)
compositeOpacity.AddPoint(3071, 0.80)


color = vtk.vtkColorTransferFunction()
color.AddRGBPoint( -3024, 0, 0, 0 )
color.AddRGBPoint( -155.41, .55, .25, .15 )
color.AddRGBPoint( 217.64, .88, .60, .29 )
color.AddRGBPoint( 419.74, 1, .94, .95 )
color.AddRGBPoint( 3071, .83, .66, 1 )

volumeMapper = vtk.vtkSmartVolumeMapper()
volumeMapper.SetInputConnection(dcmReader.GetOutputPort())
volumeMapper.SetBlendModeToComposite()

volumeProperty = vtk.vtkVolumeProperty()
volumeProperty.ShadeOn()
volumeProperty.SetInterpolationType( vtk.VTK_LINEAR_INTERPOLATION)
volumeProperty.SetColor( color )
volumeProperty.SetScalarOpacity( compositeOpacity )

volume = vtk.vtkVolume()
volume.SetMapper( volumeMapper )
volume.SetProperty( volumeProperty )

renderer = vtk.vtkRenderer()
renderer.AddViewProp( volume )
renderer.SetBackground( .1, .2, .3 )
renderer.ResetCamera()


			# vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New());

vtkWindow = vtk.vtkRenderWindow()
# vtkWindow.SetInteractor(interactor)
vtkWindow.AddRenderer( renderer )
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
interactor.SetRenderWindow(vtkWindow)
vtkWindow.Render()

def DummyFunc1(obj, ev):
    print("Before Event")

def DummyFunc2(obj, ev):
    print("After Event")

# Print interator gives you a list of registered observers of the current
# interactor style
#print(interactor)

## adding priorities allow to control the order of observer execution
## (highest value first! if equal the first added observer is called first)
# interactor.RemoveObservers('LeftButtonPressEvent')

interactor.AddObserver('LeftButtonPressEvent', DummyFunc1, 1.0)
interactor.AddObserver('LeftButtonPressEvent', DummyFunc2, -1.0)
interactor.Initialize()
interactor.Start()