import pydicom

import os
import numpy
from matplotlib import pyplot, cm
import vtk

PathDicom = "C:/Users/jurag/Downloads/CTData/CTData"
# lstFilesDCM = []  # create an empty list
# for dirName, subdirList, fileList in os.walk(PathDicom):
#     for filename in fileList:
#         if ".dcm" in filename.lower():  # check whether the file's DICOM
#             lstFilesDCM.append(os.path.join(dirName,filename))
# print(lstFilesDCM)


def get_cylinder_figure_actor():
    cone = vtk.vtkCylinderSource()
    cone.SetRadius(5)
    cone.SetCenter(70, 70, 70)
    cone.SetHeight(100)
    # cone.SetResolution(1)

    #
    # In this example we terminate the pipeline with a mapper process object.
    # (Intermediate filters such as vtkShrinkPolyData could be inserted in
    # between the source and the mapper.)  We create an instance of
    # vtkPolyDataMapper to map the polygonal data into graphics primitives. We
    # connect the output of the cone souece to the input of this mapper.
    #
    coneMapper = vtk.vtkPolyDataMapper()
    coneMapper.SetInputConnection(cone.GetOutputPort())

    #
    # Create an actor to represent the cone. The actor orchestrates rendering of
    # the mapper's graphics primitives. An actor also refers to properties via a
    # vtkProperty instance, and includes an internal transformation matrix. We
    # set this actor's mapper to be coneMapper which we created above.
    #
    coneActor = vtk.vtkActor()
    coneActor.SetMapper(coneMapper)
    coneActor.GetProperty().SetColor(1, 0, 0)
    return coneActor

dcmReader = vtk.vtkDICOMImageReader()
dcmReader.SetDirectoryName(PathDicom)
dcmReader.Update()
print(dcmReader)

compositeOpacity = vtk.vtkPiecewiseFunction()
compositeOpacity.AddPoint(-3024, 0)
compositeOpacity.AddPoint(-155.41, 0)
# compositeOpacity.AddPoint(217.64, 0.68)
# compositeOpacity.AddPoint(419.74, 0.83)
compositeOpacity.AddPoint(3071, 0.40)
# compositeOpacity.AddPoint(3071, 0.80)



color = vtk.vtkColorTransferFunction()
color.AddRGBPoint( -3024, 0, 0, 0 )
color.AddRGBPoint( -155.41, .55, .25, .15 )
color.AddRGBPoint( 217.64, .88, .60, .29 )
color.AddRGBPoint( 419.74, 1, .94, .95 )
color.AddRGBPoint( 3071, .83, .66, 1 )


cylinder =   vtk.vtkCylinderSource()

volumeMapper = vtk.vtkSmartVolumeMapper()

volumeMapper.SetInputConnection(dcmReader.GetOutputPort())
# volumeMapper.SetInputConnection(cylinder.GetOutputPort())
volumeMapper.SetBlendModeToComposite()


volumeProperty = vtk.vtkVolumeProperty()
volumeProperty.ShadeOn()
volumeProperty.SetInterpolationType( vtk.VTK_LINEAR_INTERPOLATION)
volumeProperty.SetColor( color )
volumeProperty.SetScalarOpacity( compositeOpacity )

volume = vtk.vtkVolume()
volume.SetMapper( volumeMapper )
volume.SetProperty( volumeProperty )




# ren1= vtk.vtkRenderer()
# ren1.AddActor( coneActor )
# ren1.SetBackground( 0.1, 0.2, 0.4 )
cylinder_actor = get_cylinder_figure_actor()

renderer = vtk.vtkRenderer()
renderer.AddViewProp( volume )
renderer.AddActor(cylinder_actor)
renderer.SetBackground( .1, .2, .3 )
renderer.ResetCamera()



vtkWindow = vtk.vtkRenderWindow()
# vtkWindow.SetInteractor(interactor)

# vtkWindow.AddRenderer( ren1 )
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