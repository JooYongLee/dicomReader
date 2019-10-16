#Based off of http://wiki.wxpython.org/GLCanvas
#Lots of help from http://wiki.wxpython.org/Getting%20Started
import sys
# sys.path.append("vtk_src")
import pickle
from vtk.util import numpy_support
sys.path.append("S3DGLPy")
import vtk

from PolyMesh import *
from Primitives3D import *

import vtk
from landmark_render import get_test_data, merge_data

import LaplacianMesh


landmark_color_table = []


rainbow_table =[
    [255, 0, 0],
    [255, 125, 0],
    [255, 255, 0],
    [125, 255, 0],
    [0, 255, 0],
    [0, 255, 125],
    [0, 255, 255],
    [0, 0, 255],
    [0, 5, 70],
    [100, 0, 255]
]
LANDMARK_MAXIMUM = 30

def init_color_table():

    length = len(rainbow_table)
    np_rainbow_table = np.array(rainbow_table)

    # landmark_color_table = []
    for i in range(LANDMARK_MAXIMUM):
        value = (i / (LANDMARK_MAXIMUM - 1) * (length - 1))
        upper = np.ceil(value)
        lower = np.floor(value)
        x = abs(value - upper)
        index = int(lower)
        # print(i, value, x, 1-x , index)

        if index == length - 1:
            color = np_rainbow_table[index]
        else:

            if x < 1e-10:
                x = 1

            color = x * np_rainbow_table[index] + (1 - x) * np_rainbow_table[index + 1]

        landmark_color_table.append(tuple(color / 255.0))


def render_show_pd(polydata_list, opacity=[], wire_frame=[]):
    actor_list = []
    if not opacity:
        opacity = len(polydata_list) * [1.0]

    if not wire_frame:
        wire_frame = len(polydata_list) * [1]

    for i,(pd, opa) in enumerate(zip(polydata_list,opacity)):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(pd)

        actor = vtk.vtkActor()
        actor.GetProperty().SetOpacity(opa)
        actor.SetMapper(mapper)
        if wire_frame[i] == 0:
            actor.GetProperty().SetRepresentationToWireframe()

        actor_list.append(actor)

    render_show(actor_list)

def render_show(actor_list, mark_list=[]):
    ren = vtk.vtkRenderer()

    for act in actor_list:
        ren.AddActor(act)

    for mark in mark_list:
        ren.AddActor(mark)


    # add_axes()
    # for act in DataReader.toothActors:
    #     ren.AddActor(act)
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(500, 500)



    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # iren.AddObserver("KeyPressEvent", keypress_event)

    iren.Initialize()
    iren.Start()

def stl_read(path):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(path)
    reader.Update()

    return reader.GetOutput()


def pipeine_to_act(polydata, wire=False, opacity=1.0):
    target_mapper = vtk.vtkPolyDataMapper()
    target_mapper.SetInputData(polydata)
    # tooth_mapper.SetInputConnection(reverse.GetOutputPort())

    target_mapper.SetScalarRange(polydata.GetScalarRange())
    target_mapper.ScalarVisibilityOn()

    target_actor = vtk.vtkActor()
    target_actor.SetMapper(target_mapper)
    target_actor.GetProperty().SetOpacity(opacity)

    if wire:
        target_actor.GetProperty().SetRepresentationToWireframe()

    return target_actor





def create_cone(pts, landmark_number):

    # self._landmark_number
    # coneSource = vtk.vtkConeSource()
    coneSource = vtk.vtkSphereSource()
    coneSource.SetRadius(.5)

    coneMapper = vtk.vtkDataSetMapper()
    coneMapper.SetInputConnection(coneSource.GetOutputPort())

    forwardCone =  vtk.vtkActor()#vtk.vtkActor()
    forwardCone.PickableOff()
    forwardCone.SetMapper(coneMapper)
    forwardCone.GetProperty().SetColor(landmark_color_table[landmark_number])

    forwardCone.SetPosition(*pts)

    return forwardCone

def _mkVtkIdList(it):
    """
    Makes a vtkIdList from a Python iterable. I'm kinda surprised that
     this is necessary, since I assumed that this kind of thing would
     have been built into the wrapper and happen transparently, but it
     seems not.

    :param it: A python iterable.
    :return: A vtkIdList
    """
    vil = vtk.vtkIdList()
    for i in it:
        vil.InsertNextId(int(i))

    return vil


def convert_np_2_vtk(vertices, faces):
    polydata = vtk.vtkPolyData()
    # data = polydata.GetPolys().GetData()
    # cells = polydata.GetPolys()

    polys = vtk.vtkCellArray()
    points = vtk.vtkPoints()

    for i in range(vertices.shape[0]):
        points.InsertNextPoint(tuple(vertices[i]))
        # verts.InsertNextCell(1)
        # verts.InsertCellPoint(i)

    id_list = faces.reshape([-1, 4])[:, 1:].astype(np.int)

    print(id_list.dtype)
    for ids in id_list:
        # polys.InsertNextCell(1)
        polys.InsertNextCell(_mkVtkIdList(tuple(ids)))




    polydata.SetPoints(points)
    polydata.SetPolys(polys)


    return polydata


def create_spheres(dicitems):
    spheres = []
    if isinstance(dicitems, dict):
        for number, pts in dicitems.items():
            spheres.append(create_cone(tuple(pts), number))
    elif isinstance(dicitems, np.ndarray):
        assert dicitems.shape[1] == 3
        for i, pts in enumerate(dicitems):
            spheres.append(create_cone(tuple(pts), i))

    return spheres

# if fields[0] == "v":
#     coords = [float(i) for i in fields[1:4]]
#     self.addVertex([coords[0], coords[1], coords[2]])
# if fields[0] == "f":
#     # Indices are numbered starting at 1 (so need to subtract that off)
#     indices = [int(re.split("/", s)[0]) - 1 for s in fields[1:]]
#     verts = [self.vertices[i] for i in indices]
#     self.addFace(verts)

def load_data(mesh, points, faces):
    pass

    # coords = [float(i) for i in fields[1:4]]
    faces = faces.astype(np.int)
    # faces = faces[:, 1:]
    print(faces.shape, "----------")
    for coords in points:
        mesh.addVertex([coords[0], coords[1], coords[2]])


    for indices in faces:
        # indices = [int(re.split("/", s)[0]) - 1 for s in fields[1:]]
        verts = [mesh.vertices[i] for i in indices]
        mesh.addFace(verts)


def get_closest_ids(points, marks):
    dist = np.square(np.expand_dims(points, axis=1) - np.expand_dims(marks, axis=0)).sum(axis=2)
    ids = dist.argmin(axis=0)
    return ids


def convert_act(data:PolyMesh):
    N = data.VPos.shape[0]

    polydata = vtk.vtkPolyData()
    # data = polydata.GetPolys().GetData()
    # cells = polydata.GetPolys()

    polys = vtk.vtkCellArray()
    points = vtk.vtkPoints()

    for i in range(N):
        points.InsertNextPoint(tuple(data.VPos[i]))

    for face in data.faces:
        ids = [vert.ID for vert in face.getVertices()]
        # polys.InsertNextCell(1)
        polys.InsertNextCell(_mkVtkIdList(ids))


    polydata.SetPoints(points)
    polydata.SetPolys(polys)
    return polydata


    # cn = data.faces[0]
def main():
    filename = "C:/Users/jurag/PycharmProjects/pyqt_ex/resources/001_Lower_Central__Incisors.stl"
    landmark_file = "testbookmarkt.pkl"
    # with open(landmark_file, "rb") as f:
    #     data = pickle.load(f)

    ren1 = vtk.vtkRenderer()
    ren2 = vtk.vtkRenderer()
    ren1.SetViewport(0, 0, 0.5, 1)
    ren2.SetViewport(0.5, 0, 1, 1)

    # add_axes()
    # for act in DataReader.toothActors:
    #     ren.AddActor(act)
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren1)
    renWin.AddRenderer(ren2)


    renWin.SetSize(500, 500)



    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    polydata, anchors, inds = get_test_data()

    # iren.AddObserver("KeyPressEvent", keypress_event)

    points = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
    polys_ = numpy_support.vtk_to_numpy(polydata.GetPolys().GetData())
    polys = polys_[:, 1:]

    # landmarks_data = data[11]['landmark']
    # points = data[11]['points']
    # polys = data[11]['polys']
    # pds = convert_np_2_vtk(points, polys)
    # print("points : {}, polys : {}".format(points.shape, polys.shape))
    #


    # ld_acts = create_spheres(ld_noise)
    # data_acts= pipeine_to_act(pds)

    # render_show([data_acts], ld_acts)
    # convert_np_2_vtk()


    data = PolyMesh()
    load_data(data, points, polys)
    print("converting complete")

    # render_show_pd([pds])
    # print(landmarks_data)
    # landmark_actors = create_spheres(landmarks_data)
    # render_show([acts], landmark_actors)

    out = LaplacianMesh.solveLaplacianMesh(data, anchors, inds, False)

    vtkpd = convert_act(out)
    merge_out = merge_data(vtkpd)
    data_acts = pipeine_to_act(merge_out)

    src_acts = pipeine_to_act(polydata)

    ren1.AddActor(src_acts)

    marks = create_spheres(anchors)

    for act in marks:
        ren1.AddActor(act)

    ren2.AddActor(data_acts)
    for act in marks:
        ren2.AddActor(act)


    # render_show([src_acts], ld_acts)

    iren.Initialize()
    iren.Start()

    # render_show_pd(([vtkpd]))


    # pd = stl_read(filename)
    # act = pipeine_to_act(pd)
    # render_show([act], landmark_actors)


    # render_show_pd([pd])

    # a = PolyMesh()






if __name__=="__main__":
    init_color_table()
    main()