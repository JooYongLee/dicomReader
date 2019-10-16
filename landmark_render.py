import worklist_convert
import vtk
import numpy as np
import vtk_utils
import pickle
from localData import AnatomyReader
from vtk.util import numpy_support
from sklearn.neighbors import NearestNeighbors
# reader = AnatomyReader()
def get_test_data():
    reader = vtk.vtkSTLReader()
    reader.SetFileName("resources/007_Lower_Second_Molar.stl")
    reader.Update()
    pd = reader.GetOutput()

    with open("result/landmark_sample.pkl", "rb") as f:
        data = pickle.load(f)

    tau = data["number"]
    source = data["source"]
    target = data["target"]
    root_inds = data["root_inds"]

    target_marks = worklist_convert.create_spheres(target)
    source_marks = worklist_convert.create_spheres(source)
    worklist_convert.apply_color(source_marks)
    worklist_convert.apply_color(target_marks)
    tooth_actor = worklist_convert.polydata_to_acctor(pd)
    tooth_actor.GetProperty().SetOpacity(0.5)

    points = numpy_support.vtk_to_numpy(pd.GetPoints().GetData())
    polys = numpy_support.vtk_to_numpy(pd.GetPolys().GetData()).reshape([-1, 4])

    print(points.shape, polys.shape)

    kept_inds = np.zeros([points.shape[0]], dtype=np.bool)
    # N
    kept_inds[root_inds] = True
    # M X 3
    cells = polys[:, 1:]
    # M X 3
    root_poly_inds = np.all(kept_inds[cells], axis=1)

    root_points = points[root_inds]
    root_cells = cells[root_poly_inds]

    # = np.arange([points.shape[0]])
    inverse_map = np.empty([points.shape[0]], dtype=np.int)
    inverse_map[:] = -1
    inverse_map[root_inds] = np.arange(root_inds.shape[0])

    inverse_indices = inverse_map[root_cells.ravel()]
    inverse_indices = inverse_indices.reshape(root_cells.shape)
    assert 0 <= inverse_indices.min() and inverse_indices.max() < root_cells.shape[0]


    root_cells = np.concatenate( [np.ones([inverse_indices.shape[0], 1]) * 3, inverse_indices], axis=1)
    vtk_point_array = numpy_support.numpy_to_vtk(root_points, array_type=vtk.VTK_FLOAT)
    vtk_indices_array = numpy_support.numpy_to_vtk(root_cells, array_type=vtk.VTK_ID_TYPE)

    vtkpoints = vtk.vtkPoints()
    vtkpoints.SetData(vtk_point_array)
    vtkcell = vtk.vtkCellArray()
    vtkcell.SetCells(vtk_point_array.GetNumberOfTuples(), vtk_indices_array)
    root_polydata = vtk.vtkPolyData()
    root_polydata.SetPoints(vtkpoints)
    root_polydata.SetPolys(vtkcell)

    # worklist_convert.render_show_pd([root_polydata])
    actor = worklist_convert.polydata_to_acctor(root_polydata)

    neighbor = NearestNeighbors()
    neighbor.fit(root_points)
    _, inds = neighbor.kneighbors(source)
    i = inds[-1, 0]


    marks = worklist_convert.create_spheres(root_points[i:i+1])
    target_marks = worklist_convert.create_spheres(target[-1:])

    thresh = 1

    fix_inds, = np.where(root_points[:, 2] > root_points.max(axis=0)[2] - thresh)
    samples = root_points[fix_inds]

    print(samples.shape)
    target_num = 4
    # div = samples.shape[0] // 4
    sub_sample_inds = np.linspace(0, samples.shape[0]-1, target_num, dtype=np.int)
    fix_anchors_inds = fix_inds[sub_sample_inds]
    sub_sample = samples[sub_sample_inds]
    print(sub_sample.shape)
    fixed_marks = worklist_convert.create_spheres(sub_sample)

    t = vtk.vtkTransform()
    t.Scale(20, 20, 20)
    axes = vtk.vtkAxesActor()
    axes.SetUserTransform(t)

    # with open("")
    # source_anchor = np.concatenate([sub_sample, ])
    anchors = np.concatenate([sub_sample, target], axis=0)
    # worklist_convert.apply_color(fixed_marks)

    anchors_inds = np.concatenate([fix_anchors_inds, [i]])


    marks_tt = root_points[anchors_inds]
    marks_tt[-1] = target[-1]
    sample_markers = worklist_convert.create_spheres(marks_tt)

    # worklist_convert.render_show([actor, axes, *sample_markers])
    return root_polydata,  marks_tt, anchors_inds

def merge_data(partial_polydata):
    reader = vtk.vtkSTLReader()
    reader.SetFileName("resources/007_Lower_Second_Molar.stl")
    reader.Update()
    pd = reader.GetOutput()

    with open("result/landmark_sample.pkl", "rb") as f:
        data = pickle.load(f)

    tau = data["number"]
    source = data["source"]
    target = data["target"]
    root_inds = data["root_inds"]

    target_marks = worklist_convert.create_spheres(target)
    source_marks = worklist_convert.create_spheres(source)
    worklist_convert.apply_color(source_marks)
    worklist_convert.apply_color(target_marks)
    tooth_actor = worklist_convert.polydata_to_acctor(pd)
    tooth_actor.GetProperty().SetOpacity(0.5)

    points = numpy_support.vtk_to_numpy(pd.GetPoints().GetData())
    polys = numpy_support.vtk_to_numpy(pd.GetPolys().GetData()).reshape([-1, 4])

    partial_points = numpy_support.vtk_to_numpy(partial_polydata.GetPoints().GetData())
    points[root_inds] = partial_points

    point_array = numpy_support.numpy_to_vtk(points)

    pd.GetPoints().SetData(point_array)
    return pd


    # print(points.shape, polys.shape)
    #
    # kept_inds = np.zeros([points.shape[0]], dtype=np.bool)
    # # N
    # kept_inds[root_inds] = True
    # # M X 3
    # cells = polys[:, 1:]
    # # M X 3
    # root_poly_inds = np.all(kept_inds[cells], axis=1)
    #
    # root_points = points[root_inds]
    # root_cells = cells[root_poly_inds]



if __name__=="__main__":
    polydata, source, inds = get_test_data()
    print("-->", source.shape, inds.shape)
    merge_polydata = merge_data(polydata)
    worklist_convert.render_show_pd([merge_polydata])
