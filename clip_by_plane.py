import numpy as np
import sys
import vtk
from PyQt5 import Qt
import worklist_convert
import matplotlib.pyplot as plt
import utilsfunc
import vtk_utils
import cv2
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
from vtk.util import numpy_support
from vtk_utils import timefn2
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


def draw_triangle(pts):
    """
    :param pts: 3 X 3
    :return:
    """
    ax = plt.gca()
    v0 = pts[0]
    v1 = pts[1]
    v2 = pts[2]

    ax.plot([v0[0], v1[0], v2[0], v0[0]], \
            [v0[1], v1[1], v2[1], v0[1]], \
            [v0[2], v1[2], v2[2], v0[2]])

def get_sample_data():
    path = "resources/002_Upper_Lateral__Incisors.stl"
    reader = vtk.vtkSTLReader()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()


def define_normal_origin():
    test_norm = np.array([3, 1, 10])
    origin = np.array([0.5, 0.1, 0])
    test_norm = test_norm / np.linalg.norm(test_norm)
    up = np.array([0, 1, 0])
    up = up / np.linalg.norm(up)
    v = up - np.dot(test_norm, up) * test_norm
    v = v / np.linalg.norm(v)
    u = np.cross(v, test_norm)

    # temp_u = np.array([1, 1, 0])
    # temp_u = temp_u / np.linalg.norm(temp_u)
    # u = temp_u - np.dot(test_norm, temp_u) * test_norm
    # u = u / np.linalg.norm(u)
    # v = np.cross(test_norm, u)

    transform = np.stack([u, v, test_norm], axis=0)
    print(u, v, test_norm)

    return transform, origin
class ContourExtractor(object):
    """
    http://geomalgorithms.com/a06-_intersect-2.html
    """
    def __init__(self):
        # input data
        self.points = None
        self.polys = None
        self.normal = None
        self.origin = None

        # results
        self.crossing_plane_cells = None
        self.crossed_cell = None

    # @timefn2
    def setPoints(self, points, polys, normal, up, origin):
        self.points = points
        self.polys = polys
        self.normal = normal
        self.origin = origin

        up = up / np.linalg.norm(up)
        v = up - np.dot(normal, up) * normal
        v = v / np.linalg.norm(v)
        self.v = v
        self.u = np.cross(v, self.normal)
        self.transform = np.stack([self.u, self.v, self.normal], axis=0)

        proj_points = np.dot(self.points - self.origin, self.normal)
        poly_shape = self.polys.shape

        proj_points_restored = proj_points[self.polys.ravel()].reshape(poly_shape)

        signs = np.sign(proj_points_restored)
        sum_signs = np.abs(np.sum(signs, axis=1))
        inds, = np.where(sum_signs != 3)
        self.crossing_plane_cells = inds

        vertices_inds_totals = self.polys[inds]
        # vertices_inds = self.polys[inds][:, 0]

        vertices_samples = self.points[vertices_inds_totals.ravel()].reshape([*vertices_inds_totals.shape, 3])
        self.crossing_points = vertices_samples

        if False:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            v2 = vertices_samples[:, 0, :]
            p = self.points
            ax.plot(p.T[0][::50], p.T[1][::50], p.T[2][::50], '.')
            ax.plot(v2.T[0], v2.T[1], v2.T[2], 'r*')
    # def _get_crossing_inds(self, pts_range, num_pts, direction, uvn):
    def _get_crossing_inds(self, pts, direction, uvn):
        vertices_samples = self.crossing_points

        vv0 = vertices_samples[:, 0]
        vv1 = vertices_samples[:, 1]
        vv2 = vertices_samples[:, 2]

        u, v, n = uvn[0], uvn[1], uvn[2]
        # uv_range[0,0]
        # t = np.linspace(0, 1, num_pts)
        # P0 = pts_range[0] + t.reshape([-1, 1]) * (pts_range[1] - pts_range[0])
        P0 = pts
        P1 = P0 + direction
        P01 = direction.reshape([-1, 3])

        # vv0, N X 3, P0 : M X 3
        vv0_ex = np.expand_dims(vv0, axis=0)
        P0_ex = np.expand_dims(P0, axis=1)
        poly_n_ex = np.expand_dims(n, 0)

        numerator = np.sum((vv0_ex - P0_ex) * poly_n_ex, axis=2)
        denominator = np.sum(n*direction, axis=1)
            # np.expand_dims(np.dot(n, u), axis=0)
        r = numerator / denominator

        # np.expand_dims(P0, axis=1) + np.expand_dims(r, axis=2) * P01.reshape([-1, 1, 3])
        pts_on_plane = P0.reshape([-1, 1, 3]) + np.expand_dims(r, axis=2) * P01.reshape([-1, 1, 3])

        poly_w = pts_on_plane - np.expand_dims(vv0, axis=0)

        uv = np.sum(u * v, axis=1)
        vv = np.sum(v * v, axis=1)
        uu = np.sum(u * u, axis=1)

        wv = np.sum(poly_w * np.expand_dims(v, axis=0), axis=2)
        wu = np.sum(poly_w * np.expand_dims(u, axis=0), axis=2)

        uu_ex = np.expand_dims(uu, axis=0)
        vv_ex = np.expand_dims(vv, axis=0)
        uv_ex = np.expand_dims(uv, axis=0)

        den = uv * uv - uu * vv
        si = (uv_ex * wv - vv_ex * wu) / np.expand_dims(den, axis=0)
        ti = (uv_ex * wu - uu_ex * wv) / np.expand_dims(den, axis=0)

        sumst = si + ti
        where_si_posit = si >= 0
        where_ti_posit = ti >= 0
        where_st_posit = sumst <= 1

        where_final = np.logical_and(np.logical_and(where_si_posit, where_ti_posit), where_st_posit)

        return where_final, P0, pts_on_plane

    def get_surface(self, x, y, z):
        pass


    # @timefn2
    def update_test(self, sample_u_range, sample_v_range, origin, num=100):
        # assert np.dot(u, v) <= 1e-10

        vertices_samples = self.crossing_points
        vv0 = vertices_samples[:, 0]
        vv1 = vertices_samples[:, 1]
        vv2 = vertices_samples[:, 2]

        poly_u = vv1 - vv0
        poly_v = vv2 - vv0
        poly_n = np.cross(poly_u, poly_v)
        uvn = np.stack([poly_u, poly_v, poly_n], axis=0)

        u = self.u
        v = self.v

        # 2 X 2 X 3
        sample_u_range = np.array(sample_u_range)
        sample_v_range = np.array(sample_v_range)
        origin = np.array(origin)

        uv_range = origin.reshape([1, 1, 3]) + \
              sample_u_range.reshape([2, 1, 1]) * u.reshape([1, 1, 3]) + \
              sample_v_range.reshape([1, 2, 1]) * v.reshape([1, 1, 3])

        u_range = uv_range[:, 0]
        v_range = uv_range[0, :]

        tu, tv = np.meshgrid(np.linspace(sample_u_range[0], sample_u_range[1], num),
                             np.linspace(sample_v_range[0], sample_v_range[1], num))

        plane_pts = origin.reshape([1, 1, 3]) + \
        tu.reshape([num, num, 1]) * u.reshape([1, 1, 3]) + \
        tv.reshape([num, num, 1]) * v.reshape([1, 1, 3])

        # Num = 100
        # wheres_v, Pu0, pts_on_polygon = self._get_crossing_inds(u_range, num, v, uvn)
        wheres_v, Pu0, pts_on_polygon = self._get_crossing_inds(plane_pts[0, :, :], v, uvn)

        # wheres_u = self._get_crossing_inds(v_range, Num, u, uvn)

        t = np.linspace(0, 1, num).reshape([-1, 1]) * np.diff(sample_v_range)[0]
        # P0 = pts_range[0] + t.reshape([-1, 1]) * (pts_range[1] - pts_range[0])

        crossing = []
        nonzero_elem = v != 0
        for i, (pu0, where_cross, pts_on_poly) in enumerate(zip(Pu0, wheres_v, pts_on_polygon)):
            # Num X 3
            # pose = pu0 + t * v
            pose = plane_pts[:, i, :]
            # M

            # N X 3, Num X 3
            if where_cross.sum() > 0:
                sign_polyn = np.sign(poly_n[where_cross])
                pts_crossing = pts_on_poly[where_cross]

                direct01 = pts_crossing.reshape([1, -1, 3]) - pose.reshape([-1, 1, 3])
                direct02 = v

                r =  direct01[..., nonzero_elem] / direct02[nonzero_elem]
                crossing_counts = np.sum(np.sign(r[..., 0]) > 0, axis=1)
            else:
                crossing_counts = np.zeros([num])

            crossing.append(crossing_counts)

        crossing = np.array(crossing)

        return crossing




        # P0 = pts_range[0] + t.reshape([-1, 1]) * (v_range[1] - v_range[0])



    # @timefn2
    def Update(self, test_points):
        u = self.u

        vertices_samples = self.crossing_points
        vv0 = vertices_samples[:, 0]
        vv1 = vertices_samples[:, 1]
        vv2 = vertices_samples[:, 2]

        poly_u = vv1 - vv0
        poly_v = vv2 - vv0
        poly_n = np.cross(poly_u, poly_v)
        #
        P0 = test_points.reshape([-1, 3])
        P1 = P0 + u
        P01 = (P1 - P0).reshape([-1, 3])

        # vv0, N X 3, P0 : M X 3
        vv0_ex = np.expand_dims(vv0, axis=0)
        P0_ex = np.expand_dims(P0, axis=1)
        poly_n_ex = np.expand_dims(poly_n, 0)
        numer = np.sum( (vv0_ex - P0_ex) * poly_n_ex, axis=2)
        denerate = np.expand_dims(np.dot(poly_n, u), axis=0)
        r = numer / denerate
        # r = np.sum(vv0.reshape([-1, vv0.shape[0], 3]) - np.expand_dims(P0, axis=1) * poly_n.reshape([-1, poly_n.shape[0], 3]), axis=2)/\

        # * poly_n, axis=1) / np.dot(poly_n, u)
        # np.expand_dims(P0, axis=1) + np.expand_dims(r, axis=2) * P01.reshape([-1, 1, 3])
        pts_on_plane = P0.reshape([-1, 1, 3]) + np.expand_dims(r, axis=2) * P01.reshape([-1, 1, 3])

        poly_w = pts_on_plane - np.expand_dims(vv0, axis=0)

        where_crossing_positive = r >= 0

        uv = np.sum(poly_u * poly_v, axis=1)
        vv = np.sum(poly_v * poly_v, axis=1)
        uu = np.sum(poly_u * poly_u, axis=1)

        wv = np.sum(poly_w * np.expand_dims(poly_v, axis=0), axis=2)
        wu = np.sum(poly_w *  np.expand_dims(poly_u, axis=0), axis=2)

        uu_ex = np.expand_dims(uu, axis=0)
        vv_ex = np.expand_dims(vv, axis=0)
        uv_ex = np.expand_dims(uv, axis=0)

        den = uv * uv - uu * vv
        si = (uv_ex * wv - vv_ex * wu) / np.expand_dims(den, axis=0)
        ti = (uv_ex * wu - uu_ex * wv) / np.expand_dims(den, axis=0)

        sumst = si + ti
        # where_si_posit = si[where_crossing_positive] >= 0  # np.logical_and( si[where_crossing_temp] >=0, si[where_crossing_temp] <= 1)
        # where_ti_posit = ti[where_crossing_positive] >= 0  # np.logical_and( ti[where_crossing_temp] >=0, ti[where_crossing_temp] <= 1)
        # where_st_posit = sumst[where_crossing_positive] <= 1  # np.logical_and( sumst[where_crossing_temp] >=0, sumst[where_crossing_temp] <= 1)
        where_si_posit = si >= 0  # np.logical_and( si[where_crossing_temp] >=0, si[where_crossing_temp] <= 1)
        where_ti_posit = ti >= 0  # np.logical_and( ti[where_crossing_temp] >=0, ti[where_crossing_temp] <= 1)
        where_st_posit = sumst <= 1  # np.logical_and( sumst[where_crossing_temp] >=0, sumst[where_crossing_temp] <= 1)

        where_final = np.logical_and(np.logical_and(np.logical_and(where_si_posit, where_ti_posit), where_st_posit), where_crossing_positive)
        where_cond = where_si_posit * where_ti_posit

        print("----------->", where_final.sum())

        self.crossing = where_final
        self.crossed_cell = np.argwhere(where_final)
        self.crossed_cell_indices = self.crossing_plane_cells[self.crossed_cell[:, 1]]

        # final_crossing_cells = vertices_samples[self.crossed_cell]
        DEBUG = False
        if DEBUG:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            posit = vv0[where_crossing_positive[0]]

            ax.plot(vv0.T[0], vv0.T[1], vv0.T[2], 'x')
            ax.plot(P0.T[0], P0.T[1], P0.T[2], 'o')
            ax.plot(P1.T[0], P1.T[1], P1.T[2], '*')

            ax.plot(posit.T[0], posit.T[1], posit.T[2], 'ro')
            plt.show()


def main():
    polydata = get_sample_data()

    transform, origin = define_normal_origin()

    u, v, n = transform[0], transform[1], transform[2]

    points = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
    celldata = numpy_support.vtk_to_numpy(polydata.GetPolys().GetData())
    celldata_reshape = celldata.reshape([-1, 4])
    cells = celldata_reshape[:, 1:]

    extractor = ContourExtractor()
    extractor.setPoints(points, cells, n, up=np.array([0, 1, 0]), origin=origin)



    # U, V = np.meshgrid(np.linspace(-5, 5, 25), np.linspace(-5, 5, 25))
    U, V = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
    center = extractor.crossing_points.reshape([-1, 3]).mean(axis=0)

    p = np.dot(extractor.transform.T, center - extractor.origin)
    p[2] = 0
    center_proj = np.dot(extractor.transform.T, p) + extractor.origin


    test_points = np.expand_dims(U, axis=2) * extractor.u + np.expand_dims(V, axis=2) * extractor.v + center_proj

    extractor.update_test((-4, 4), (-3, 3), center_proj, 200)
    extractor.Update(test_points)
    proj_uv = extractor.crossing.sum(axis=1)
    proj_img = proj_uv.reshape(U.shape)
    drawing = np.zeros_like(proj_img, proj_img.dtype)
    drawing[proj_img==1] = 1
    plt.imshow(drawing)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # crossing_poly = extractor.polys[extractor.crossed_cell_points]
    # crossing_polygon = extractor.points[crossing_poly]
    crossing_points = extractor.crossing_points

    # for pt in crossing_polygon:
    #     draw_triangle(pt)
    crossing_points_reshape = crossing_points.reshape([-1, 3])
    ax.plot(crossing_points_reshape.T[0], crossing_points_reshape.T[1], crossing_points_reshape.T[2], "g*")

    grid_points = test_points.reshape([-1, 3])
    ax.plot(grid_points.T[0], grid_points.T[1], grid_points.T[2], 'ro')
    plt.show()

def main_test():

    polydata = get_sample_data()

    transform, origin = define_normal_origin()
    n = transform[2]
    u = transform[0]
    v = transform[1]

    points = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
    celldata = numpy_support.vtk_to_numpy(polydata.GetPolys().GetData())
    celldata_reshape = celldata.reshape([-1, 4])
    cells = celldata_reshape[:, 1:]

    ext = ContourExtractor()
    ext.setPoints(points, cells, n, np.array([0, 1, 0]), origin)
    center = ext.crossing_points.reshape([-1, 3]).mean(axis=0)
    p = np.dot(ext.transform.T, center - ext.origin)
    p[2] = 0
    center_proj = np.dot(ext.transform.T, p) + ext.origin

    origin = center_proj

    proj_points = np.dot(points - origin, n)
    poly_shape = cells.shape

    proj_points_restored = proj_points[cells.ravel()].reshape([*poly_shape])


    signs = np.sign(proj_points_restored)
    sum_signs = np.abs(np.sum(signs, axis=1))
    inds,  = np.where(sum_signs != 3)

    vertices_inds_totals = cells[inds]
    vertices_inds = cells[inds][:, 0]

    vertices_samples = points[vertices_inds_totals.ravel()].reshape([*vertices_inds_totals.shape, 3])

    vv0 = vertices_samples[:, 0]
    vv1 = vertices_samples[:, 1]
    vv2 = vertices_samples[:, 2]

    poly_u = vv1 - vv0
    poly_v = vv2 - vv0
    poly_n = np.cross(poly_u, poly_v)

    points_samples_by_clip_plane = points[vertices_inds]
    pt = points_samples_by_clip_plane

    P0 = origin
    P1 = u  + P0
    P01 = (P1 - P0).reshape([-1, 3])

    r_i = np.sum( (vertices_samples[:, 0, :] - P0)*poly_n, axis=1) / np.dot(poly_n, P1-P0)



    Pr_i = P0.reshape([-1, 3]) + r_i.reshape([-1, 1]) *P01
    print("positive", np.sum(r_i>0), "/", r_i.shape)
    nv = np.sum(poly_n * (Pr_i - vv0), axis=1)
    print(nv.max(), nv.min())

    where_crossing_temp = r_i >= 0
    poly_w = Pr_i - vv0

    # poly_w = P0 + r_i.reshape([-1, 1]) * P01

    uv = np.sum(poly_u* poly_v, axis=1)
    vv = np.sum(poly_v* poly_v, axis=1)
    uu = np.sum(poly_u* poly_u, axis=1)

    wv = np.sum(poly_w* poly_v, axis=1)
    wu = np.sum(poly_w* poly_u, axis=1)

    den = uv * uv - uu * vv
    si = (uv * wv - vv * wu) / den
    ti = (uv * wu - uu * wv) / den


    # test_all_true = np.ones(where_crossing_temp.shape, dtype=np.bool)
    sumst = si + ti
    where_si_posit = si[where_crossing_temp] >=0 #np.logical_and( si[where_crossing_temp] >=0, si[where_crossing_temp] <= 1)
    where_ti_posit =  ti[where_crossing_temp] >=0 #np.logical_and( ti[where_crossing_temp] >=0, ti[where_crossing_temp] <= 1)
    where_st_posit =  sumst[where_crossing_temp] <= 1 #np.logical_and( sumst[where_crossing_temp] >=0, sumst[where_crossing_temp] <= 1)

    where_final = np.logical_and(np.logical_and(where_si_posit, where_ti_posit), where_st_posit)
    print("---------", where_final.sum(), where_st_posit.sum(), where_st_posit.sum(), where_ti_posit.sum())
    posit_crossing = vertices_samples[where_crossing_temp]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    final_sample = vertices_samples[where_crossing_temp][where_final]
    ii,=np.where(where_crossing_temp)
    inds = ii[where_final]

    final_crossing = vertices_samples[where_crossing_temp][where_final]

    final_crossing_on_poly = final_crossing[:, 0] +\
                             si[inds].reshape([-1, 1]) * (final_crossing[:, 1]-final_crossing[:, 0])+ \
                             ti[inds].reshape([-1, 1]) * (final_crossing[:, 2]-final_crossing[:, 0])
    # ti[where_final]

    for pp in final_crossing_on_poly:
        ax.plot([pp[0]], [pp[1]], [pp[2]], 'b*')

    ratio = (final_crossing_on_poly - final_crossing[:, 0, :]) / (P1-P0)

    for pt in final_crossing:
        draw_triangle(pt)


    for pt in vertices_samples:
        draw_triangle(pt)

    posit = vertices_samples[where_crossing_temp][:, 0]
    ax.plot(posit.T[0], posit.T[1], posit.T[2], 'yo')
    posit_crossing_test = posit_crossing[:, 0, :]
    ax.plot(posit_crossing_test.T[0], posit_crossing_test.T[1], posit_crossing_test.T[2], "g*")
    # vertices_sample
    sample_cells = vertices_samples.reshape([-1, 3])
    # ax.plot(sample_cells.T[0], sample_cells.T[1], sample_cells.T[2], 'r.')

    # plt.plot(pt.T[0], pt.T[1], pt.T[2], '.')
    ax.plot([P0[0]], [P0[1]], [P0[2]], 'gx')
    ax.plot([P1[0]], [P1[1]], [P1[2]], 'go')

    p1_ = final_crossing_on_poly[0]
    ax.plot(
        [P0[0], p1_[0]],
    [P0[1], p1_[1]],
    [P0[2], p1_[2]]
    )
    plt.show()


def boxCallback(obj, event):
    # print(type(obj), type(event))
    print(obj.GetOrigin(), obj.GetPoint1(), obj.GetPoint2())


def plnae_widget_test():
    polydata = get_sample_data()
    teeth_actor = worklist_convert.polydata_to_acctor(polydata)
    ren = vtk.vtkRenderer()

    # ren.AddActor(teeth_actor)

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    planeWidget = vtk.vtkPlaneWidget()

    # planeWidget.SetPoint1(-8, -8, 0)
    # planeWidget.SetPoint2(8, 8, 0)
    # planeWidget.SetNormal(10, 3, 2)
    pts = np.array([[-8, -8, 0], [8, -8, 0], [0, 0, 0], [8, 8, 0]])
    sphere_marks  = worklist_convert.create_spheres(pts)
    planeWidget.SetPoint1(*tuple(pts[0]))
    planeWidget.SetPoint2(*tuple(pts[1]))
    ren.AddActor(teeth_actor)

    planeWidget.UpdatePlacement()

    planeWidget.SetInteractor(iren)


    for a in sphere_marks:
        ren.AddActor(a)




    planeWidget.AddObserver("InteractionEvent", boxCallback)

    planeWidget.EnabledOn()



    iren.Initialize()

    ren.ResetCamera()
    renWin.Render()
    iren.Start()

class MainWindow(Qt.QMainWindow):

    def anaysis_pca(self):
        self.transform_pts = self.pca.transform(self.test_points)
        self.scalars_theta = np.arctan2(self.transform_pts[:, 0], self.transform_pts[:, 2])
        self.scalars_phi = np.arctan2(self.transform_pts[:, 1], self.transform_pts[:, 2])

        self.theta_scalars = numpy_support.numpy_to_vtk(self.scalars_theta, array_type=vtk.VTK_FLOAT)
        self.phi_scalars = numpy_support.numpy_to_vtk(self.scalars_phi, array_type=vtk.VTK_FLOAT)


        pass
    def __init__(self, parent=None):
        super(MainWindow, self).__init__()
        # self.planeWidget = vtk.vtkPlaneWidget()
        self.planeWidget = vtk.vtkBoxWidget()
        self.test_polydata = get_sample_data()

        self.test_points = numpy_support.vtk_to_numpy(self.test_polydata.GetPoints().GetData())

        self.pca = PCA()
        self.pca.fit(self.test_points)

        self.anaysis_pca()

        self.mode_theta = True



        celldata = numpy_support.vtk_to_numpy(self.test_polydata.GetPolys().GetData())
        celldata_reshape = celldata.reshape([-1, 4])
        cells = celldata_reshape[:, 1:]
        self.test_cells = cells

        planesource = vtk.vtkPlaneSource()
        planesource.SetNormal(0, 0, 1)
        # planesource.SetOrigin(0, 0, 0)
        planesource.Update()

        plane_data = planesource.GetOutput()

        # self.plane_polydata = plane_data
        tranfrom = vtk_utils.myTransform()
        tranfrom.Scale(10, 10, 10)
        self.init_transform = tranfrom
        plane_data = vtk_utils.apply_transform(plane_data, tranfrom)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(plane_data)

        t = vtk_utils.myTransform()
        self.plane_actor = vtk.vtkActor()
        self.plane_actor.SetMapper(mapper)
        self.plane_actor.SetUserTransform(t)
        # self.plane_actor.GetProperty().SetColor(1, 1, 1)

        # self. vtk.vtkBoxWidget()


        self.test_actor = worklist_convert.polydata_to_acctor(self.test_polydata)
        self.vtkWidget = QVTKRenderWindowInteractor()

        self.ren = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()

        self.extractor = ContourExtractor()

        init_marks = np.zeros([4, 3])
        self.init_marks = worklist_convert.create_spheres(init_marks)
        worklist_convert.apply_color(self.init_marks)

        self.image_label = Qt.QLabel()

        self.image_width = 400
        self.image_height = 400
        init_image = np.zeros([self.image_height, self.image_width, 3], dtype=np.uint8)

        self.image_label2 = Qt.QLabel()

        self.image_label3 = Qt.QLabel()


        qimage = utilsfunc.numpy2qimage(init_image)
        pixmap = Qt.QPixmap(qimage)

        qimage2 = utilsfunc.numpy2qimage(init_image)
        qimage3 = utilsfunc.numpy2qimage(init_image)

        self.image_label.setPixmap(pixmap)
        self.image_label2.setPixmap(Qt.QPixmap(qimage2))
        self.image_label3.setPixmap(Qt.QPixmap(qimage3))

        for act in self.init_marks:
            self.ren.AddActor(act)

        self.ren.AddActor(self.plane_actor)

        self.ren.AddActor(self.test_actor)

        self.init_plane_widget()

        self.extractor_surf = ContourExtractor()

        self.update_texture()
        self.update_surface()

        self.layout = Qt.QGridLayout()
        self.layout.addWidget(self.vtkWidget, 0, 0, 1, 1)
        self.layout.addWidget(self.image_label, 0, 1, 1, 1)
        self.layout.addWidget(self.image_label2, 1, 0, 1, 1)
        self.layout.addWidget(self.image_label3, 1, 1, 1, 1)



        self.frame = Qt.QFrame()
        self.frame.setLayout(self.layout)

        self.setCentralWidget(self.frame)

        self.iren.AddObserver("KeyPressEvent", self.key_press_event)

        self.iren.Initialize()
        self.iren.Start()

        self.show()
    def key_press_event(self, obj, event):
        key = obj.GetKeySym()
        if key =="c":
            if self.mode_theta:
                scalars = self.phi_scalars
            else:
                scalars = self.theta_scalars

            self.mode_theta = not self.mode_theta

            self.test_polydata.GetPointData().SetScalars(scalars)
            self.test_actor.GetMapper().SetScalarRange(self.test_polydata.GetScalarRange())
        self.iren.Render()

    # @timefn2
    def update_surface(self):
        plane = self.plane_actor.GetMapper().GetInput()
        trasnfrom = self.plane_actor.GetUserTransform()
        transform_plane = vtk_utils.apply_transform(plane, trasnfrom)
        plane_pts = []
        for i in range(transform_plane.GetNumberOfPoints()):
            plane_pts.append(np.array(transform_plane.GetPoint(i)))

        if self.mode_theta:
            scalars = self.scalars_phi
        else:
            scalars = self.scalars_theta

        origin = plane_pts[0]
        pt1 = plane_pts[1]
        pt2 = plane_pts[2]

        normal = np.cross(pt1-origin, pt2 - origin)
        normal = normal / np.linalg.norm(normal)

        plane_center = (pt1 + pt2)/2

        proj_points = np.dot(self.test_points - plane_center, normal)
        where_positive, = np.where(proj_points > 0)

        roi_points = self.test_points[where_positive]

        # x_axis = np.cross(y, normal)

        z = normal
        y_axis = np.array([0, 1, 0])
        x_axis = np.cross(y_axis, z)
        x = x_axis / np.linalg.norm(x_axis)
        y = np.cross(z, x)

        sx = np.dot(roi_points, x)
        sy = np.dot(roi_points, y)
        sz = proj_points[where_positive]

        max_sz = sz.max()

        sx_min = sx.min()
        sx_max = sx.max()
        sy_min = sy.min()
        sy_max = sy.max()

        r = (max_sz) / (sz)

        sxyz = np.stack([sx, sy, sz], axis=1)
        ps = r.reshape([-1, 1]) * sxyz

        where_x = np.logical_and( ps[:, 0] >= sx_min, ps[:, 0] <= sx_max)
        where_y = np.logical_and( ps[:, 1] >= sy_min, ps[:, 1] <= sy_max)

        where_xy = np.logical_and(where_x, where_y)
        pts = ps[where_xy]
        #
        # make an agg figure
        fig, ax = plt.subplots()
        # ax.plot(phi_list, z_list, '*')
        # ax.plot([1, 2, 3])
        ax.plot(pts.T[0], pts.T[1], '.')
        ax.set_title('a simple figure')
        fig.canvas.draw()

        # grab the pixel buffer and dump it into a numpy array
        buf = fig.canvas.buffer_rgba()
        l, b, w, h = fig.bbox.bounds
        # The array needs to be copied, because the underlying buffer
        # may be reallocated when the window is resized.
        X = np.frombuffer(buf, np.uint8).copy()

        image = X.reshape([int(h), int(w), 4])
        qimage = utilsfunc.numpy2qimage(image)








        self.image_label3.setPixmap(Qt.QPixmap(qimage))
        #
        # phi = np.arctan2(y, x)
        # xy = np.stack([x, y], axis=1)
        # radius = np.linalg.norm(xy, axis=1)
        # dtheta = (np.pi * 2) / 100
        # dz = 0.2
        # dt_bins = np.linspace(-np.pi, np.pi, 100)
        # dz_bins = np.linspace(-z.min(), z.max(), 100)
        #
        # phi_inds = np.digitize(phi, dt_bins)
        # dz_inds = np.digitize(z, dz_bins)
        #
        # phi_list = []
        # z_list = []
        #
        # for i_p in range(phi_inds.min(), phi_inds.max()+1):
        #     theta_range = phi_inds == i_p
        #     # theta_arg, = np.where(theta_range)
        #     if theta_range.sum() > 0:
        #         for i_z in range(dz_inds.min(), dz_inds.max()+1):
        #             z_range = dz_inds == i_z
        #             # z_arg, = np.where(z_range)
        #             if theta_range.sum() > 0:
        #                 cond = np.logical_and(theta_range, z_range)
        #                 if cond.sum() > 0:
        #                     cond_arg, = np.where(cond)
        #                     max_radius_arg = np.argmax(radius[cond])
        #                     sample_i = cond_arg[max_radius_arg]
        #                     phi_list.append(phi[sample_i])
        #                     z_list.append(z[sample_i])
        #             else:
        #                 pass
        #
        # phi_list = np.array(phi_list)
        # z_list = np.array(z_list)
        #
        # # make an agg figure
        # fig, ax = plt.subplots()
        # ax.plot(phi_list, z_list, '*')
        # # ax.plot([1, 2, 3])
        # ax.set_title('a simple figure')
        # fig.canvas.draw()
        #
        # # grab the pixel buffer and dump it into a numpy array
        # buf = fig.canvas.buffer_rgba()
        # l, b, w, h = fig.bbox.bounds
        # # The array needs to be copied, because the underlying buffer
        # # may be reallocated when the window is resized.
        # X = np.frombuffer(buf, np.uint8).copy()
        #
        # image = X.reshape([int(h), int(w), 4])
        #
        # qimage = utilsfunc.numpy2qimage(image)
        #
        # self.image_label3.setPixmap(Qt.QPixmap(qimage))


    def update_texture(self):
        plane = self.plane_actor.GetMapper().GetInput()
        trasnfrom = self.plane_actor.GetUserTransform()
        transform_plane = vtk_utils.apply_transform(plane, trasnfrom)
        plane_pts = []
        for i in range(transform_plane.GetNumberOfPoints()):
            plane_pts.append(np.array(transform_plane.GetPoint(i)))

        if self.mode_theta:
            scalars = self.scalars_phi
        else:
            scalars = self.scalars_theta

        origin = plane_pts[0]
        pt1 = plane_pts[1]
        pt2 = plane_pts[2]

        normal = np.cross(pt1-origin, pt2 - origin)
        normal = normal / np.linalg.norm(normal)

        plane_center = (pt1 + pt2)/2

        proj_points = np.dot(self.test_points - plane_center, normal)
        where_positive, = np.where(proj_points > 0)

        roi_points = self.test_points[where_positive]

        # x_axis = np.cross(y, normal)

        z = normal
        xx = pt1 - origin
        yy = pt2 - origin
        x = xx / np.linalg.norm(xx)
        y = yy / np.linalg.norm(yy)

        sx = np.dot(self.test_points - plane_center, x)
        sy = np.dot(self.test_points - plane_center, y)
        sz = np.dot(self.test_points - plane_center, z)

        sxyz = np.stack([sx, sy, sz], axis=1)

        print(normal, plane_center)


        # points = self.test_points[self.test_cells]

        cells = proj_points[self.test_cells]
        posit = np.all(cells > 0, axis=1)

        # N X 3 X 3
        sample_pts = sxyz[self.test_cells[posit]]

        v0 = sample_pts[:, 0]
        v1 = sample_pts[:, 1]
        v2 = sample_pts[:, 2]

        poly_normal = np.cross(v1 - v0, v2 - v0)
        # N X 3
        rois  = np.dot(poly_normal, z) > 0

        samples = v0

        # make an agg figure
        fig, ax = plt.subplots()
        # ax.plot(phi, z, '*')
        # ax.plot([1, 2, 3])
        # ax.imshow(scalar_image, cmap='hsv')
        ax.plot(samples.T[0], samples.T[1], 'x')
        ax.set_title('a simple figure')
        ax.set_xlim(-7, 7)
        ax.set_ylim(-7, 7)
        fig.canvas.draw()

        # grab the pixel buffer and dump it into a numpy array
        buf = fig.canvas.buffer_rgba()
        l, b, w, h = fig.bbox.bounds
        # The array needs to be copied, because the underlying buffer
        # may be reallocated when the window is resized.
        X = np.frombuffer(buf, np.uint8).copy()

        image = X.reshape([int(h), int(w), 4])

        qimage = utilsfunc.numpy2qimage(image)

        self.image_label2.setPixmap(Qt.QPixmap(qimage))
        # self.image_label3.setPixmap(Qt.QPixmap(qimage2))






        # self.test_points.
    def update_texture2(self):
        plane = self.plane_actor.GetMapper().GetInput()
        trasnfrom = self.plane_actor.GetUserTransform()
        transform_plane = vtk_utils.apply_transform(plane, trasnfrom)
        plane_pts = []
        for i in range(transform_plane.GetNumberOfPoints()):
            plane_pts.append(np.array(transform_plane.GetPoint(i)))

        origin = plane_pts[0]
        pt1 = plane_pts[1]
        pt2 = plane_pts[2]

        normal = np.cross(pt1-origin, pt2 - origin)
        normal = normal / np.linalg.norm(normal)

        z_axis = normal
        y_axis = np.array([0, 1, 0])
        x_axis = np.cross(y_axis, z_axis)


        plane_center = (pt1 + pt2)/2

        transform = np.stack([x_axis, y_axis, z_axis], axis=1)
        # N X 3
        transform_xyz = np.dot(transform, (self.test_points - plane_center).T).T

        # N X 3 X 3
        # tri_poly_center_transform = np.sum(transform_xyz[self.test_cells], axis=1) / 3

        rois = transform_xyz[..., 2] > 0
        # rois = np.all(transform_xyz[..., 2][self.test_cells] > 0, axis=1)

        roi_vertices = transform_xyz[rois]

        roi_min = roi_vertices.min(axis=0)
        roi_max = roi_vertices.max(axis=0)

        roi_range = roi_max - roi_min

        mid_roi = (roi_min + roi_max) / 2
        max_range = np.max(roi_range[:2])
        max_range_2 = max_range / 2
        num = 100
        x = np.linspace(mid_roi[0] - max_range_2, mid_roi[0] +max_range_2, num)
        y = np.linspace(mid_roi[1] - max_range_2, mid_roi[1] +max_range_2, num)

        inds_x = np.digitize(roi_vertices[..., 0], x)
        inds_y = np.digitize(roi_vertices[..., 1], y)

        verts_x = [roi_vertices[inds_x == i] for i in range(num)]
        verts_y = [roi_vertices[inds_y == i] for i in range(num)]

        arg_x = [np.squeeze(np.argwhere(inds_x == i), axis=1) for i in range(num)]
        arg_y = [np.squeeze(np.argwhere(inds_y == i), axis=1) for i in range(num)]

        # sample_list = []
        # index_list = []
        sampling_pts = []
        sampling_index = []
        sampling = np.zeros([num, num], dtype=np.int)
        for ix in range(num):
            if arg_x[ix].size > 0:
                for iy in range(num):
                    if arg_y[iy].size > 0:
                        argx = arg_x[ix]
                        argy = arg_y[iy]
                        try:
                            arg = argx[np.any(argx == argy.reshape([-1, 1]), axis=0)]
                        except:
                            print(argx)
                        if arg.size > 0:
                            sample = roi_vertices[arg]
                            # z 축으로 가장 큰 값을 샘플링
                            i = np.argmax(sample[:, 2])
                            index = arg[i]
                            sampling_pts.append(sample[i])
                            sampling_index.append(index)
                            # index_list.append(index)
                            sampling[ix, iy] = 1

        sampling_index = np.array(sampling_index)
        sampling_scalars = self.scalars_theta[sampling_index]
        sampling_scalars_phi = self.scalars_phi[sampling_index]

        scalar_image = np.zeros([num, num], dtype=np.float32)
        scalar_image[sampling==1] = sampling_scalars

        scalar_image_phi = np.zeros([num, num], dtype=np.float32)
        scalar_image_phi[sampling==1] = sampling_scalars_phi

        # make an agg figure
        fig, ax = plt.subplots()
        # ax.plot(phi, z, '*')
        # ax.plot([1, 2, 3])
        ax.imshow(scalar_image, cmap='hsv')
        ax.set_title('a simple figure')
        fig.canvas.draw()

        # grab the pixel buffer and dump it into a numpy array
        buf = fig.canvas.buffer_rgba()
        l, b, w, h = fig.bbox.bounds
        # The array needs to be copied, because the underlying buffer
        # may be reallocated when the window is resized.
        X = np.frombuffer(buf, np.uint8).copy()

        image = X.reshape([int(h), int(w), 4])

        qimage = utilsfunc.numpy2qimage(image)


        # make an agg figure
        fig, ax = plt.subplots()
        # ax.plot(phi, z, '*')
        # ax.plot([1, 2, 3])
        ax.imshow(scalar_image_phi, cmap='hsv')
        ax.set_title('a simple figure')
        fig.canvas.draw()

        # grab the pixel buffer and dump it into a numpy array
        buf = fig.canvas.buffer_rgba()
        l, b, w, h = fig.bbox.bounds
        # The array needs to be copied, because the underlying buffer
        # may be reallocated when the window is resized.
        X = np.frombuffer(buf, np.uint8).copy()

        image = X.reshape([int(h), int(w), 4])

        qimage2 = utilsfunc.numpy2qimage(image)


        self.image_label2.setPixmap(Qt.QPixmap(qimage))





    def update_markers(self, pts):
        # pt1 = self.planeWidget.GetOrigin()
        # pt2 = self.planeWidget.GetPoint1()
        # pt3 = self.planeWidget.GetPoint2()
        # pts = [pt1, pt2, pt3]
        k = len(self.init_marks)

        for i, p in enumerate(pts):
            if i < k:
                # print(p)
                self.init_marks[i].SetPosition(*p)

    def init_plane_widget(self):

        planeWidget =  self.planeWidget
        planeWidget.SetInteractor(self.iren)

        planeWidget.SetProp3D(self.plane_actor)
        planeWidget.SetPlaceFactor(1.25) # make the box 1.25x larger than the actor

        planeWidget.PlaceWidget()
        planeWidget.EnabledOn()


        # pts = np.array([[-8, -8, 0], [8, -8, 0], [0, 0, 0], [8, 8, 0]])
        # sphere_marks = worklist_convert.create_spheres(pts)
        # planeWidget.SetPoint1(*tuple(pts[0]))
        # planeWidget.SetPoint2(*tuple(pts[1]))
        # planeWidget.UpdatePlacement()
        #
        # planeWidget.SetInteractor(self.iren)

        planeWidget.AddObserver("InteractionEvent", self.plane_callback)
        #
        # planeWidget.EnabledOn()

        # self.update_markers()

    def update_projection_image(self, plane_pts, proj_range, num):

        origin = plane_pts[0]
        pt1 = plane_pts[1]
        pt2 = plane_pts[2]

        normal = np.cross(pt1-origin, pt2 - origin)
        normal = normal / np.linalg.norm(normal)

        plane_center = (pt1 + pt2)/2

        self.extractor.setPoints(self.test_points, self.test_cells, normal, np.array([0, 1, 0]), origin)

        counts = self.extractor.update_test(proj_range, proj_range, plane_center, num)

        counts_image = (counts == 1).astype(np.uint8) * 255


        counts_image = cv2.resize(counts_image, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)
        qimage = utilsfunc.numpy2qimage(counts_image)
        self.image_label.setPixmap(Qt.QPixmap(qimage))

        return counts_image

    def plane_callback(self, obj, event):

        t = vtk.vtkTransform()
        obj.GetTransform(t)

        init_t = vtk_utils.myTransform()
        # init_t.DeepCopy(self.init_transform)
        concat = vtk_utils.myTransform()
        concat.Concatenate(t)
        # concat.Concatenate(init_t)
        obj.GetProp3D().SetUserTransform(concat)

        plane = self.plane_actor.GetMapper().GetInput()
        transform_plane = vtk_utils.apply_transform(plane, concat)
        plane_pts = []
        for i in range(transform_plane.GetNumberOfPoints()):
            plane_pts.append(np.array(transform_plane.GetPoint(i)))

        self.update_markers(plane_pts)
        num = 200
        plane_range = (-4, 4)
        plane_image = self.update_projection_image(plane_pts, plane_range, num)

        self.update_texture()
        self.update_surface()

        # self.update_projection_surface(plane_image, )
        # print( self.planeWidget.GetPoint1(), self.planeWidget.GetPoint1().
        # self.update_markers()

        # crossing = np.array(crossing)
        # plt.imshow(crossing)
        # plt.show()
        pass
def catch_exceptions(t, val, tb):
    # Qt.QMessageBox.critical(None,
    #                                "An exception was raised",
    #                                "Exception type: {}".format(t))
    raise RuntimeError("An exception was raised",
                                   "Exception type: {}".format(t))
    exit()
    old_hook(t, val, tb)

if __name__=="__main__":
    # main()
    # plnae_widget_test()
    # main_test()
    old_hook = sys.excepthook
    sys.excepthook = catch_exceptions
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
