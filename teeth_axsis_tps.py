import numpy as np

import worklist_convert
from configure import *
import vtk
from vtk.util import numpy_support
from vtk_utils import myTransform
import tps
import pickle
import os
from localData import AnatomyReader
import vtk_utils


class axesActor(vtk.vtkActor):

    def __init__(self, parent=None):
        super(axesActor, self).__init__()

def apply_color(act, color):
    for a in act:
        a.GetProperty().SetColor(*color)


class TeethAxisTps(object):
    def __init__(self):
        self.reader = AnatomyReader()
        self.anatomay_axis = {}
        tau = 17
        polydata = self.reader.get_polydata(tau)

        polydata_points = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())

        lines = self.update_axis(tau)

        actor = worklist_convert.polydata_to_acctor(polydata)
        samples, inds_list, crown_samples = self.sampling_roots(tau)
        marks = []
        N = 10
        num_axis = 3
        mode_root = 2
        crown_samples = np.concatenate(crown_samples, axis=0)
        print("crown sample", crown_samples.shape)
        N = 25
        div = crown_samples.shape[0] // N

        num = len(samples)
        test_points = [ ]

        keep_pts = [crown_samples[::div]]
        crown_marks = worklist_convert.create_spheres(crown_samples[::div])
        apply_color(crown_marks, (0.5, 0.1, 0.2))
        for i, (sample, inds) in enumerate(zip(samples, inds_list)):
            sub_sample = sample
            if sample.shape[0] > N:
                div = sample.shape[0] // N
                sub_sample = sample[::div]
                sub_inds = inds[::div]

            s0 = sub_sample[sub_inds==0]
            s1 = sub_sample[sub_inds == 1]
            s2 = sub_sample[sub_inds == 2]
            m0 = worklist_convert.create_spheres(s0)
            m1 = worklist_convert.create_spheres(s1)
            m2 = worklist_convert.create_spheres(s2)

            keep_pts.append(s0)
            keep_pts.append(s1)

            test_points.append(s2)

            if i == 0:
                x0 = s2.mean(axis=0)
            if i == num-1:
                xn = s2.mean(axis=0)


            apply_color(m0, (1, 0, 0))
            apply_color(m1, (0, 1, 0))
            apply_color(m2, (0, 0, 1))
            marks += [*m0, *m1, *m2]

        test_points = np.concatenate(test_points, axis=0)



        t = myTransform()
        t.RotateWXYZ(-10, 1, 5, 0.5)
        t.Translate(*tuple(-x0))
        mat4x4 = t.convert_np_mat()
        rot_line = vtk_utils.apply_transform_np(test_points, mat4x4)
        keep_pts = np.concatenate(keep_pts, axis=0)
        src = np.concatenate([keep_pts, test_points], axis=0)
        tar = np.concatenate([keep_pts, rot_line], axis=0)
        tps_results = tps.apply_tps(src, tar, polydata_points)

        vtk_array = numpy_support.numpy_to_vtk(tps_results, vtk.VTK_FLOAT)
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(vtk_array)
        tps_pd = vtk.vtkPolyData()
        tps_pd.SetPoints(vtk_points)
        tps_pd.SetPolys(polydata.GetPolys())
        tps_act = worklist_convert.polydata_to_acctor(tps_pd)


        rot_marks = worklist_convert.create_spheres(rot_line)
        apply_color(rot_marks, (1, 1, 0))
        # worklist_convert.apply_color(marks)
        worklist_convert.render_show([ tps_act,  *lines, *marks, *rot_marks, *crown_marks])


    def load_axis_info(self):
        if not os.path.exists(ANATOMY_AXIS_DEFINITION_FILEPATH):
            raise FileNotFoundError(ANATOMY_AXIS_DEFINITION_FILEPATH)

        with open(ANATOMY_AXIS_DEFINITION_FILEPATH, "rb") as f:
            self.anatomay_axis = pickle.load(f)

        # copy 21 ~ 27 from 11 ~ 17, and 31~37 from 41 ~ 47
        left_upper = [i + 20 for i in range(1, 8)]
        left_lower = [i + 30 for i in range(1, 8)]
        for tau in left_upper:
            self.anatomay_axis[tau] = self.anatomay_axis[tau - 10].copy()

        for tau in left_lower:
            self.anatomay_axis[tau] = self.anatomay_axis[tau + 10].copy()

    def remove_axes_info(self):
        pass
        # for act in self.ren.GetActors():
        #     if isinstance(act, axesActor):
        #         self.ren.RemoveActor(act)

    def compute_distance(self, a, n, p):
        """
        https: // en.wikipedia.org / wiki / Distance_from_a_point_to_a_line
        vector formulation
        x = a + t*n* n & point p
        (a-p) - ((a-p)*n)*n
        :param a: N X 3
        :param n: N X 3
        :param p: M X 3
        :return: M
        """
        p_ex = p.reshape([-1, 1, 3])
        # M X N
        t = np.sum(a - p_ex * n, axis=2)
        t = np.expand_dims(t, axis=2)

        # M X N X 3
        out = (a - p_ex) - t * n
        dist = np.linalg.norm(out, axis=2)
        inds = np.argmin(dist, axis=1)
        return inds



    def sampling_roots(self, tau):
        polydata = self.reader.get_polydata(tau)
        ids = np.array(self.anatomay_axis[tau]['ids'])

        points = np.array([polydata.GetPoint(i) for i in ids])

        points_line = points.reshape([-1, 2, 3])

        mean_lines = points_line.mean(axis=0)



        points = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())

        Nz = 10
        t = np.linspace(0, 1, Nz).reshape([-1, 1])
        pt1 = mean_lines[0]
        pt2 = mean_lines[1]
        sample_z_src = pt1 * (1 - t) + pt2 * t
        #
        # lines = points_line[2]
        # pt1 = lines[0]
        # pt2 = lines[1]
        #
        # N X 3
        pts_axis = points_line[:, 1, :] - points_line[:, 0, :]
        pts_axis = pts_axis / np.linalg.norm(pts_axis, axis=1).reshape([-1, 1])
        #
        # sample_z_src =  np.expand_dims(points_line[:, 0, :], axis=0) * (1-t) + np.expand_dims(points_line[:, 1, :], axis=0) * t
        # # N X 3

        direction = pt2 - pt1
        direction = direction / np.linalg.norm(direction)
        sample_z = np.dot(sample_z_src - pt1, direction)

        z = np.dot(points - pt1, direction)



        thersh = 0.1
        cond = []
        samples = []
        thresh_z = 12
        sort_inds = []

        other_sample = []
        for dz in sample_z:
            where = np.logical_and(z >= dz - thersh, z < dz + thersh)

            if dz < thresh_z:
                other_sample.append(points[where])
                pass
            else:
                # where = np.logical_and(z >= dz-thersh, z < dz + thersh)

                sample_points = points[where]

                inds = self.compute_distance(points_line[:, 0, :], pts_axis, sample_points)
                sort_inds.append(inds)

                cond.append(where)
                samples.append(sample_points)

        return samples, sort_inds, other_sample





    def update_axis(self, tau):
        # tau = self._tooth_number
        # self._tooth_number
        if len(self.anatomay_axis) == 0:
            self.load_axis_info()

        self.remove_axes_info()
        # self.remove_axes()
        # polydata = self.tooth_actors[tau].GetMapper().GetInput()
        polydata = self.reader.get_polydata(tau)

        def extend_line(pts):
            t = -1
            p1 = pts[0] * (1 - t) + pts[1] * t
            t = +2
            p2 = pts[0] * (1 - t) + pts[1] * t
            return np.stack([p1, p2], axis=0)

        if polydata is not None:
            ids = np.array(self.anatomay_axis[tau]['ids'])

            points = np.array([polydata.GetPoint(i) for i in ids])

            points_line = points.reshape([-1, 2, 3])

            lines = []
            for pts in points_line:

                line = vtk_utils.create_line_source(extend_line(pts))
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(line)

                axes = axesActor()
                axes.SetMapper(mapper)
                lines.append(axes)
                # self.ren.AddActor(axes)


            if points_line.shape[0] > 1:
                worklist_convert.apply_color(lines)
            else:
                lines[0].GetProperty().SetColor(1, 0, 0)
            return lines
            # self.iren.Render

if __name__=="__main__":
    t = TeethAxisTps()
