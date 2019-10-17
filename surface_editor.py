# https://www.paraview.org/Wiki/VTK/Examples/Cxx/PolyData/SelectPolyData
from vtk.util import numpy_support
import vtk
from sklearn.neighbors import NearestNeighbors
import worklist_convert
import vtk_utils
import pickle
import numpy as np
from configure import FILE_NAME_TABLE



import sys
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5 import Qt
from dmm_main import TrainPolyData
import active_shape_model

class MainWindow(Qt.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.extract_data = {}
        for tooth_key in FILE_NAME_TABLE.keys():
            self.extract_data[tooth_key] = []


        #####
        self.vtkWidget = QVTKRenderWindowInteractor()
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.ren = vtk.vtkRenderer()
        self.iren.GetRenderWindow().AddRenderer(self.ren)

        self.vtkWidget2 = QVTKRenderWindowInteractor()
        self.iren2 = self.vtkWidget2.GetRenderWindow().GetInteractor()
        self.ren2 = vtk.vtkRenderer()
        self.iren2.GetRenderWindow().AddRenderer(self.ren2)
        #self.ren2.SetBackground(0, 0, 0)




        self.spin = Qt.QSpinBox()
        self.spin.setMinimum(11)
        self.spin.setMaximum(27)
        self.spin.setValue(11)

        self.spin.valueChanged.connect(self.tooth_changed)

        # self.iren.GetRenderWindow().AddRenderer(self.ren)

        # self.menuBar.
        self.mainMenu = self.menuBar()
        self.fileMenu = self.mainMenu.addMenu('File')
        self.openMenu = self.fileMenu.addMenu("Open")
        # self.fileMenu.triggered
        # fileMenu.addAction(extractAction)

        # sphereSource = vtk.vtkSphereSource()
        # sphereSource.SetThetaResolution(40)
        # sphereSource.SetPhiResolution(20)
        # sphereSource.Update()
        number = self.spin.value()
        path =  FILE_NAME_TABLE[number]


        reader = vtk.vtkSTLReader()
        reader.SetFileName(path)
        reader.Update()

        self.polyData = reader.GetOutput()

        self.triangleFilter = vtk.vtkTriangleFilter()
        self.triangleFilter.SetInputData(self.polyData)
        self.triangleFilter.Update()

        # vtk.VTK_ID_TYPE

        self.pd = self.triangleFilter.GetOutput()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self.triangleFilter.GetOutputPort())
        mapper.ScalarVisibilityOn()


        self.teeth_actor = vtk.vtkActor()
        self.teeth_actor.SetMapper(mapper)
        self.teeth_actor.GetProperty().SetInterpolationToFlat()

        mapper = vtk.vtkPolyDataMapper()
        self.partial_actor = vtk.vtkActor()
        self.partial_actor.SetMapper(mapper)
        # self.ren = vtk.vtkRenderer()
        # renWin = vtk.vtkRenderWindow()
        # renWin.AddRenderer(self.ren)
        # self.iren = vtk.vtkRenderWindowInteractor()
        # iren = self.iren
        # iren.SetRenderWindow(renWin)

        self.ren.AddActor(self.teeth_actor)

        self.contourWidget = vtk.vtkContourWidget()
        self.contourWidget2 = vtk.vtkContourWidget()

        self.activae_contour_widget()
        self.active_contour_widget2()
        # imagefilter.SetM



        # self.algorithm = active_shape_model.TestASM()
        #
        # self.model_teeth = self.algorithm.model_actos
        # self.scanned_teeth = self.algorithm.scannned_teeth_list
        # self.etc_acts = self.algorithm.etc_acts
        #
        # for act in self.scanned_teeth + self.etc_acts + self.model_teeth:
        #     self.ren.AddActor(act)

        self.layout = Qt.QVBoxLayout()

        ##################  set up lay out #################
        self.create_button()
        self.connect_button()

        # lookuptable = mapper.GetLookupTable()
        self.scalarBar = vtk.vtkScalarBarActor()
        # self.scalarBar.SetLookupTable(lut)
        self.scalarBar.SetTitle("Distance")
        self.scalarBar.SetNumberOfLabels(5)
        self.ren2.AddActor(self.scalarBar)


        self.layout.addWidget(self.spin)

        self.layout.addWidget(self.vtkWidget)
        self.layout.addWidget(self.vtkWidget2)

        frame = Qt.QFrame()
        frame.setLayout(self.layout)

        self.setCentralWidget(frame)

        self.iren.AddObserver("KeyPressEvent", self.keypress_event)

        # renWin.Render()
        self.iren.Initialize()

        self.iren.Start()
        self.contourWidget.EnabledOn()

        self.iren2.Initialize()
        self.iren2.Start()

        self.iren2.Render()
        # self.iren2.Start()

        self.show()

    def add_in_render2(self, item):
        if isinstance(item, vtk.vtkPolyData):

            if False:
                distance = numpy_support.vtk_to_numpy(item.GetPointData().GetScalars())
                points = numpy_support.vtk_to_numpy(item.GetPoints().GetData())
                samples = points[distance < .3]
                point_polydata= vtk_utils.create_points_test(samples)
                worklist_convert.render_show_pd([point_polydata])

            # mapper = vtk.vtkPolyDataMapper()
            # mapper.SetInputData(item)
            #
            # actor = vtk.vtkActor()
            # actor.SetMapper(mapper)
            self.partial_actor.GetMapper().SetInputData(item)
            self.partial_actor.GetMapper().SetScalarRange(item.GetScalarRange())

            self.scalarBar.SetLookupTable(self.partial_actor.GetMapper().GetLookupTable())

            self.ren2.AddActor(self.partial_actor)
        elif isinstance(item, vtk.vtkActor):
            self.ren2.AddActor(item)



    def remove_in_render2(self):
        if self.partial_actor in self.ren2.GetViewProps():
            self.ren2.RemoveActor(self.partial_actor)
        # for act in self.ren2.GetViewProps():
        #     self.ren2.RemoveActor(act)


    def pd2_actor(self, pd):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(pd)

        actr = vtk.vtkActor()
        actr.SetMapper(mapper)
        return actr

    def patch_color_tooth(self, polydata:vtk.vtkPolyData, ids):
        N = polydata.GetNumberOfPoints()
        color_np = np.ones([N, 3], np.uint8) * 255
        color_np[ids, :] = np.array([0, 255, 0])
        color_vtk = numpy_support.numpy_to_vtk(color_np, array_type=vtk.VTK_UNSIGNED_CHAR)

        polydata.GetPointData().SetScalars(color_vtk)

    def tooth_changed(self):

        number = self.spin.value()
        self.remove_in_render2()
        if number in FILE_NAME_TABLE:
            reader = vtk.vtkSTLReader()
            reader.SetFileName(FILE_NAME_TABLE[number])
            reader.Update()

            self.polyData.DeepCopy(reader.GetOutput())

            # triangleFilter = vtk.vtkTriangleFilter()
            self.triangleFilter.SetInputData(self.polyData)
            self.triangleFilter.Update()

            self.pd.DeepCopy(self.triangleFilter.GetOutput())
            self.clear_items()

            if number in self.extract_data:
                if self.extract_data[number]:
                    id_list = []
                    for item in self.extract_data[number]:
                        polydata = item["polydata"]
                        self.add_in_render2(polydata)

                        id_list.append(item['ids'])
                        # actor = self.pd2_actor(polydata)
                        # self.ren2.AddActor(actor)

                    id_list = np.concatenate(id_list, axis=0)
                    print("#{}extract area :{}".format(number, id_list.size))
                    self.patch_color_tooth(self.polyData, id_list)
                else:
                    pass
                    # emtpy polydata
                    # self.actor2.GetMapper().SetInputData(vtk.vtkPolyData())

            self.reset_camera2()

        self.Render()


    def Render(self):
        self.iren.Render()
        self.iren2.Render()



    def clear_items(self):
        for it in self.ren.GetViewProps():
            if it == self.teeth_actor:
                pass
            else:
                self.ren.RemoveActor(it)
        # contourWidget.GetContourRepresentation().
        # contourWidget.GetRepresentation().ClearAllNodes()
        self.activae_contour_widget()
        self.contourWidget.EnabledOn()

        self.iren.Render()

    def _write_extracted_data(self, filename):
        for key, values in self.extract_data.items():

            if not values:
                for item in values:
                    if "polydata" in item:
                        values.pop("polydata")

                # "polydata":pp,
                # "points":extract_pts
                # "ids":inds

                pass
            else:
                pass

        with open(filename, "wb") as f:
            pickle.dump(self.extract_data, f)

    def _load_file(self, filename):
        with open(filename, "rb") as f:
            items = pickle.load(f)

        if isinstance(items, dict):
            for key, listitem in items.items():
                if not listitem:
                    for values in listitem:
                        pd = vtk.vtkPolyData()
                        # vtk.Poin
                        cell = numpy_support.numpy_to_vtk(values["cells"], array_type=vtk.VTK_ID_TYPE)
                        pts = numpy_support.numpy_to_vtk(values["points"], array_type=vtk.VTK_FLOAT)
                        inds = values["ids"]
                        # self.polyData
                        # "ids": corresponding_inds,

                        N = self.polyData.GetNumberOfPoints()

                        # SetScalars

                        vtkpoints = vtk.vtkPoints()
                        vtkpoints.SetData(pts)

                        vtkcell = vtk.vtkCellArray()
                        # vtkcell = pd.GetPolys().GetData()
                        # assert
                        vtkcell.SetCells(cell.GetNumberOfTuples()//4, cell)
                        # vtkcell.DeepCopy(cell)
                        # vtkcell.SetCells(cell)
                        pd.SetPolys(vtkcell)
                        pd.SetPoints(vtkpoints)

                        values.update({"polydata":pd})

                        self.extract_data[key] = values
                        # vtk.VTK_ID_TYPE_IMPL

        else:
            pass

    def apply_load_button(self):
        dlg = Qt.QFileDialog()

        filename, _ = dlg.getOpenFileName(self, "Open Surface Data", "", \
                                          "Surface data(*.pkl)")

        if filename:
            self._load_file(filename)
            self.tooth_changed()

        else:
            pass


    def apply_save_button(self):
        dlg = Qt.QFileDialog()

        filename, _ = dlg.getSaveFileName(self, "Save Surface Teeth", "", "User Define(*.pkl)")

        if filename:
            print(filename, "--", filename[-1])
            self._write_extracted_data(filename)

        else:
            pass

    def create_button(self):

        toolbar = Qt.QToolBar()

        self.apply_action = Qt.QAction(Qt.QIcon("../vtk_qt/resources/smoothing.png"), "load", self)

        self.save_button = Qt.QAction(Qt.QIcon("./resources/saveicon.png"), "save", self)
        self.load_button = Qt.QAction(Qt.QIcon("./resources/open-file-icon.png"), "loadr", self)
        self.clear_extraction_button = Qt.QAction(Qt.QIcon("Clear"), "clear", self)

        toolbar.addAction(self.apply_action)
        toolbar.addAction(self.save_button)
        toolbar.addAction(self.load_button)
        toolbar.addAction(self.clear_extraction_button)

        self.opacity_scrollbar = Qt.QScrollBar(Qt.Qt.Horizontal)
        self.opacity_scrollbar.setMinimum(0)
        self.opacity_scrollbar.setMaximum(100)
        # self.opacity_scrollbar.setMaximumWidth(100)
        self.opacity_scrollbar.setValue(100)

        self.layout.addWidget(toolbar)
        self.layout.addWidget(self.opacity_scrollbar)
        pass

    def clear_extract_surface(self):
        number = self.spin.value()
        if number in self.extract_data:
            self.extract_data[number].clear()


    def connect_button(self):
        self.apply_action.triggered.connect(self.apply_asm_iteratation)
        self.save_button.triggered.connect(self.apply_save_button)
        self.load_button.triggered.connect(self.apply_load_button)
        self.clear_extraction_button.triggered.connect(self.clear_extract_surface)

        self.opacity_scrollbar.valueChanged.connect(self.control_opacity)


    def control_opacity(self):
        value = self.opacity_scrollbar.value() / self.opacity_scrollbar.maximum()

        self.scanned_teeth[0].GetProperty().SetOpacity(value)

        self.iren.Render()

    def apply_asm_iteratation(self):
        print("apply_asm_iteratation")
        self.algorithm.find_optimal_model()
        self.iren.Render()

    def activae_contour_widget(self):
        # global contourWidget


        self.contourWidget = vtk.vtkContourWidget()
        self.contourWidget.SetInteractor(self.iren)
        # contourWidget.AddObserver()
        rep = vtk.vtkOrientedGlyphContourRepresentation.SafeDownCast(self.contourWidget.GetRepresentation())
        rep.GetLinesProperty().SetColor(1, 0.2, 0)
        rep.GetLinesProperty().SetLineWidth(3.0)

        pointPlacer = vtk.vtkPolygonalSurfacePointPlacer()
        pointPlacer.AddProp(self.teeth_actor)
        pointPlacer.GetPolys().AddItem(self.pd)
        rep.SetPointPlacer(pointPlacer)

        interpolator = vtk.vtkPolygonalSurfaceContourLineInterpolator()
        interpolator.GetPolys().AddItem(self.pd)
        rep.SetLineInterpolator(interpolator)


    def active_contour_widget2(self):
        # global contourWidget
        if not self.partial_actor in self.ren2.GetActors():
            return

        self.contourWidget2 = vtk.vtkContourWidget()
        self.contourWidget2.SetInteractor(self.iren2)
        # contourWidget.AddObserver()
        rep = vtk.vtkOrientedGlyphContourRepresentation.SafeDownCast(self.contourWidget2.GetRepresentation())
        rep.GetLinesProperty().SetColor(1, 0.2, 0)
        rep.GetLinesProperty().SetLineWidth(3.0)

        pointPlacer = vtk.vtkPolygonalSurfacePointPlacer()
        pointPlacer.AddProp(self.partial_actor)
        pointPlacer.GetPolys().AddItem(self.partial_actor.GetMapper().GetInput())
        rep.SetPointPlacer(pointPlacer)

        interpolator = vtk.vtkPolygonalSurfaceContourLineInterpolator()
        interpolator.GetPolys().AddItem(self.partial_actor.GetMapper().GetInput())
        rep.SetLineInterpolator(interpolator)

        self.contourWidget2.SetEnabled(True)


    def reset_camera2(self):
        c = []
        for act in self.ren2.GetActors():
            center = act.GetCenter()
            c.append(np.array(center))

        if c:
            c = np.mean(c, axis=0)

            camera = self.ren2.GetActiveCamera()
            camera.SetFocalPoint(c[0], c[1], c[2])
            camera.SetPosition(c[0] + 50, c[1], c[2])
            camera.SetViewUp(0, 0, -1)

    def keypress_event(self, obj, event):
        key = obj.GetKeySym()
        if key == "g":
            # contourWidget.EnabledOff()

            pp = self.contourWidget.GetRepresentation().GetContourRepresentationAsPolyData()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(pp)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            actor.GetProperty().SetColor(0, 1, 0)
            self.ren.AddActor(actor)


            # f = vtk.vtkVoxelContoursToSurfaceFilter()
            # f.SetInputData(pp)
            # f.SetMemoryLimitInBytes(100000)

            # filter = vtk.vtkTransformFilter()
            # filter.SetInputConnection(f.GetOutputPort())
            # filter.SetTransform(transform)
            # filter.Update()
            loop = vtk.vtkSelectPolyData()
            loop.SetInputData(self.polyData)
            loop.SetLoop(pp.GetPoints())
            loop.GenerateSelectionScalarsOn()
            # loop.SetSelectionModeToSmallestRegion() # negative scalars inside
            loop.SetSelectionModeToLargestRegion()

            clip = vtk.vtkClipPolyData()
            clip.SetInputConnection(loop.GetOutputPort())
            clip.Update()

            if False:
                edges = loop.GetSelectionEdges()
                source_points = numpy_support.vtk_to_numpy(self.polyData.GetPoints().GetData())
                edge_vertex_indices = numpy_support.vtk_to_numpy(edges.GetLines().GetData())

                marks = worklist_convert.create_spheres(source_points[edge_vertex_indices[0:1]])
                # worklist_convert.apply_color(marks)
                polydactor = worklist_convert.polydata_to_acctor(self.polyData)
                worklist_convert.render_show([*marks, polydactor])





                points = numpy_support.vtk_to_numpy(clip.GetOutput().GetPoints().GetData())
                marks = worklist_convert.create_spheres(points)
                tooth_act = worklist_convert.polydata_to_acctor(self.polyData)
                marks.append(tooth_act)
                worklist_convert.render_show(marks)

            # copy = vtk.vtkPolyData()
            # copy.DeepCopy(clip.GetOutput())
            # source_pd = sphereSource.GetOutput()
            src_pts = numpy_support.vtk_to_numpy(self.polyData.GetPoints().GetData())

            pp = vtk.vtkPolyData()
            pp.DeepCopy(clip.GetOutput())

            ex_pts = numpy_support.vtk_to_numpy(pp.GetPoints().GetData())

            near = NearestNeighbors()
            near.fit(src_pts)
            dist, inds = near.kneighbors(ex_pts, n_neighbors=1)
            # print(inds.shape, dist.max(), ex_pts.shape, src_pts.shape)




            m = vtk.vtkPolyDataMapper()
            m.SetInputConnection(clip.GetOutputPort())

            a = vtk.vtkActor()
            a.SetMapper(m)

            self.ren.AddActor(a)

            # transform = vtk.vtkTransform()
            # transform.Translate(1, 1, 1)

            # filter = vtk.vtkTransformFilter()
            # filter.SetInputData(pp)
            # filter.SetTransform(transform)
            # filter.Update()

            # worklist_convert.render_show_pd([filter.GetOutput()])

            # pd_copy = vtk.vtkPolyData()
            # pd_copy.DeepCopy(filter.GetOutput())
            pts = numpy_support.vtk_to_numpy(self.polyData.GetPoints().GetData())
            extract_pts = numpy_support.vtk_to_numpy(pp.GetPoints().GetData())
            cells = numpy_support.vtk_to_numpy(pp.GetPolys().GetData())

            near = NearestNeighbors()
            near.fit(pts)
            _, corresponding_inds = near.kneighbors(extract_pts, n_neighbors=1)
            # print(corresponding_inds.shape)
            items = {
                "polydata":pp,
                "points":extract_pts,
                "ids":corresponding_inds[:, 0],
                "cells":cells
            }

            # print("extract points : {}".format(extract_pts.shape))
            self.extract_data[self.spin.value()].append(items)



            # self.polydata




            # self.actor2.GetMapper().SetInputData(pp)


            self.add_in_render2(pp)
            self.reset_camera2()
            self.active_contour_widget2()




            #
            #
            #
            # mm = vtk.vtkPolyDataMapper()
            # mm.SetInputData(filter.GetOutput())
            # aa = vtk.vtkActor()
            # aa.SetMapper(mm)
            #
            # rn.AddActor(aa)

        elif key == "d":
            self.clear_items()
            # for it in self.ren.GetViewProps():
            #     if it == self.teeth_actor:
            #         pass
            #     else:
            #         self.ren.RemoveActor(it)
            # # contourWidget.GetContourRepresentation().
            # # contourWidget.GetRepresentation().ClearAllNodes()
            # self.activae_contour_widget()
            # self.contourWidget.EnabledOn()
            #
            # self.iren.Render()
            # self.iren2.Render()
        self.Render()

def catch_exceptions(t, val, tb):
    # Qt.QMessageBox.critical(None,
    #                                "An exception was raised",
    #                                "Exception type: {}".format(t))
    raise RuntimeError("An exception was raised",
                                   "Exception type: {}".format(t))
    exit()
    old_hook(t, val, tb)



def main():
    pass

if __name__=="__main__":
    old_hook = sys.excepthook
    sys.excepthook = catch_exceptions

    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
