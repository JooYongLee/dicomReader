import numpy as np
import os
import ctypes
from localData import DataReader
from configure import *
import vtk
import worklist_convert
from vtk.util import numpy_support
import vtk_utils
#https://stackoverflow.com/questions/18367007/python-how-to-write-to-a-binary-file

WRITE_FIlE_NAME = "testpoly.bin"
def test_creat_file():
    """
    buffer파일 생성
    :return:
    """
    n = 100

    x = np.linspace(0, n-1, n)[::-1]
    elemsize = np.prod(x.shape).astype(np.int32)


    x = x.astype(np.float32)

    with open("file.bin", "wb") as f:
        f.write(elemsize.tobytes())
        f.write(x.tobytes())
        # x.tofile(f)
    # ex_data = [i for i in range(100)]
    #
    # byte_data = bytearray(ex_data)
    #
    #
    # with open("file.bin", "wb") as f:
    #     f.write(byte_data)


    with open("file.bin", "rb") as f:
        int_size = ctypes.sizeof(ctypes.c_int32)

        data_size = np.frombuffer(f.read(int_size), dtype=np.int32)

        print("int size {}, data size {}".format(int_size, data_size))
        f.seek(0, os.SEEK_END)
        remain_size = (f.tell() - int_size) // ctypes.sizeof(ctypes.c_float)

        assert data_size[0] == remain_size
        f.seek(4)
        data = f.read()



    # print(data)

    d = np.frombuffer(data, dtype=np.float32)
    # print(d)
    print("write & read isclose ", np.isclose(x, d).all())

def read_from_c_implement():
    test_file = "D:/SVN/branch/AP_Testbed/AP_Testbed/testmatrix.bin"
    assert os.path.exists(test_file)


    with open(test_file, "rb") as f:
        data = f.read()



    d = np.frombuffer(data, dtype=np.float32)
    print(d)

class Converter(object):
    def __init__(self, filename=WRITE_FIlE_NAME, mode="wb"):
        assert mode in ["wb", "rb"]
        self.file = open(filename, mode)

    def addTeeth(self, tau):
        data = DataReader.get_teeth_polydata(tau)
        import worklist_convert
        # worklist_convert.render_show_pd([data])

        points = numpy_support.vtk_to_numpy(data.GetPoints().GetData())
        polys = numpy_support.vtk_to_numpy(data.GetPolys().GetData())
        points_size = np.array(points.shape, dtype=np.int32)
        polys_size = np.array(polys.shape, dtype=np.int32)

        # print(points.shape, polys.shape, polys_size.dtype, points_size.dtype, points_size)
        # print(points.dtype, polys.dtype)

        fixed_name_size = 256
        name = "tooth{:02}\0".format(tau)
        teeth_number = np.array(tau, dtype=np.int32)
        # with open(WRITE_FIlE_NAME, "wb") as f:
        self.file.write(bytearray(name.ljust(fixed_name_size).encode()))
        self.file.write(teeth_number.tobytes())
        self.file.write(points_size.tobytes())
        self.file.write(polys_size.tobytes())
        self.file.write(points.tobytes())
        self.file.write(polys.tobytes())

    def readData(self):

        int_32_size = ctypes.sizeof(ctypes.c_int32)
        int_64_size = ctypes.sizeof(ctypes.c_int64)
        float_32_size = ctypes.sizeof(ctypes.c_float)
        fixed_name_size = 256

        self.file.seek(0, os.SEEK_END)
        read_data_size = self.file.tell()
        self.file.seek(0, os.SEEK_SET)

        concat_size = 0

        polydata_list = []
        while concat_size < read_data_size:
            self.file.seek(concat_size, os.SEEK_SET)
            # header 읽기
            name_bytes = self.file.read(fixed_name_size)


            teeth_number= np.frombuffer(self.file.read(int_32_size), dtype=np.int32)


            header_size = fixed_name_size + int_32_size + int_32_size*2 + int_32_size*2

            points_shape = np.frombuffer(self.file.read(int_32_size*2), dtype=np.int32)
            points_size = np.prod(points_shape)

            polys_shape = np.frombuffer(self.file.read(int_32_size * 2), dtype=np.int32)
            polys_size = np.prod(polys_shape)

            contents_size = points_size * float_32_size + polys_size * int_64_size

            print(name_bytes.decode())
            print(teeth_number)
            # assert read_data_size == header_size + contents_size
            concat_size += header_size + contents_size


            buffer = self.file.read(float_32_size*points_size)
            points = np.frombuffer(buffer, dtype=np.float32)
            points_reshape = points.reshape(points_shape)


            buffer = self.file.read(int_64_size*polys_size)
            polys = np.frombuffer(buffer, dtype=np.int64)
            polys_reshape = polys.reshape(polys_shape)


        #vtk_utils.create_points_test(points_reshape)
            polydata = vtk_utils.reconstruct_polydata(points_reshape, polys_reshape)


            polydata_list.append(polydata)


        return polydata_list



    def __del__(self):
        self.file.close()



def convert_test():
    # tau = 17
    converter = Converter(mode="wb")
    for tau in [11, 12, 13, 14, 15, 16, 17, 31, 32, 33, 34, 35, 36, 37]:
        converter.addTeeth(tau)
    # del converter

    converter_loader = Converter(mode="rb")
    datas = converter_loader.readData()
    print(len(datas))
    worklist_convert.render_show_pd(datas)


    # data = DataReader.get_teeth_polydata(tau)
    # import worklist_convert
    # # worklist_convert.render_show_pd([data])
    #
    # points = numpy_support.vtk_to_numpy(data.GetPoints().GetData())
    # polys = numpy_support.vtk_to_numpy(data.GetPolys().GetData())
    # points_size = np.array(points.shape, dtype=np.int32)
    # polys_size = np.array(polys.shape, dtype=np.int32)
    #
    #
    # print(points.shape, polys.shape, polys_size.dtype, points_size.dtype, points_size)
    # print(points.dtype, polys.dtype)
    #
    # fixed_name_size = 256
    # name = "tooth{:02}\0".format(tau)
    # teeth_number = np.array(tau, dtype=np.int32)
    # with open(WRITE_FIlE_NAME, "wb") as f:
    #     f.write(bytearray(name.ljust(fixed_name_size).encode()))
    #     f.write(teeth_number.tobytes())
    #     f.write(points_size.tobytes())
    #     f.write(polys_size.tobytes())
    #     f.write(points.tobytes())
    #     f.write(polys.tobytes())

def load_test():
    with open(WRITE_FIlE_NAME, "rb") as file:
        int_32_size = ctypes.sizeof(ctypes.c_int32)
        int_64_size = ctypes.sizeof(ctypes.c_int64)
        float_32_size = ctypes.sizeof(ctypes.c_float)
        fixed_name_size = 256

        file.seek(0, os.SEEK_END)
        read_data_size = file.tell()
        file.seek(0, os.SEEK_SET)


        # header 읽기
        name_bytes = file.read(fixed_name_size)

        print(name_bytes.decode())
        teeth_number= np.frombuffer(file.read(int_32_size), dtype=np.int32)
        print(teeth_number)

        header_size = fixed_name_size + int_32_size + int_32_size*2 + int_32_size*2

        points_shape = np.frombuffer(file.read(int_32_size*2), dtype=np.int32)
        points_size = np.prod(points_shape)

        polys_shape = np.frombuffer(file.read(int_32_size * 2), dtype=np.int32)
        polys_size = np.prod(polys_shape)

        contents_size = points_size * float_32_size + polys_size * int_64_size
        assert read_data_size == header_size + contents_size


        buffer = file.read(float_32_size*points_size)
        points = np.frombuffer(buffer, dtype=np.float32)
        points_reshape = points.reshape(points_shape)
        print("points", points_size)

        # assert buff


        buffer = file.read(int_64_size*polys_size)
        polys = np.frombuffer(buffer, dtype=np.int64)
        polys_reshape = polys.reshape(polys_shape)
        print(polys_reshape.shape)

        #vtk_utils.create_points_test(points_reshape)
        polydata = vtk_utils.reconstruct_polydata(points_reshape, polys_reshape)

        print(points_reshape[:10])
        print(polys_reshape[:10])

        worklist_convert.render_show_pd([polydata])

        # size = np.frombuffer(f.read(int_32_size) * 2, dtype=np.int32)

        print("polys", polys_size)
    # a = np.arange(2).astype(np.int32)
    # b = np.arange(10).reshape([5, 2]).astype(np.float32)
    # c = np.arange(6).reshape([3, 2]).astype(np.float32)
    # print(b.dtype)
    # with open("testss.bin", "wb") as f:
    #     f.write(a.tobytes())
    #     f.write(b.tobytes())
    #     f.write(c.tobytes())
    #
    # with open("testss.bin", "rb") as f:
    #     aa = f.read(2*4)
    #     np_s = np.frombuffer(aa, dtype=np.int32)
    #     np_f = np.frombuffer(f.read(10*4), dtype=np.float32)
    #     dd = np.frombuffer(f.read(), dtype=np.float32)
    #
    #     print(np_s)
    #     print(np_f)
    #     print(dd)


        # f.write(points_size.tobytes())
        # f.write(points.tobytes())
        # f.write(polys_size.tobytes())
        # f.write(polys.tobytes())


if __name__=="__main__":
    # test_creat_file()

    # print(ctypes.sizeof(ctypes.c_int32))

    # test_creat_file()
    convert_test()


    # load_test()
    #read_from_c_implement()


# WRITE_FIlE_NAME = "testfile.bin"