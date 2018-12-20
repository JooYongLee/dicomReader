import pydicom
import dicom_numpy
import glob
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import scipy

############################## load json##############################
def load_json(path):
    with open(path,"r") as f:
        return json.load(f)
def extract_class(json_obj,class_str):
    """
    :param json_obj:json object
    :param class_str: to be extracted the name of the class
    :return: the list of box, the box of the extracting class
    """
    boxes_list = []

    for key, boxes in json_obj.items():
        if key == class_str:
            boxes_list.append(np.array(boxes).reshape([-1,4]))
    if boxes_list:
        return np.concatenate(boxes_list,axis=0)
    else:
        return np.array([])
def split_box_pos(ext_boxes,pos,img_shape):
    ext_box = []
    if pos =="left":
        for box in ext_boxes:
            ctr_x = (box[0]+box[2])/2
            if ctr_x < img_shape[1]/2:
                ext_box = box
    elif pos =="right":
        for box in ext_boxes:
            ctr_x = (box[0]+box[2])/2
            if ctr_x > img_shape[1]/2:
                ext_box = box
    else:
        raise NotImplemented
    return np.array(ext_box)

def load_josn_and_parser(path_list,img_shape, ext_class = "lowercase", pos = "right"):
    merge_pnt = []
    for ind, path in enumerate(path_list):
        fname = os.path.splitext(os.path.basename(path))[0]
        num_slice = int(fname)
        json_obj = load_json(path)
        ext_boxes = extract_class(json_obj,ext_class)
        split_ext_box = split_box_pos(ext_boxes,pos,img_shape)
        if split_ext_box.size>0:
            x_ctr = (split_ext_box[0] + split_ext_box[2])/2
            y_ctr = (split_ext_box[1] + split_ext_box[3]) / 2
            merge_pnt.append([ x_ctr,
                              y_ctr,
                               num_slice
                              ])
    return np.stack(merge_pnt,axis=0)

def load_json_test():
    json_path = 'C:/DNN/dataset/67998_171116120306(4)/67998_171116120306(4)/anno'
    json_list = glob.glob(json_path+'/*.json')
    box_pnt = load_josn_and_parser(json_list,[450,550])
    box_pnt = box_pnt.astype(np.int)
    return box_pnt

############################## load volume, voxel ##############################

def extract_voxel_data(list_of_dicom_files):
    datasets = [pydicom.read_file(f) for f in list_of_dicom_files]
    try:
        voxel_ndarray, ijk_to_xyz = dicom_numpy.combine_slices(datasets)
    except dicom_numpy.DicomImportException as e:
        # invalid DICOM data
        raise
    return voxel_ndarray

def show_plane_from_volume(volumes):

    plt.subplot(131)
    # coronal
    plt.imshow(volumes[:,300,:],cmap='gray')
    x = np.linspace(0,100,100)
    y = 2*x
    # plt.plot(x,y)
    plt.subplot(132)
    #sagittal
    plt.imshow(volumes[:,:,100],cmap='gray')
    plt.subplot(133)
    plt.imshow(volumes[322, :, :], cmap='gray')
def get_plane_points_array(box_pnt, plane_shape):
    """
    :param box_pnt: N X 3 , [x,y,z]
    :param plane_shape:
    :return:
    """
    M = plane_shape[0]
    N = plane_shape[1]
    xx = box_pnt[:, 0]
    zz = box_pnt[:, 2]
    yy = box_pnt[:, 1]

    # line smoothing
    p = scipy.polyfit(xx, zz, 3)
    ix = np.linspace(xx.min(), xx.max(), N)
    iz = scipy.polyval(p, ix)

    # line smoothing
    _x_ = np.arange(yy.shape[0])
    py = scipy.polyfit(_x_,yy,3)
    yy = scipy.polyval(py,_x_)


    plt.plot(ix, iz)
    # plt.show()
    heights = 100

    # iy = np.linspace(yy.min(),yy.max(),N)
    dy = []
    iy = np.interp(np.linspace(0,yy.shape[0],N),np.arange(yy.shape[0]),yy)
    for y in iy:
        dy.append(np.linspace(y-100,y+100,M))
    vy = np.stack(dy,axis=1)
    vy = np.expand_dims(vy,axis=2)

    # dy = np.linspace()
    # dy = np.linspace(200, 440, M)
    # (200,)

    # 200 X 100
    # M = dy.shape[0]
    # N = ix.shape[0]
    # vy = np.reshape(dy, [M, 1, 1])
    # vy = np.repeat(vy, N, 1)

    vx = np.reshape(ix, [1, N, 1])
    vz = np.reshape(iz, [1, N, 1])

    vx = np.repeat(vx, M, 0)
    vz = np.repeat(vz, M, 0)

    # planes = np.concatenate([vz,vy,vz],axis=2)
    planes = np.concatenate([vy, vz, vx], axis=2)
    return planes
def visualize_plane_image_from_voxel_dicom():
    dcmpath = 'C:/DNN/dataset/67998_171116120306(4)/67998_171116120306(4)/CTData2'
    dcm_list = glob.glob(dcmpath +'/*.dcm')
    voldumedata = extract_voxel_data(dcm_list)

    scales = 255.0/(voldumedata.max()-voldumedata.min())
    volum_8bits = (voldumedata  - voldumedata.min())*scales
    volum_8bits = volum_8bits.astype(np.uint8).transpose()
    # conver axis...
    volum_8bits = volum_8bits[::-1,:,:]
    print("volume image------------",volum_8bits.shape)

    show_plane_from_volume(volum_8bits)
    box_pnt = load_json_test()

    plane_shape = [200,200]
    planes = get_plane_points_array(box_pnt,plane_shape)

    planes_reshape = np.reshape(planes,[-1,3])
    plane_interp = plane_from_volume(volum_8bits,planes_reshape)

    planes_image = plane_interp.reshape(plane_shape)

    fig1 = plt.figure()
    axe1 = fig1.subplots()
    axe1.imshow(planes_image,cmap='gray')
    plt.show()

    #



def plane_from_volume(volumes,pts):
    """
    :param volumes: 3d volume data
    :param pts: extract points... N X 3 [x,y,z]
    :return:
    """
    from scipy.interpolate import RegularGridInterpolator
    x = np.linspace(0,volumes.shape[0],volumes.shape[0])
    y = np.linspace(0, volumes.shape[1],volumes.shape[1])
    z = np.linspace(0, volumes.shape[2],volumes.shape[2])
    interpolating_function = RegularGridInterpolator((x, y, z), volumes)

    return interpolating_function(pts)

def test_examples_interp():
    from scipy.interpolate import RegularGridInterpolator

    def f(x, y, z):
        return 2 * x ** 3 + 3 * y ** 2 - z

    x = np.linspace(1, 4, 11)
    y = np.linspace(4, 7, 22)
    z = np.linspace(7, 9, 33)
    data = f(*np.meshgrid(x, y, z, indexing='ij', sparse=True))
    # print(data.shape)
    # x = np.linspace(0, volumes.shape[0], volumes.shape[0])
    # y = np.linspace(0, volumes.shape[1], volumes.shape[1])
    # z = np.linspace(0, volumes.shape[2], volumes.shape[2])
    interpolating_function = RegularGridInterpolator((x, y, z), data)

    # return interpolating_function(pts)
    pts = np.array([
        [2.1, 6.2, 8.3],
        [3.3, 5.2, 7.1],
        [4,5,8]
    ])

    print(interpolating_function(pts))

if __name__=="__main__":
    visualize_plane_image_from_voxel_dicom()

