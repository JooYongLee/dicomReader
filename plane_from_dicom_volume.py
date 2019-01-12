import pydicom
import dicom_numpy
import glob
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import scipy
import os
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
            if len(ext_boxes) == 2:
                ctr_x = (box[0]+box[2])/2
                if ctr_x < img_shape[1]/2:
                    ext_box = box
            elif len(ext_boxes)==1:
                # 1:2, 2:1
                x_max = box[2]
                x_min = box[0]
                internal_x = (1*x_max + 2*x_min)/3
                ext_box = box
                ext_box[2] = internal_x
                # y is identical, preserve

    elif pos =="right":
        for box in ext_boxes:
            if len(ext_boxes) == 2:
                ctr_x = (box[0]+box[2])/2
                if ctr_x > img_shape[1]/2:
                    ext_box = box
            elif len(ext_boxes) == 1:
                # 2:1
                x_max = box[2]
                x_min = box[0]
                internal_x = (2*x_max + 1*x_min)/3
                ext_box = box
                ext_box[0] = internal_x


    else:
        raise NotImplemented


    return np.array(ext_box)
def sorting_pts(pts,axis):
    """
    :param pts: N X 3 , [x,y,z]
    :return:
    """
    assert pts.shape[1] > axis
    ext_x = pts[:, 0]
    ext_y = pts[:, 1]
    ext_z = pts[:, 2]
    ex_pts = pts[:, axis]
    sort_inds =np.argsort(ex_pts)
    return np.stack([
        ext_x[sort_inds],
        ext_y[sort_inds],
        ext_z[sort_inds]
        ], axis=1)

def split_outer_inner_arch_pt(ext_boxes, num_slice):
    outer_arch_pts = []
    inner_arch_pts = []
    if ext_boxes.shape[0] == 2:
        x_ctr = (ext_boxes[:, 2] + ext_boxes[:, 0]) / 2
        # y_ctr = (ext_boxes[:,3] + ext_boxes[:,1])/2
        inds = np.argsort(x_ctr)
        left_box = ext_boxes[inds[0]]
        right_box = ext_boxes[inds[1]]

        ## outer box
        outer_arch_pts.append([
            left_box[0],
            (left_box[1] + left_box[3]) / 2,
            num_slice
        ])
        outer_arch_pts.append([
            right_box[2],
            (right_box[1] + right_box[3]) / 2,
            num_slice
        ])

        ## inner box
        inner_arch_pts.append([
            left_box[2],
            (left_box[1] + left_box[3]) / 2,
            num_slice
        ])
        inner_arch_pts.append([
            right_box[0],
            (right_box[1] + right_box[3]) / 2,
            num_slice
        ])
    elif ext_boxes.shape[0] == 1:
        ext_boxes = ext_boxes[0]
        outer_arch_pts.append([
            ext_boxes[0],
            np.sum(ext_boxes[0::2] / 2),
            num_slice
        ])
        outer_arch_pts.append([
            ext_boxes[2],
            np.sum(ext_boxes[1::2] / 2),
            num_slice
        ])
        inner_arch_pts.append([
            np.sum(ext_boxes[::2] / 2),
            np.sum(ext_boxes[1::2] / 2),
            num_slice
        ])
    else:
        pass
    return outer_arch_pts, inner_arch_pts
def load_josn_outter_inner_arch(basepath, volumeshape, ext_class = "lowercase", pos = "left"):
    """
    volume-shape [y,z,x]
    pts [ x,y,z]
    :param path_list: json annotation path
    :param ext_class: extracted class name
    :param pos:
    :return: outer points array,  inner points array of the arch
    """
    if not os.path.isdir(basepath):
        raise  ValueError("cannot find dir.{}".format(basepath))
    path_list = glob.glob(basepath + '/*.json')
    inner_arch_pts = []
    outer_arch_pts = []
    for ind, path in enumerate(path_list):
        fname = os.path.splitext(os.path.basename(path))[0]
        num_slice = int(fname)
        json_obj = load_json(path)
        # ext_boxes = extract_class(json_obj,ext_class)
        # if ext_boxes.size == 0:
        ext_boxes = extract_class(json_obj, ext_class)
        # if( lowercase_boxes.size > 0 ):
        #     outer_arch_pt, inner_arch_pt = split_outer_inner_arch_pt(lowercase_boxes, num_slice)
        # else:
        #     print("------------get nerve sampling-----------------")
        #     nerve_boxes = extract_class(json_obj, "nerve")
        outer_arch_pt, inner_arch_pt = split_outer_inner_arch_pt(ext_boxes, num_slice)
        outer_arch_pts += outer_arch_pt
        inner_arch_pts += inner_arch_pt

    try:
        inner_arch_pts = np.stack(inner_arch_pts, axis=0)
        outer_arch_pts = np.stack(outer_arch_pts, axis=0)
        outer_arch_pts = sorting_pts(outer_arch_pts, axis=0)
        inner_arch_pts = sorting_pts(inner_arch_pts, axis=0)

    except:
        print(inner_arch_pts)
        raise  ValueError
    return outer_arch_pts, inner_arch_pts

def load_josn_and_parser(path_list, ext_class = "nerve", pos = "left"):
    merge_pnt = []
    for ind, path in enumerate(path_list):
        fname = os.path.splitext(os.path.basename(path))[0]
        num_slice = int(fname)
        json_obj = load_json(path)
        ext_boxes = extract_class(json_obj,ext_class)
        if ext_boxes.size == 0:
            ext_boxes = extract_class(json_obj, "lowercase")


        for box in ext_boxes:
            x_ctr = (box[0] + box[2])/2
            y_ctr = (box[1] + box[3]) / 2
            merge_pnt.append([ x_ctr,
                              y_ctr,
                               num_slice
                              ])
    merge_pnt = np.stack(merge_pnt,axis=0)
    merge_pnt = sorting_pts(merge_pnt,axis=0)
    return merge_pnt



        # split_ext_box = split_box_pos(ext_boxes,pos,img_shape)
        # if split_ext_box.size>0:
        #     x_ctr = (split_ext_box[0] + split_ext_box[2])/2
        #     y_ctr = (split_ext_box[1] + split_ext_box[3]) / 2
        #     merge_pnt.append([ x_ctr,
        #                       y_ctr,
        #                        num_slice
        #                       ])
    # return np.stack(merge_pnt,axis=0)
def load_jsonlist_to_outer_inner_pts(json_path):

    if not os.path.isdir(json_path):
        raise ValueError("cannot find dir:{}".format(json_path))
    json_list = glob.glob(json_path + '/*.json')
    load_josn_outter_inner_arch(json_list)
def load_jsonlist_to_boxes_pts(json_path):
    # json_path = 'C:/DNN/dataset/67998_171116120306(4)/67998_171116120306(4)/anno'
    json_list = glob.glob(json_path+'/*.json')
    box_pnt = load_josn_and_parser(json_list)
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

def show_plane_from_volume(volumes,x=-1,y=-1,z=-1):
    """
    :param volumes:[y,z,x]
    :param x:
    :param y:
    :param z:
    :return:
    """
    ix = 100 if x < 0 else x
    iy = 330 if y < 0 else y
    iz = 300 if z < 0 else z
    plt.subplot(131)
    # coronal
    plt.imshow(volumes[:,iz,:],cmap='gray')
    # plt.plot(x,y)
    plt.subplot(132)
    #sagittal
    plt.imshow(volumes[:,:,ix],cmap='gray')
    plt.subplot(133)
    plt.imshow(volumes[iy, :, :], cmap='gray')
def get_plane_cube_points_array(outer_arch_pts, innter_arch_pts, plane_shape,volume_shape, L =20, color='r'):
    """

    :param outer_arch_pts: [x,y,z]
    :param innter_arch_pts:
    :param plane_shape:
    :param volume_shape:[y,z,x]
    :param L:
    :param color:
    :return:
    """
    length_x = volume_shape[2]
    length_y = volume_shape[0]
    length_z = volume_shape[1]
    M = plane_shape[0]
    N = plane_shape[1]
    outer_xx = np.minimum(np.maximum(outer_arch_pts[:, 0], 0),length_x)
    outer_yy = np.minimum(np.maximum(outer_arch_pts[:, 1], 0),length_y)
    outer_zz = np.minimum(np.maximum(outer_arch_pts[:, 2], 0),length_z)

    inner_xx = np.minimum(np.maximum(innter_arch_pts[:, 0], 0),length_x)
    inner_yy = np.minimum(np.maximum(innter_arch_pts[:, 1], 0),length_y)
    inner_zz = np.minimum(np.maximum(innter_arch_pts[:, 2], 0),length_z)

    # line smoothing
    p = scipy.polyfit(inner_xx, inner_zz, 2)
    inner_x = np.linspace(inner_xx.min(), inner_xx.max(), N)
    inner_z = scipy.polyval(p, inner_x)

    # line smoothing
    p = scipy.polyfit(outer_xx, outer_zz, 2)
    outer_x = np.linspace(outer_xx.min(), outer_xx.max(), N)
    outer_z = scipy.polyval(p, outer_x)
    # L = 20
    xs = []
    zs = []
    heights = 150
    avg_y = inner_yy.mean()
    start_y = np.maximum(avg_y - heights, 0)
    end_y = np.minimum(avg_y + heights, volume_shape[0])
    iy = np.linspace(start_y, end_y, M)
    vy = np.reshape(iy, [M, 1, 1])
    vy = np.repeat(vy, N, 1)
    vy = np.expand_dims(vy,axis=0)
    vy = np.repeat(vy,L,0)


    for in_x,in_z,out_x,out_z in zip(inner_x,inner_z,outer_x,outer_z):
        xs.append(np.linspace(in_x,out_x,L))
        zs.append(np.linspace(in_z,out_z,L))

    xs = np.array(xs).transpose()
    zs = np.array(zs).transpose()

    sample = [xs, zs]
    graph_show = False
    if graph_show:
        # sample = [xs,zs]
        # for i in range(outer_xx.size):
        # plt.plot(outer_xx,outer_zz,'r*')
        plt.plot(    inner_xx,     inner_zz, 'g*')

        plt.show()


    vx = np.reshape(xs, [L, 1, N, 1])
    vz = np.reshape(zs, [L, 1, N, 1])

    vx = np.repeat(vx, M, 1)
    vz = np.repeat(vz, M, 1)

    return np.concatenate([
        vy,vz,vx
    ],axis=3), sample



def get_plane_points_array(box_pnt, plane_shape,volume_shape,color='r'):
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
    p = scipy.polyfit(xx, zz, 2)
    ix = np.linspace(xx.min(), xx.max(), N)
    iz = scipy.polyval(p, ix)

    # line smoothing
    _x_ = np.arange(yy.shape[0])
    py = scipy.polyfit(_x_,yy,2)
    yy = scipy.polyval(py,_x_)

    plt.plot(xx,zz,color+'*')
    plt.plot(ix, iz)
    # plt.show()
    heights = 300

    # iy = np.linspace(yy.min(),yy.max(),N)
    dy = []

    fixed_y = True
    if fixed_y:
        avg_y = yy.mean()
        start_y = np.maximum(avg_y - heights, 0)
        end_y = np.minimum(avg_y + heights, volume_shape[0])
        iy = np.linspace(start_y,end_y,M)
        vy = np.reshape(iy, [M, 1, 1])
        vy =  np.repeat(vy, N,1)


    else:
        iy = np.interp(np.linspace(0,yy.shape[0],N),np.arange(yy.shape[0]),yy)

        for y in iy:
            start_y  = np.maximum(y - heights,0)
            end_y = np.minimum(y+heights,volume_shape[0])
            dy.append(np.linspace(start_y, end_y,M))
        vy = np.stack(dy,axis=1)
        vy = np.expand_dims(vy,axis=2)



    vx = np.reshape(ix, [1, N, 1])
    vz = np.reshape(iz, [1, N, 1])

    vx = np.repeat(vx, M, 0)
    vz = np.repeat(vz, M, 0)

    # planes = np.concatenate([vz,vy,vz],axis=2)
    planes = np.concatenate([vy, vz, vx], axis=2)
    return planes
def sample_show(outer_sample, inner_sample):

    graph_show = True
    if graph_show:
        outer_xs = outer_sample[0]
        outer_zs = outer_sample[1]
        inner_xs = inner_sample[0]
        inner_zs = inner_sample[1]
        # sample = [xs,zs]
        for i in range(outer_xs.shape[0]):
            if i % 5 == 0:
                plt.plot(outer_xs[i],outer_zs[i],'r')

        for i in range(inner_xs.shape[0]):
            if i % 5 == 0:
                plt.plot(inner_xs[i],inner_zs[i],'b')

        plt.show()
def visualize_plane_image_from_voxel_dicom_nerve():


    # dcmpath = "D:/DataSet/DataSet2018/20181113/" + worklist + '/CTData'
    casename = "67998_171201192022 (3) (4)"
    # dcmpath = "C:/DNN/validation/20181113/67998_171121112638 (3) (4)/CTData"
    # D:\DataSet\DataSet2018\validation\20181113\67998_171129100349(3)(4)
    dcmpath = "D:/DataSet/DataSet2018/All/20181113/" + casename + "/CTData"
    dcm_list = glob.glob(dcmpath +'/*.dcm')
    print(len(dcm_list))
    # json_path = "D:/DataSet/DataSet2018/validation/20181113/" + worklist + '/anno'
    # json_path = "C:/DNN/validation/20181113/67998_171121112638 (3) (4)/anno"
    json_path = "D:/DataSet/Evaluation/test_image_all_0102/" + casename + "/anno"
    # json_path = "D:/DataSet/DataSet2018/testset/" + worklist + '/anno'
    voldumedata = extract_voxel_data(dcm_list)

    sampling_size = 35

    scales = 255.0/(voldumedata.max()-voldumedata.min())
    volum_8bits = (voldumedata  - voldumedata.min())*scales
    volum_8bits = volum_8bits.astype(np.uint8).transpose()
    # conver axis...
    volum_8bits = volum_8bits[::-1,:,:]
    print("volume image------------",volum_8bits.shape)


    # N X  [x,y,z]
    # box_pnt = load_jsonlist_to_boxes_pts(json_path)
    outer_arch_lower_pts, inner_arch_lower_pts = load_josn_outter_inner_arch(json_path, volum_8bits.shape,ext_class="lowercase")
    outer_arch_nerve_pts, inner_arch_nerve_pts = load_josn_outter_inner_arch(json_path, volum_8bits.shape, ext_class="nerve")


    mid_y = int(outer_arch_lower_pts[:,1].mean())
    show_plane_from_volume(volum_8bits,y=mid_y)

    plane_shape = [200,400]
    all_plane =  True

    save_dir = casename
    if not os.path.isdir(save_dir) : os.mkdir(save_dir)
    if all_plane :

        # planes_list = get_plane_cube_points_array(outer_arch_lower_pts, inner_arch_lower_pts, plane_shape, volum_8bits.shape, L = sampling_size, color='g')

        outer_planes_list, outer_sample = get_plane_cube_points_array(outer_arch_lower_pts, outer_arch_nerve_pts, plane_shape,
                                                  volum_8bits.shape, L=sampling_size, color='g')

        inner_planes_list, inner_sample = get_plane_cube_points_array(outer_arch_nerve_pts, inner_arch_lower_pts, plane_shape,
                                                  volum_8bits.shape, L=sampling_size, color='g')

        # sample_show(outer_sample, inner_sample)
        plane_list = [*inner_planes_list,*outer_planes_list]
        # for cnt, plane in enumerate(planes_list[::-1]):
        for cnt, plane in enumerate(plane_list[::-1]):
            plane_reshape = np.reshape(plane,[-1,3])
            plane_interp = plane_from_volume(volum_8bits,plane_reshape)
            plane_image = plane_interp.reshape(plane_shape)

            plt.imsave("{}/{}.png".format(save_dir,cnt),plane_image.astype(np.uint8),cmap='gray')
    else:
        outer_planes = get_plane_points_array(outer_arch_pts,plane_shape,volum_8bits.shape,color='r')
        inner_planes = get_plane_points_array(inner_arch_pts, plane_shape, volum_8bits.shape, color='g')

        outer_planes_reshape = np.reshape(outer_planes,[-1,3])
        outerplane_interp = plane_from_volume(volum_8bits,outer_planes_reshape)
        outerplane_image = outerplane_interp.reshape(plane_shape)

        inner_planes_reshape = np.reshape(inner_planes, [-1, 3])
        inner_planes_interp = plane_from_volume(volum_8bits, inner_planes_reshape)
        inner_planes_image = inner_planes_interp.reshape(plane_shape)

        fig1 = plt.figure()
        axe1 = fig1.add_subplot(211)
        axe2 = fig1.add_subplot(212)
        axe1.imshow(outerplane_image,cmap='gray')
        axe2.imshow(inner_planes_image, cmap='gray')
        plt.show()


def visualize_plane_image_from_voxel_dicom():


    # dcmpath = "D:/DataSet/DataSet2018/20181113/" + worklist + '/CTData'
    casename = "76278_171123124810_(3) (4)"
    # dcmpath = "C:/DNN/validation/20181113/67998_171121112638 (3) (4)/CTData"
    # D:\DataSet\DataSet2018\validation\20181113\67998_171129100349(3)(4)
    dcmpath = "D:/DataSet/DataSet2018/All/20181113/" + casename + "/CTData"
    dcm_list = glob.glob(dcmpath +'/*.dcm')
    print(len(dcm_list))
    # json_path = "D:/DataSet/DataSet2018/validation/20181113/" + worklist + '/anno'
    # json_path = "C:/DNN/validation/20181113/67998_171121112638 (3) (4)/anno"
    json_path = "D:/DataSet/panorama/source_20181113/" + casename + "/anno"
    # json_path = "D:/DataSet/DataSet2018/testset/" + worklist + '/anno'
    voldumedata = extract_voxel_data(dcm_list)

    sampling_size = 70

    scales = 255.0/(voldumedata.max()-voldumedata.min())
    volum_8bits = (voldumedata  - voldumedata.min())*scales
    volum_8bits = volum_8bits.astype(np.uint8).transpose()
    # conver axis...
    volum_8bits = volum_8bits[::-1,:,:]
    print("volume image------------",volum_8bits.shape)


    # N X  [x,y,z]
    # box_pnt = load_jsonlist_to_boxes_pts(json_path)
    outer_arch_pts, inner_arch_pts = load_josn_outter_inner_arch(json_path, volum_8bits.shape)

    mid_y = int(outer_arch_pts[:,1].mean())
    show_plane_from_volume(volum_8bits,y=mid_y)

    plane_shape = [200,400]
    all_plane =  True

    save_dir = casename
    if not os.path.isdir(save_dir) : os.mkdir(save_dir)
    if all_plane :

        planes_list = get_plane_cube_points_array(outer_arch_pts, inner_arch_pts, plane_shape, volum_8bits.shape, L = sampling_size, color='g')

        # for cnt, plane in enumerate(planes_list[::-1]):
        for cnt, plane in enumerate(planes_list[::-1]):
            plane_reshape = np.reshape(plane,[-1,3])
            plane_interp = plane_from_volume(volum_8bits,plane_reshape)
            plane_image = plane_interp.reshape(plane_shape)

            plt.imsave("{}/{}.png".format(save_dir,cnt),plane_image.astype(np.uint8),cmap='gray')

    else:


    #
        outer_planes = get_plane_points_array(outer_arch_pts,plane_shape,volum_8bits.shape,color='r')
        inner_planes = get_plane_points_array(inner_arch_pts, plane_shape, volum_8bits.shape, color='g')

        outer_planes_reshape = np.reshape(outer_planes,[-1,3])
        outerplane_interp = plane_from_volume(volum_8bits,outer_planes_reshape)
        outerplane_image = outerplane_interp.reshape(plane_shape)

        inner_planes_reshape = np.reshape(inner_planes, [-1, 3])
        inner_planes_interp = plane_from_volume(volum_8bits, inner_planes_reshape)
        inner_planes_image = inner_planes_interp.reshape(plane_shape)

        fig1 = plt.figure()
        axe1 = fig1.add_subplot(211)
        axe2 = fig1.add_subplot(212)
        axe1.imshow(outerplane_image,cmap='gray')
        axe2.imshow(inner_planes_image, cmap='gray')
        plt.show()


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
    # visualize_plane_image_from_voxel_dicom()
    visualize_plane_image_from_voxel_dicom_nerve()
