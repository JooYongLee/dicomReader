import glob
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
def load_ious_list(ious_path):
    import glob
    json_list = glob.glob(ious_path+'/*.json')

    iou_loads = []
    for iouspath in json_list:
        with open(iouspath,"r") as f:
            # print(iouspath)
            tmpious = json.load(f)
            if tmpious:
                # print(tmpious)
                iou_loads.append(tmpious)
    return iou_loads
def iou_load_savetest():

    test_path_list = [
        'sample'
    ]
    ## gt boxes 경로
    gt_pathname = 'sample/groundtruth_rect.txt'
    for ind, test_path in enumerate(test_path_list):
        ## prediction 결과 가져오기
        ious_load = load_ious_list(test_path)

        if ious_load :
            x,y = compare_merged_trackpoint(ious_load, gt_pathname,'result')



        else:
            print("pass",test_path)




def merge_name_boxes(name_list,boxes_pnt_list):
    """
    name_list ------------> x coordinate, make sure all integer
    boxes_pnt_list--------------> rectangular info to y, z center
    """
    x_coordinate = name_list.astype(np.float)
    y_coordinate = (boxes_pnt_list[:,0] + boxes_pnt_list[:,2])/2
    z_coordinate = (boxes_pnt_list[:,1] + boxes_pnt_list[:,3])/2
    return np.stack([x_coordinate,y_coordinate,z_coordinate],axis=1)


def compare_merged_trackpoint(iou_loads, gt_pathname, savefig_dir):
    """
    파일명 : x축
    단면 영상 height---> y축
    단명 영상 width----> z축
    :param iou_loads: list & dict type, dict key : ious, filename, boxes
    :param gt_pathname: gt txt files
    :param savefig_dir:
    :return:
    """
    if not os.path.isdir(savefig_dir) : os.mkdir(savefig_dir)


    ## prediction 결과 가져오기
    plt.close('all')
    name_list = []
    ious_list = []
    boxes_pnt_list = []
    for ious in iou_loads:
        for iou in ious:
            name_list.append(int(iou['filename']))
            ious_list.append(iou['overlaps'])
            boxes_pnt_list.append(iou['boxes'])
    ious_list = ious_list

    ## 파일명 순으로 sorting

    name_list = np.array(name_list)
    ious_list = np.array(ious_list)
    boxes_pnt_list = np.array(boxes_pnt_list)
    sort_ind = np.argsort(name_list)
    sorted_name = name_list[sort_ind]
    sorted_ious = ious_list[sort_ind]
    sorted_boxes = boxes_pnt_list[sort_ind]

    # 병합 prediction boxes
    # 파일명 : x축
    # 단면 영상 height---> y축
    # 단명 영상 width----> z축
    merge_point = merge_name_boxes(sorted_name, sorted_boxes)


    # 병합 gt boxes
    gt = np.loadtxt(gt_pathname, delimiter=',')
    gt_x = np.arange(176,404)
    gt_merged = merge_name_boxes(gt_x,gt)

    gt_merged /= 500
    merge_point /= 500
    # anlaysis_nerve_plotting(gt_merged)
    anlaysis_nerve_plotting_compare(gt_merged, merge_point)


    return sorted_name, sorted_ious
def get_interpolate(track_point,size = 500):
    num_true_pts = np.shape(track_point)[0]

    x_sample = track_point[::,0]
    y_sample = track_point[:,1]
    z_sample = track_point[:,2]

    tck, u = interpolate.splprep([x_sample, y_sample, z_sample], s=2)
    x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
    u_fine = np.linspace(0, 1, size)
    # u_fine = np.linspace(x_sample.min(), x_sample.max(), num_true_pts)
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)


    return np.stack([x_fine,y_fine,z_fine],axis=1), np.stack([x_knots,y_knots,z_knots],axis=1)
def compute_shortest_distance_between_skew_line(line01,line02):
    """
    reference to line01,compute shortest distance from line2
    :param line01: N X 3,
    :param line02: M X 3
    :return:
    """
    ex_line01 = np.expand_dims(line01, axis=1)
    ex_line02 = np.expand_dims(line02, axis=0)
    # N X M X 3
    diff_line = ex_line01 - ex_line02
    # N X M
    distance = np.sum(np.square(diff_line),axis=2)
    min_inds_line2 = np.argmin(distance,axis=1)

    # N X 3
    shortest_point_on_line2 = line02[min_inds_line2]
    # N X 3
    shortest_point_on_line1 = line01
    return shortest_point_on_line1, shortest_point_on_line2



def anlaysis_nerve_plotting_compare(track_point_01, track_point_02):
    """
    :param track_point:N X 3, array, [x,y,z],
    :return:
    """

    fine01, knots01 = get_interpolate(track_point_01.copy(),200)
    fine02, knots02 = get_interpolate(track_point_02.copy(),200)

    track_point_01 *= 500
    track_point_02 *= 500
    fine01 *= 500
    fine02 *= 500

    x_knots01 = track_point_01[:, 0]
    y_knots01 = track_point_01[:, 1]
    z_knots01 = track_point_01[:, 2]

    x_knots02 = track_point_02[:, 0]
    y_knots02 = track_point_02[:, 1]
    z_knots02 = track_point_02[:, 2]

    x_fine01 = fine01[:, 0]
    y_fine01 = fine01[:, 1]
    z_fine01 = fine01[:, 2]

    x_fine02 = fine02[:, 0]
    y_fine02 = fine02[:, 1]
    z_fine02 = fine02[:, 2]


    fig2 = plt.figure(2)
    ax3d = fig2.add_subplot(111, projection='3d')


    #
    ax3d.plot(x_fine01, y_fine01, z_fine01, 'g')
    ax3d.plot(x_fine02, y_fine02, z_fine02, 'r')

    short_line01, short_line02 = compute_shortest_distance_between_skew_line(fine01, fine02)
    norm_x = np.stack([short_line01[:, 0], short_line02[:,0]],axis=1)
    norm_y = np.stack([short_line01[:, 1], short_line02[:, 1]], axis=1)
    norm_z = np.stack([short_line01[:, 2], short_line02[:, 2]], axis=1)
    for ind in range(norm_x.shape[0]):
        ax3d.plot(norm_x[ind],norm_y[ind],norm_z[ind],'b-.')


    ax3d.set_xlabel('fornt')
    ax3d.set_ylabel('up')
    ax3d.set_zlabel('sagit')

    fig2.show()
    plt.show()
def anlaysis_nerve_plotting(track_point):
    """
    :param track_point:N X 3, array, [x,y,z]
    :return:
    """

    num_true_pts = np.shape(track_point)[0]

    num_sample_pts = 80
    # s_sample = np.linspace(0, total_rad, num_sample_pts)
    x_sample = track_point[:,0]
    y_sample = track_point[:,1]
    z_sample = track_point[:,2]

    tck, u = interpolate.splprep([x_sample, y_sample, z_sample], s=2)
    x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
    u_fine = np.linspace(0, 1, num_true_pts)
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)

    fig2 = plt.figure(2)
    ax3d = fig2.add_subplot(111, projection='3d')
    # ax3d.plot(x_true, y_true, z_true, 'b')
    # ax3d.plot(x_sample, y_sample, z_sample, 'r*')
    ax3d.plot(x_knots, y_knots, z_knots, 'go')
    ax3d.plot(x_fine, y_fine, z_fine, 'g')
    plt.xlim([0, 500])
    plt.ylim([0, 550])
    plt.zlim([0, 450])
    fig2.show()
    plt.show()
if __name__=="__main__":


    # init_bbox = gt[0]


    iou_load_savetest()