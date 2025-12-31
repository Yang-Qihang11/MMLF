import  numpy as np
import torch
import cyolo.src.config3d.kitti_config as cnf
import math
import  time
from cyolo.src.data_process import kitti_data_utils, kitti_bev_utils,transformation
def project_to_image(pts_3d, P):
    ''' Project 3d points to image plane.

    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix

      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)

      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    '''
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    # print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]

def roty(t):
    # Rotation about the y-axis.
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

def roty_plus(t):
    c = np.cos(t)
    s = np.sin(t)
    x = np.zeros((t.shape[0],3,3))
    x[:,1,1]=1
    x[:,0,0]=c.reshape(-1)
    x[:,2,2]=c.reshape(-1)
    x[:,0,2]=s.reshape(-1)
    x[:,2,0]=(-s).reshape(-1)

    return x


def compute_box_3d(objry,objl,objw,objh,objt,P):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    '''
    # compute rotational matrix around yaw axis


    R = roty_plus(objry)


    # 3d bounding box dimensions
    l = objl
    w = objw
    h = objh
    zeros = np.zeros((h.shape))
    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [zeros,zeros, zeros, zeros, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    # corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    stack = np.transpose(np.dstack([x_corners, y_corners, z_corners]), (1, 2, 0))
    # corners_3d = np.dot(R,stack )
    corners_3d = np.einsum('ijk,ikl->ijl', R, stack)
    # print corners_3d.shape
    corners_3d[:, 0, :] = corners_3d[:, 0, :] + objt[:,0].reshape(-1, 1)

    corners_3d[:, 1, :] = corners_3d[:, 1, :] + objt[:,1].reshape(-1, 1)
    corners_3d[:, 2, :] = corners_3d[:, 2, :] + objt[:,2].reshape(-1, 1)
    # print 'cornsers_3d: ', corners_3d
    # only draw 3d bounding box for objs in front of the camera

    # if np.any(corners_3d[:, 2, :] < 0.1): #todo 没做剔除
    #     corners_2d = None
    #     return corners_2d, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
    # corners_2d = project_to_image(np.transpose(corners_3d), P)
    # print 'corners_2d: ', corners_2d
    corners_3d = np.transpose((corners_3d),(0,2,1))
    return corners_3d


def compute_box2d(img_detections,calib,img_shape_2d,img_size):
    detections = img_detections
    predictions1 = []
    start_time2= time.time()
    x, y, w, l, im, re, _, _, cls_pred = detections.T
    x=x.reshape(-1, 1)
    y=y.reshape(-1, 1)
    w=w.reshape(-1, 1)
    l=l.reshape(-1, 1)
    im=im.reshape(-1, 1)
    re=re.reshape(-1, 1)
    cls_pred= cls_pred.reshape(-1, 1)
    x = x/ img_size
    y = y/ img_size
    w = w/ img_size
    l = l/ img_size
    predictions = np.concatenate((cls_pred,x,y,w,l,im,re), axis=1)

    predictions = kitti_bev_utils.inverse_yolo_target(predictions, cnf.boundary)
    if predictions.shape[0]:
        predictions[:, 1:] = transformation.lidar_to_camera_box(predictions[:, 1:], calib.V2C, calib.R0, calib.P)
    l = predictions[:,:]
    objt = l[:,1:4]
    objh = l[:,4:5]
    objw = l[:,5:6]
    objl = l[:,6:7]
    objry = np.arctan2(np.sin(l[:,7:8]), np.cos(l[:,7:8]))

    corners3d = compute_box_3d(objry,objl,objw,objh,objt, calib.P)

    img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)

    img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape_2d[1] - 1)
    img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape_2d[0] - 1)
    img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape_2d[1] - 1)
    img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape_2d[0] - 1)

    img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
    img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]



    return img_boxes
# #
# detections = torch.ones(22743,9)*20
# img_paths = '/DATA/yqh/KITTI_DATASET/KITTI/object/training/image_2/006000.png'
# calib = kitti_data_utils.Calibration(img_paths.replace(".png", ".txt").replace("image_2", "calib"))
# img_rgb_shape=(375, 1242, 3)
# configs_img_size=608
# box2d = compute_box2d(detections, calib, img_rgb_shape, configs_img_size)
# c=1