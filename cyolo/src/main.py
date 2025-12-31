# todo 2d import
import argparse
import os
import platform
import sys
from pathlib import Path

import torch
# from modelTMC import TMC

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv3 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
import numpy as np
import torch.nn.functional as F

# todo 3d import
import argparse
import sys
import os
import time

from easydict import EasyDict as edict
import cv2
import torch
import numpy as np

sys.path.append('../')

import config.kitti_config as cnf
from data_process import kitti_data_utils, kitti_bev_utils
from data_process.kitti_dataloader import create_test_dataloader
from models3d.model_utils import create_model
from utils3d.misc import make_folder
from utils3d.evaluation_utils import post_processing, rescale_boxes, post_processing_v2
from utils3d.misc import time_synchronized
from utils3d.visualization_utils import show_image_with_boxes, merge_rgb_to_bev, predictions_to_kitti_format


def run(
        weights1=ROOT / 'kitti3.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    # todo 加载yolov3 2d模型
    device = select_device(device)
    model2d = DetectMultiBackend(weights=weights1, device=device, dnn='store_true', data='', fp16=False)
    stride, names, pt = model2d.stride, model2d.names, model2d.pt

    bs = 1
    dataset = LoadImages('./data/images', img_size=(640, 640), stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs
    for path, im, im0s, vid_cap, s in dataset:

        im = torch.from_numpy(im).to(model2d.device)
        im = im.half() if model2d.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.00 = {Tensor: (1, 18900, 85)} tensor([[[3.18626e+00, 4.33510e+00, 7.56535e+00,  ..., 2.28822e-03, 8.22380e-04, 1.32398e-03],\n         [1.07272e+01, 4.57727e+00, 2.16162e+01,  ..., 2.61278e-03, 9.71546e-04, 1.43484e-03],\n         [1.72457e+01, 3.52825e+00, 2.95080e+01,  ..., 1.93753e-03, 8.68013e-04, 1.67624e-03],\n         ...,\n         [3.99942e+02, 6.11914e+02, 1.61500e+02,  ..., 2.06167e-03, 1.25473e-03, 1.71470e-03],\n         [4.28277e+02, 6.08182e+02, 1.24622e+02,  ..., 2.15229e-03, 1.32601e-03, 1.70663e-03],\n         [4.57669e+02, 6.16798e+02, 1.31828e+02,  ..., 2.05022e-03, 1.36457e-03, 1.50254e-03]]], device='cuda:0')… View
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        pred = model2d(im, augment=augment, visualize=visualize)
        # 补充 特征图分类值
        pre1 = pred[1][0].reshape(1, -1, 8)
        pre2 = pred[1][1].reshape(1, -1, 8)
        pre3 = pred[1][2].reshape(1, -1, 8)
        pre1 = pre1[:, :, -4:]
        pre2 = pre2[:, :, -4:]
        pre3 = pre3[:, :, -4:]
        pre123 = torch.cat((pre1, pre2, pre3), dim=1)
        pred[0] = torch.cat((pred[0], pre123), dim=2)
        # todo 2D做NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        print(pred[0])
        pred[0][:, -3:] = torch.sigmoid(pred[0][:, -3:])

    # todo 3D C-yolov3
    configs = parse_test_configs()
    configs.distributed = False  # For testing

    model = create_model(configs)
    model.print_network()
    print('\n\n' + '-*=' * 30 + '\n\n')

    device_string = 'cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx)

    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location=device_string))

    configs.device = torch.device(device_string)
    model = model.to(device=configs.device)

    out_cap = None

    model.eval()

    test_dataloader = create_test_dataloader(configs)
    with torch.no_grad():
        for batch_idx, (img_paths, imgs_bev) in enumerate(test_dataloader):
            input_imgs = imgs_bev.to(device=configs.device).float()
            t1 = time_synchronized()
            outputs = model(input_imgs)
            t2 = time_synchronized()
            detections = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)

            img_detections = []  # Stores detections for each image index
            img_detections.extend(detections)

            img_bev = imgs_bev.squeeze() * 255
            img_bev = img_bev.permute(1, 2, 0).numpy().astype(np.uint8)
            img_bev = cv2.resize(img_bev, (configs.img_size, configs.img_size))
            for detections in img_detections:
                if detections is None:
                    continue
                # Rescale boxes to original image
                detections = rescale_boxes(detections, configs.img_size, img_bev.shape[:2])
                for x, y, w, l, im, re, *_, cls_pred in detections:
                    yaw = np.arctan2(im, re)
                    # Draw rotated box
                    kitti_bev_utils.drawRotatedBox(img_bev, x, y, w, l, yaw, cnf.colors[int(cls_pred)])

            img_rgb = cv2.imread(img_paths[0])
            calib = kitti_data_utils.Calibration(img_paths[0].replace(".png", ".txt").replace("image_2", "calib"))
            objects_pred = predictions_to_kitti_format(img_detections, calib, img_rgb.shape, configs.img_size)
            img_rgb = show_image_with_boxes(img_rgb, objects_pred, calib, False)

            img_bev = cv2.flip(cv2.flip(img_bev, 0), 1)

            out_img = merge_rgb_to_bev(img_rgb, img_bev, output_width=608)




def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights1',
                        nargs='+',
                        type=str,
                        default=ROOT / 'kitti3.pt',
                        help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
