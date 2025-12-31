#!/usr/bin/env python3
"""
Main program for fusion of 2D and 3D object detection
Integrates YOLOv3 (2D) and Complex YOLOv4 (3D) detectors
"""

import argparse
import os
import sys
import warnings
import time
import platform
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from easydict import EasyDict as edict
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely")

# ==================== 2D Detection Imports ====================
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER, Profile, check_file, check_img_size, check_imshow,
    check_requirements, colorstr, cv2, increment_path, non_max_suppression,
    print_args, scale_boxes, strip_optimizer, xyxy2xywh
)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import (
    Albumentations, augment_hsv, classify_albumentations, classify_transforms,
    copy_paste, letterbox, mixup, random_perspective
)

# ==================== 3D Detection Imports ====================
sys.path.append('./')
import cyolo.src.config3d.kitti_config as cnf
from cyolo.src.data_process import kitti_data_utils, kitti_bev_utils
from cyolo.src.data_process.kitti_dataloader import (
    create_test_dataloader, create_val_dataloader, create_train_dataloader
)
from cyolo.src.models3d.model_utils import create_model
from cyolo.src.utils3d.misc import make_folder, time_synchronized, AverageMeter, ProgressMeter
from cyolo.src.utils3d.evaluation_utils import (
    post_processing, rescale_boxes, post_processing_v2, nms,
    get_batch_statistics_rotated_bbox, ap_per_class, load_classes
)
from cyolo.src.utils3d.visualization_utils import show_image_with_boxes, merge_rgb_to_bev, predictions_to_kitti_format

# ==================== Fusion Module Imports ====================
import fusion
from modelTMC import TMC
from final_loss import loss_compute
from get_corners import compute_box2d
from cyolo.src.utils3d.IOU import IOU, IOU2, IOU3

# ==================== Path Settings ====================
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv3 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# ==================== Configuration Classes ====================

class DetectionConfig2D:
    """2D detection configuration"""

    @staticmethod
    def parse():
        """Parse command line arguments for 2D detection"""
        parser = argparse.ArgumentParser(description='2D YOLOv3 Detection Configuration')
        parser.add_argument('--weights1', nargs='+', type=str,
                            default=ROOT / 'kitti.pt', help='model path or triton URL')
        parser.add_argument('--source', type=str, default=ROOT / 'data/images',
                            help='file/dir/URL/glob/screen/0(webcam)')
        parser.add_argument('--data', type=str, default=ROOT / 'data/kitti.yaml',
                            help='dataset configuration file path')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+',
                            type=int, default=[640], help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.01,
                            help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.99,
                            help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=1000,
                            help='maximum detections per image')
        parser.add_argument('--device', default='0',
                            help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true',
                            help='show results')
        parser.add_argument('--save-txt', action='store_true',
                            help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true',
                            help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true',
                            help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true',
                            help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int,
                            help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true',
                            help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true',
                            help='augmented inference')
        parser.add_argument('--visualize', action='store_true',
                            help='visualize features')
        parser.add_argument('--update', action='store_true',
                            help='update all models')
        parser.add_argument('--project', default=ROOT / 'runs/detect',
                            help='save results to project/name')
        parser.add_argument('--name', default='exp',
                            help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true',
                            help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int,
                            help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true',
                            help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true',
                            help='hide confidences')
        parser.add_argument('--half', action='store_true',
                            help='use FP16 half-precision inference')
        parser.add_argument('--dnn', action='store_true',
                            help='use OpenCV DNN for ONNX inference')
        parser.add_argument('--vid-stride', type=int, default=1,
                            help='video frame-rate stride')

        opt = parser.parse_args()
        opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # Expand
        print_args(vars(opt))
        return opt


class DetectionConfig3D:
    """3D detection configuration"""

    @staticmethod
    def parse_train():
        """Parse training configuration for 3D detection"""
        parser = argparse.ArgumentParser(description='Complex YOLOv4 Training Configuration')

        # Basic settings
        parser.add_argument('--seed', type=int, default=2020,
                            help='random seed for reproducibility')
        parser.add_argument('--saved_fn', type=str, default='complexer_yolo',
                            help='name for saving logs, models, etc.')
        parser.add_argument('--working-dir', type=str, default='../',
                            help='root working directory')

        # Model configuration
        parser.add_argument('-a', '--arch', type=str, default='darknet',
                            help='model architecture name')
        parser.add_argument('--cfgfile', type=str,
                            default='cyolo/src/config3d/cfg/complex_yolov4.cfg',
                            help='configuration file path (only for darknet)')
        parser.add_argument('--pretrained_path', type=str,
                            default='cyolo/checkpoints/complex_yolov4/complex_yolov4_mse_loss.pth',
                            help='pretrained checkpoint path')
        parser.add_argument('--use_giou_loss', action='store_true',
                            help='use GIoU loss instead of MSE loss')

        # DataLoader and training configuration
        parser.add_argument('--img_size', type=int, default=608,
                            help='input image size')
        parser.add_argument('--hflip_prob', type=float, default=0.5,
                            help='horizontal flip probability')
        parser.add_argument('--multiscale_training', action='store_true',
                            help='use scaling data for training')
        parser.add_argument('--mosaic', action='store_true',
                            help='use mosaic augmentation')
        parser.add_argument('--no-val', action='store_true',
                            help='do not evaluate on validation set')
        parser.add_argument('--num_samples', type=int, default=None,
                            help='subset of dataset for debugging')
        parser.add_argument('--num_workers', type=int, default=4,
                            help='number of data loading threads')
        parser.add_argument('--batch_size', type=int, default=1,
                            help='batch size for training')

        # Training strategy
        parser.add_argument('--start_epoch', type=int, default=1,
                            help='starting epoch')
        parser.add_argument('--num_epochs', type=int, default=300,
                            help='total number of epochs')
        parser.add_argument('--lr_type', type=str, default='cosin',
                            help='learning rate scheduler type')
        parser.add_argument('--lr', type=float, default=0.001,
                            help='initial learning rate')
        parser.add_argument('--momentum', type=float, default=0.949,
                            help='momentum')
        parser.add_argument('--weight_decay', type=float, default=5e-4,
                            help='weight decay')

        # Distributed training
        parser.add_argument('--gpu_idx', default=0, type=int,
                            help='GPU index to use')
        parser.add_argument('--no_cuda', action='store_true',
                            help='do not use CUDA')

        # Evaluation
        parser.add_argument('--evaluate', action='store_true',
                            help='only evaluate, do not train')
        parser.add_argument('--conf-thresh', type=float, default=0.5,
                            help='confidence threshold for evaluation')
        parser.add_argument('--nms-thresh', type=float, default=0.5,
                            help='NMS threshold for evaluation')
        parser.add_argument('--iou-thresh', type=float, default=0.5,
                            help='IoU threshold for evaluation')

        configs = edict(vars(parser.parse_args()))
        configs.device = torch.device('cpu' if configs.no_cuda else 'cuda')
        configs.pin_memory = True

        # Dataset paths
        configs.dataset_dir = '/media/lenovo/DATA1/yqh/KITTI_DATASET/KITTI/object'
        configs.checkpoints_dir = os.path.join(configs.working_dir, 'checkpoints', configs.saved_fn)
        configs.logs_dir = os.path.join(configs.working_dir, 'logs', configs.saved_fn)

        # Create directories
        os.makedirs(configs.checkpoints_dir, exist_ok=True)
        os.makedirs(configs.logs_dir, exist_ok=True)

        return configs

    @staticmethod
    def parse_eval():
        """Parse evaluation configuration for 3D detection"""
        parser = argparse.ArgumentParser(description='Complex YOLOv4 Evaluation Configuration')

        parser.add_argument('--classnames-infor-path', type=str,
                            default='/DATA/yqh/KITTI_DATASET/KITTI/object/classes_names.txt',
                            help='path to class names file')
        parser.add_argument('-a', '--arch', type=str, default='darknet',
                            help='model architecture name')
        parser.add_argument('--cfgfile', type=str,
                            default='cyolo/src/config3d/cfg/complex_yolov4.cfg',
                            help='configuration file path')
        parser.add_argument('--pretrained_path', type=str,
                            default='cyolo/checkpoints/complex_yolov4/complex_yolov4_mse_loss.pth',
                            help='pretrained checkpoint path')
        parser.add_argument('--use_giou_loss', action='store_true',
                            help='use GIoU loss')

        parser.add_argument('--no_cuda', action='store_true',
                            help='do not use CUDA')
        parser.add_argument('--gpu_idx', default=1, type=int,
                            help='GPU index to use')

        parser.add_argument('--img_size', type=int, default=608,
                            help='input image size')
        parser.add_argument('--num_samples', type=int, default=None,
                            help='subset of dataset')
        parser.add_argument('--num_workers', type=int, default=4,
                            help='number of data loading threads')
        parser.add_argument('--batch_size', type=int, default=4,
                            help='batch size')

        # Evaluation thresholds
        parser.add_argument('--conf-thresh', type=float, default=0.5,
                            help='confidence threshold')
        parser.add_argument('--nms-thresh', type=float, default=0.5,
                            help='NMS threshold')
        parser.add_argument('--iou-thresh', type=float, default=0.5,
                            help='IoU threshold')

        configs = edict(vars(parser.parse_args()))
        configs.pin_memory = True
        configs.dataset_dir = '/DATA/yqh/KITTI_DATASET/KITTI/object'

        return configs


class FusionTrainer:
    """Main trainer for fusing 2D and 3D detections"""

    def __init__(self, configs_2d, configs_3d):
        """
        Initialize the fusion trainer

        Args:
            configs_2d: 2D detection configuration
            configs_3d: 3D detection configuration
        """
        self.configs_2d = configs_2d
        self.configs_3d = configs_3d
        self.device = select_device(configs_2d.device)

        # Initialize models
        self._initialize_models()

        # Initialize fusion components
        self._initialize_fusion_components()

        # Training parameters
        self.epoch_a = 30  # Total training epochs
        self.epoch_losses = []
        self.epoch_metrics = {
            'precision': [],
            'recall': [],
            'ap': [],
            'f1': [],
            'ap_class': []
        }

    def _initialize_models(self):
        """Initialize 2D and 3D detection models"""
        # Initialize 3D model
        print("Initializing 3D detection model...")
        self.model3d = create_model(self.configs_3d)
        device_string = 'cpu' if self.configs_3d.no_cuda else f'cuda:{self.configs_3d.gpu_idx}'

        assert os.path.isfile(self.configs_3d.pretrained_path), \
            f"No file at {self.configs_3d.pretrained_path}"

        self.model3d.load_state_dict(
            torch.load(self.configs_3d.pretrained_path, map_location=device_string)
        )
        self.model3d.to(device=self.configs_3d.device)

        # Freeze 3D model parameters
        for param in self.model3d.parameters():
            param.requires_grad = False

        # Initialize 2D model
        print("Initializing 2D detection model...")
        self.model2d = DetectMultiBackend(
            weights=self.configs_2d.weights1,
            device=self.device,
            dnn=True,
            data='',
            fp16=False
        )

        # Freeze 2D model parameters
        for param in self.model2d.parameters():
            param.requires_grad = False

        print("Models initialized successfully.")

    def _initialize_fusion_components(self):
        """Initialize fusion layer and TMC model"""
        # Initialize CLOC fusion layer
        self.fusion_layer = fusion.fusion()
        self.fusion_layer.to(device=self.configs_3d.device)

        # Initialize TMC model
        self.modelTMC = TMC(3, 2, [[3], [3]], 50)  # cls views, dims, lambda_epochs
        self.modelTMC.cuda()

        # Initialize optimizers
        self.optimizer1 = optim.Adam(self.modelTMC.parameters(), lr=0.003, weight_decay=1e-5)
        self.optimizer2 = optim.SGD(self.fusion_layer.parameters(), lr=0.003)

        # Load pretrained weights if available
        self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        """Load pretrained weights for fusion components"""
        file_path1 = 'checkpoints/151fusion_layer_weights_epoch_0_map_0.9380.pth'
        file_path2 = 'checkpoints/151modelTMC_weights_epoch_0_map_0.9380.pth'

        if os.path.exists(file_path1):
            self.fusion_layer.load_state_dict(torch.load(file_path1))
            print(f"Loaded fusion layer weights from {file_path1}")

        if os.path.exists(file_path2):
            self.modelTMC.load_state_dict(torch.load(file_path2))
            print(f"Loaded TMC model weights from {file_path2}")

    def train_epoch(self, epoch, train_dataloader):
        """
        Train for one epoch

        Args:
            epoch: Current epoch number
            train_dataloader: Training data loader
        """
        self.modelTMC.train()
        self.fusion_layer.train()

        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{self.epoch_a}")

        for batch_idx, batch_data in enumerate(progress_bar):
            # Process batch
            loss = self._process_batch(batch_data, epoch)

            # Update weights
            self._update_weights(loss)

            # Track loss
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(progress_bar)
        self.epoch_losses.append(epoch_loss)

        print(f"Epoch {epoch + 1}/{self.epoch_a}, Loss: {epoch_loss:.4f}")

    def _process_batch(self, batch_data, epoch):
        """
        Process a single batch of data

        Args:
            batch_data: Batch of training data
            epoch: Current epoch number

        Returns:
            Calculated loss
        """
        img_paths, imgs, targets = batch_data

        # Move data to device
        targets = targets.to(self.configs_3d.device, non_blocking=True)
        imgs = imgs.to(self.configs_3d.device, non_blocking=True)

        # Get 3D detections
        _, outputs, targets_22743 = self.model3d(imgs, targets)

        # Process 3D predictions
        image_pred = outputs[0]
        class_confs, class_preds = image_pred[:, 7:].max(dim=1, keepdim=True)
        detections = torch.cat((image_pred[:, :7].float(), class_confs.float(), class_preds.float()), dim=1)
        cls_fea = image_pred[:, -3:].float()

        # Calculate distance to LiDAR
        dis_to_lidar = torch.norm(detections[:, :2], p=2, dim=1, keepdim=True) / 608.0

        # Load and process RGB image for 2D detection
        img_rgb = cv2.imread(img_paths[0])
        calib = kitti_data_utils.Calibration(
            img_paths[0].replace(".png", ".txt").replace("image_2", "calib")
        )

        # Compute 2D boxes from 3D detections
        box2d = compute_box2d(detections, calib, img_rgb.shape, self.configs_3d.img_size)
        d3_boxto_2d = torch.from_numpy(box2d)

        # Get 2D detections
        pred_2d = self._get_2d_detections(img_rgb)

        # Calculate IoU between 2D and 3D boxes
        iou_test, tensor_index, max_num, num_3dbox, num_2dbox, ratio_overlap = self._calculate_iou(
            d3_boxto_2d, pred_2d, detections, cls_fea, dis_to_lidar
        )

        # Get TMC features
        tmc_features = self._get_tmc_features(pred_2d, cls_fea, tensor_index, epoch)

        # Prepare fusion input
        fusion_input, ev_and_u, evidences_s = self._prepare_fusion_input(
            iou_test, tensor_index, tmc_features, num_3dbox, num_2dbox
        )

        # Perform fusion
        final_preds, _, no_match_indices, match_indices = self.fusion_layer(
            fusion_input.cuda(), tensor_index.cuda(),
            num_3dbox, num_2dbox, ev_and_u, evidences_s
        )

        # Calculate loss
        total_loss = loss_compute(
            final_preds, targets_22743, no_match_indices,
            match_indices, epoch, self.epoch_a
        )

        return total_loss

    def _get_2d_detections(self, img_rgb):
        """
        Get 2D detections from YOLOv3

        Args:
            img_rgb: RGB image

        Returns:
            2D predictions
        """
        # Preprocess image
        resized_image, ratio, _ = letterbox(img_rgb, (608, 608), stride=32, auto=True)
        resized_image = resized_image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        resized_image = np.ascontiguousarray(resized_image)
        resized_image = torch.from_numpy(resized_image).to(self.model2d.device)

        if self.model2d.fp16:
            resized_image = resized_image.half()
        else:
            resized_image = resized_image.float()
        resized_image /= 255.0

        if len(resized_image.shape) == 3:
            resized_image = resized_image[None]

        # Get predictions
        pred = self.model2d(resized_image, augment=self.configs_2d.augment,
                            visualize=self.configs_2d.visualize)

        # Add feature information
        self._add_feature_information(pred)

        # Apply NMS
        pred = non_max_suppression(
            pred, self.configs_2d.conf_thres, self.configs_2d.iou_thres,
            self.configs_2d.classes, self.configs_2d.agnostic_nms,
            max_det=self.configs_2d.max_det
        )

        # Apply sigmoid to last 3 features
        if len(pred[0]) > 0:
            pred[0][:, -3:] = torch.sigmoid(pred[0][:, -3:])

        return pred

    def _add_feature_information(self, pred):
        """Add feature information to predictions"""
        # 直接提取最后4个维度并拼接
        pre_list = [pred[1][i].reshape(1, -1, 8)[:, :, -4:] for i in range(3)]
        pre_list = torch.cat(pre_list, dim=1)

        # 合并到pred[0]
        pred[0] = torch.cat((pred[0], pre_list), dim=2)

    def _calculate_iou(self, d3_boxes, pred_2d, detections, cls_fea, dis_to_lidar):
        """
        Calculate IoU between 2D and 3D boxes

        Args:
            d3_boxes: 3D boxes projected to 2D
            pred_2d: 2D predictions
            detections: 3D detections
            cls_fea: 3D classification features
            dis_to_lidar: Distance to LiDAR

        Returns:
            IoU calculation results
        """
        # Prepare 2D boxes
        if len(pred_2d[0]) == 0:
            d2_box = np.ones((1, 4))
            d2_box[0, 2] = 1
            d2_box[0, 3] = 1
            pred_2d = [torch.zeros((1, 10))]
        else:
            batch_of_tensors = pred_2d[0]
            x1 = batch_of_tensors[:, 0].reshape(-1, 1)
            y1 = batch_of_tensors[:, 1].reshape(-1, 1)
            x2 = batch_of_tensors[:, 2].reshape(-1, 1)
            y2 = batch_of_tensors[:, 3].reshape(-1, 1)
            d2_box = torch.cat((x1, y1, x2, y2), dim=1).cpu().numpy()

        # Initialize arrays for IoU calculation
        overlaps1 = np.zeros((9000000, 4), dtype=d2_box.dtype)
        overlaps_ratio = np.zeros((9000000, 1), dtype=d2_box.dtype)
        tensor_index1 = np.zeros((9000000, 2), dtype=d2_box.dtype)
        overlaps1[:, :] = -1
        tensor_index1[:, :] = -1

        # Calculate IoU
        iou_test, tensor_index, max_num, num_3dbox, num_2dbox, ratio_overlap = IOU3(
            d3_boxes.detach().cpu().numpy(),
            d2_box,
            -1,
            detections[:, -3].detach().cpu().numpy().reshape(-1, 1),
            pred_2d[0][:, 4].detach().cpu().numpy().reshape(-1, 1),
            dis_to_lidar.detach().cpu().numpy(),
            overlaps1,
            tensor_index1,
            overlaps_ratio
        )

        return iou_test, tensor_index, max_num, num_3dbox, num_2dbox, ratio_overlap

    def _get_tmc_features(self, pred_2d, cls_fea, tensor_index, epoch):
        """
        Get TMC model features

        Args:
            pred_2d: 2D predictions
            cls_fea: 3D classification features
            tensor_index: Tensor indices
            epoch: Current epoch

        Returns:
            TMC features
        """
        if tensor_index.shape[0] == 0:
            return None

        # Extract 2D features
        fea2d = pred_2d[0][tensor_index[:, 0:1]]
        fea2d = fea2d[:, :, -3:]
        fea2d = fea2d.view(fea2d.shape[0], fea2d.shape[2])

        # Extract 3D features
        fea3d = cls_fea[tensor_index[:, 1:2]]
        fea3d = fea3d.view(fea3d.shape[0], fea3d.shape[2])

        # Prepare fake targets for TMC
        fake_targets = torch.randint(low=0, high=3, size=(fea3d.shape[0],))
        fake_targets = fake_targets.cuda()

        # Prepare feature dictionary
        fea = {0: fea2d.cuda(), 1: fea3d.cuda()}

        # Get TMC outputs
        evidences, evidence_a, loss_fake, u_a, u0, u1 = self.modelTMC(
            fea, fake_targets, epoch
        )

        return {
            'evidences': evidences,
            'evidence_a': evidence_a,
            'u_a': u_a,
            'u0': u0,
            'u1': u1
        }

    def _prepare_fusion_input(self, iou_test, tensor_index, tmc_features,
                              num_3dbox, num_2dbox):
        """
        Prepare input for fusion layer

        Args:
            iou_test: IoU test results
            tensor_index: Tensor indices
            tmc_features: TMC features
            num_3dbox: Number of 3D boxes
            num_2dbox: Number of 2D boxes

        Returns:
            Fusion input tensors
        """
        # Convert IoU test to tensor
        iou_test_tensor = torch.FloatTensor(iou_test)
        iou_test_tensor = iou_test_tensor.permute(1, 0)
        iou_test_tensor = iou_test_tensor.reshape(1, 4, 1, 9000000)

        # Convert tensor index to tensor
        tensor_index_tensor = torch.LongTensor(tensor_index)
        tensor_index_tensor = tensor_index_tensor.reshape(-1, 2)

        # Get non-empty IoU results
        if iou_test_tensor.shape[-1] == 0:
            non_empty_iou_test_tensor = torch.zeros(1, 4, 1, 2)
            non_empty_iou_test_tensor[:, :, :, :] = -1
            non_empty_tensor_index_tensor = torch.zeros(2, 2)
            non_empty_tensor_index_tensor[:, :] = -1
        else:
            non_empty_iou_test_tensor = iou_test_tensor[:, :, :, :iou_test_tensor.shape[-1]]
            non_empty_tensor_index_tensor = tensor_index_tensor[:iou_test_tensor.shape[-1], :]

        # Prepare TMC evidence and uncertainty
        evidence_a = tmc_features['evidence_a']
        u_a = tmc_features['u_a']

        # Add padding to evidence
        padding_tensor = torch.zeros(evidence_a.shape[0], 1).cuda()
        evidence_a = torch.cat((evidence_a, padding_tensor), dim=1)

        # Prepare evidence stack
        evidences_s = torch.cat((tmc_features['evidences'][0], tmc_features['evidences'][1]), dim=1)
        evidences_s = evidences_s.transpose(0, 1)
        evidences_s = evidences_s.unsqueeze(0).unsqueeze(2)

        # Prepare uncertainty tensor
        u_a1 = torch.transpose(u_a, 0, 1).cuda()
        u_a1 = u_a1.unsqueeze(0).cuda()
        u_a1 = u_a1.unsqueeze(0).cuda()

        # Combine IoU test with uncertainty
        fusion_input = torch.cat((non_empty_iou_test_tensor.cuda(), u_a1), dim=1).cuda()

        # Combine evidence and uncertainty
        ev_and_u = torch.cat((u_a, evidence_a), dim=1)
        ev_and_u = ev_and_u.transpose(0, 1)
        ev_and_u = ev_and_u.unsqueeze(0).unsqueeze(2)

        return fusion_input, ev_and_u, evidences_s

    def _update_weights(self, loss):
        """Update model weights based on loss"""
        # Zero gradients
        self.modelTMC.zero_grad()
        self.fusion_layer.zero_grad()

        # Backward pass
        loss.backward()

        # Update weights
        self.optimizer1.step()
        self.optimizer2.step()

    def validate(self, epoch, val_dataloader):
        """
        Validate the model

        Args:
            epoch: Current epoch number
            val_dataloader: Validation data loader

        Returns:
            Validation metrics
        """
        self.modelTMC.eval()
        self.model3d.eval()
        self.model2d.eval()
        self.fusion_layer.eval()

        labels = []
        sample_metrics = []
        class_names = ['Car', 'Pedestrian', 'Cyclist']

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(val_dataloader, desc="Validation")):
                img_paths, imgs, targets = batch_data

                # Move data to device
                targets = targets.to(self.configs_3d.device, non_blocking=True)
                imgs = imgs.to(self.configs_3d.device, non_blocking=True)

                # Get 3D detections
                _, outputs, _ = self.model3d(imgs, targets)

                # Process detections (similar to training)
                # ... (omitted for brevity, follows similar pattern as training)

                # Collect labels and metrics
                labels += targets[:, 1].tolist()

                # TODO: Implement full validation logic
                # This should include:
                # 1. Getting final predictions after fusion
                # 2. Applying NMS
                # 3. Calculating batch statistics

        # Calculate metrics
        if sample_metrics:
            true_positives, pred_scores, pred_labels = [
                np.concatenate(x, 0) for x in list(zip(*sample_metrics))
            ]

            precision, recall, AP, f1, ap_class = ap_per_class(
                true_positives, pred_scores, pred_labels, labels
            )

            # Store metrics
            self.epoch_metrics['precision'].append(precision)
            self.epoch_metrics['recall'].append(recall)
            self.epoch_metrics['ap'].append(AP)
            self.epoch_metrics['f1'].append(f1)
            self.epoch_metrics['ap_class'].append(ap_class)

            # Print metrics
            for idx, cls in enumerate(ap_class):
                print(f"\t>>>\t Class {cls} ({class_names[cls][:3]}): "
                      f"precision = {precision[idx]:.4f}, recall = {recall[idx]:.4f}, "
                      f"AP = {AP[idx]:.4f}, f1: {f1[idx]:.4f}")

            print(f"\nmAP: {AP.mean()}\n")

            # Save models
            self._save_models(epoch, AP.mean())

        return AP.mean() if sample_metrics else 0.0

    def _save_models(self, epoch, map_score):
        """Save model checkpoints"""
        torch.save(
            self.modelTMC.state_dict(),
            f'checkpoints/new_modelTMC_weights_epoch_{epoch}_map_{map_score:.4f}.pth'
        )

        torch.save(
            self.fusion_layer.state_dict(),
            f'checkpoints/new_fusion_layer_weights_epoch_{epoch}_map_{map_score:.4f}.pth'
        )

    def save_training_stats(self):
        """Save training statistics to files"""
        # Save epoch losses
        file_path = 'checkpoints/epoch_losses.txt'
        with open(file_path, 'w') as file:
            for item in self.epoch_losses:
                file.write(str(item) + '\n')

        # Save metrics
        file_path2 = 'checkpoints/training_metrics.txt'
        with open(file_path2, "w") as file:
            for i in range(len(self.epoch_losses)):
                if i < len(self.epoch_metrics['ap']):
                    line = (f"Epoch {i + 1}: Precision={self.epoch_metrics['precision'][i]}, "
                            f"Recall={self.epoch_metrics['recall'][i]}, "
                            f"AP={self.epoch_metrics['ap'][i]}, "
                            f"F1={self.epoch_metrics['f1'][i]}\n")
                    file.write(line)

    def run_training(self):
        """Main training loop"""
        print("Starting training...")

        # Create data loaders
        train_dataloader, train_sampler = create_train_dataloader(self.configs_3d)
        val_dataloader = create_val_dataloader(self.configs_3d)

        # Training loop
        for epoch in range(self.epoch_a):
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch + 1}/{self.epoch_a}")
            print(f"{'=' * 50}")

            # Train for one epoch
            self.train_epoch(epoch, train_dataloader)

            # Validate
            map_score = self.validate(epoch, val_dataloader)

            # Print epoch summary
            print(f"Epoch {epoch + 1} Summary:")
            print(f"  Loss: {self.epoch_losses[-1]:.4f}")
            print(f"  mAP: {map_score:.4f}")

        # Save final training statistics
        self.save_training_stats()
        print("\nTraining completed!")


def main():
    """Main function"""
    # Parse configurations
    configs_2d = DetectionConfig2D.parse()
    configs_3d = DetectionConfig3D.parse_train()
    configs_3d.distributed = False

    # Initialize trainer
    trainer = FusionTrainer(configs_2d, configs_3d)

    # Run training
    trainer.run_training()


if __name__ == '__main__':
    main()