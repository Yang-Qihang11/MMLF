#!/usr/bin/env python3
"""
2D-3D Object Detection Fusion Inference Module
Integrates YOLOv3 (2D) and Complex YOLOv4 (3D) for inference only
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import torch
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
from utils.augmentations import letterbox

# ==================== 3D Detection Imports ====================
sys.path.append('./')
import cyolo.src.config3d.kitti_config as cnf
from cyolo.src.data_process import kitti_data_utils, kitti_bev_utils
from cyolo.src.data_process.kitti_dataloader import create_test_dataloader
from cyolo.src.models3d.model_utils import create_model
from cyolo.src.utils3d.misc import make_folder
from cyolo.src.utils3d.evaluation_utils import rescale_boxes, nms
from cyolo.src.utils3d.visualization_utils import show_image_with_boxes, merge_rgb_to_bev, predictions_to_kitti_format

# ==================== Fusion Module Imports ====================
import fusion
from modelTMC import TMC
from get_corners import compute_box2d
from cyolo.src.utils3d.IOU import IOU3

# ==================== Path Settings ====================
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv3 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


# ==================== Configuration Classes ====================

class InferenceConfig2D:
    """2D detection inference configuration"""

    @staticmethod
    def parse():
        """Parse command line arguments for 2D inference"""
        parser = argparse.ArgumentParser(description='2D YOLOv3 Inference Configuration')
        parser.add_argument('--weights1', nargs='+', type=str,
                            default=ROOT / 'kitti.pt', help='2D model weights path')
        parser.add_argument('--source', type=str, default=ROOT / 'data/images',
                            help='input source: file/dir/URL/0 for webcam')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+',
                            type=int, default=[640], help='inference size (height, width)')
        parser.add_argument('--conf-thres', type=float, default=0.01,
                            help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.99,
                            help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=1000,
                            help='maximum detections per image')
        parser.add_argument('--device', default='0',
                            help='cuda device: 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true',
                            help='display results')
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
        parser.add_argument('--project', default=ROOT / 'runs/detect',
                            help='save results to project/name')
        parser.add_argument('--name', default='exp',
                            help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true',
                            help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int,
                            help='bounding box thickness (pixels)')
        parser.add_argument('--half', action='store_true',
                            help='use FP16 half-precision inference')
        parser.add_argument('--dnn', action='store_true',
                            help='use OpenCV DNN for ONNX inference')
        parser.add_argument('--vid-stride', type=int, default=1,
                            help='video frame-rate stride')

        opt = parser.parse_args()
        opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
        return opt


class InferenceConfig3D:
    """3D detection inference configuration"""

    @staticmethod
    def parse():
        """Parse command line arguments for 3D inference"""
        parser = argparse.ArgumentParser(description='Complex YOLOv4 Inference Configuration')

        parser.add_argument('--saved_fn', type=str, default='complexer_yolov4',
                            help='name for saving outputs')
        parser.add_argument('-a', '--arch', type=str, default='darknet',
                            help='model architecture name')
        parser.add_argument('--cfgfile', type=str,
                            default='cyolo/src/config3d/cfg/complex_yolov4.cfg',
                            help='model configuration file path')
        parser.add_argument('--pretrained_path', type=str,
                            default='cyolo/checkpoints/complex_yolov4/complex_yolov4_mse_loss.pth',
                            help='pretrained model path')

        parser.add_argument('--no_cuda', action='store_true',
                            help='do not use CUDA')
        parser.add_argument('--gpu_idx', default=0, type=int,
                            help='GPU index to use')

        parser.add_argument('--img_size', type=int, default=608,
                            help='input image size')
        parser.add_argument('--num_samples', type=int, default=None,
                            help='number of samples to process (for debugging)')
        parser.add_argument('--num_workers', type=int, default=1,
                            help='number of data loading workers')
        parser.add_argument('--batch_size', type=int, default=1,
                            help='batch size for inference')

        parser.add_argument('--conf_thresh', type=float, default=0.01,
                            help='confidence threshold')
        parser.add_argument('--nms_thresh', type=float, default=0.99,
                            help='NMS threshold')

        parser.add_argument('--show_image', action='store_true',
                            help='display images during inference')
        parser.add_argument('--save_output', action='store_true',
                            help='save inference outputs')
        parser.add_argument('--output_format', type=str, default='image',
                            help='output format: image or video')
        parser.add_argument('--output_video_fn', type=str, default='out_complexer_yolov4',
                            help='output video filename')

        configs = edict(vars(parser.parse_args()))
        configs.pin_memory = True

        # Dataset path
        configs.dataset_dir = '/DATA/yqh/KITTI_DATASET/KITTI/object'

        # Create results directory if saving outputs
        if configs.save_output:
            configs.results_dir = os.path.join('../', 'results', configs.saved_fn)
            make_folder(configs.results_dir)

        return configs


class FusionConfig:
    """Fusion model configuration"""

    @staticmethod
    def parse():
        """Parse command line arguments for fusion"""
        parser = argparse.ArgumentParser(description='Fusion Model Configuration')

        parser.add_argument('--fusion_weights', type=str,
                            default='checkpoints/fusion_layer_weights.pth',
                            help='fusion layer weights path')
        parser.add_argument('--tmc_weights', type=str,
                            default='checkpoints/modelTMC_weights.pth',
                            help='TMC model weights path')
        parser.add_argument('--fusion_threshold', type=float, default=0.5,
                            help='fusion confidence threshold')
        parser.add_argument('--save_fusion_results', action='store_true',
                            help='save fusion results')

        return parser.parse_args()


class DetectionResult:
    """Container for detection results"""

    def __init__(self):
        self.boxes_2d = None
        self.boxes_3d = None
        self.scores_2d = None
        self.scores_3d = None
        self.classes_2d = None
        self.classes_3d = None
        self.fused_boxes = None
        self.fused_scores = None
        self.fused_classes = None
        self.uncertainties = None
        self.timestamp = None
        self.image_path = None


class FusionInference:
    """Main class for 2D-3D fusion inference"""

    def __init__(self, configs_2d: InferenceConfig2D,
                 configs_3d: InferenceConfig3D,
                 configs_fusion: FusionConfig):
        """
        Initialize the fusion inference system

        Args:
            configs_2d: 2D detection configuration
            configs_3d: 3D detection configuration
            configs_fusion: Fusion configuration
        """
        self.configs_2d = configs_2d
        self.configs_3d = configs_3d
        self.configs_fusion = configs_fusion

        # Initialize devices
        self.device_2d = select_device(configs_2d.device)
        self.device_3d = torch.device('cpu' if configs_3d.no_cuda else f'cuda:{configs_3d.gpu_idx}')

        # Initialize models
        self._initialize_models()

        # Initialize fusion components
        self._initialize_fusion_components()

        # Results storage
        self.results = []

        print("Fusion inference system initialized successfully.")

    def _initialize_models(self):
        """Initialize 2D and 3D detection models"""
        print("Initializing detection models...")

        # Initialize 3D model
        print("  Loading 3D detection model...")
        self.model3d = create_model(self.configs_3d)
        device_string = 'cpu' if self.configs_3d.no_cuda else f'cuda:{self.configs_3d.gpu_idx}'

        if os.path.isfile(self.configs_3d.pretrained_path):
            self.model3d.load_state_dict(
                torch.load(self.configs_3d.pretrained_path, map_location=device_string)
            )
            self.model3d.to(device=self.device_3d)
            self.model3d.eval()
            print(f"  3D model loaded from {self.configs_3d.pretrained_path}")
        else:
            print(f"  Warning: 3D model weights not found at {self.configs_3d.pretrained_path}")

        # Initialize 2D model
        print("  Loading 2D detection model...")
        self.model2d = DetectMultiBackend(
            weights=self.configs_2d.weights1,
            device=self.device_2d,
            dnn=self.configs_2d.dnn,
            data='',
            fp16=self.configs_2d.half
        )
        self.model2d.eval()
        print(f"  2D model loaded from {self.configs_2d.weights1}")

    def _initialize_fusion_components(self):
        """Initialize fusion layer and TMC model"""
        print("Initializing fusion components...")

        # Initialize CLOC fusion layer
        self.fusion_layer = fusion.fusion()
        self.fusion_layer.to(device=self.device_3d)
        self.fusion_layer.eval()

        # Initialize TMC model
        self.modelTMC = TMC(3, 2, [[3], [3]], 50)
        self.modelTMC.cuda()
        self.modelTMC.eval()

        # Load weights if available
        if os.path.exists(self.configs_fusion.fusion_weights):
            self.fusion_layer.load_state_dict(torch.load(self.configs_fusion.fusion_weights))
            print(f"  Fusion layer loaded from {self.configs_fusion.fusion_weights}")

        if os.path.exists(self.configs_fusion.tmc_weights):
            self.modelTMC.load_state_dict(torch.load(self.configs_fusion.tmc_weights))
            print(f"  TMC model loaded from {self.configs_fusion.tmc_weights}")

    def process_image(self, image_path: str, bev_image: Optional[np.ndarray] = None) -> DetectionResult:
        """
        Process a single image through the fusion pipeline

        Args:
            image_path: Path to RGB image
            bev_image: Optional BEV image (if None, will be generated)

        Returns:
            DetectionResult object containing all detections
        """
        result = DetectionResult()
        result.image_path = image_path
        result.timestamp = time.time()

        try:
            # Load and preprocess RGB image
            img_rgb = cv2.imread(image_path)
            if img_rgb is None:
                raise ValueError(f"Could not load image: {image_path}")

            # Step 1: Get 3D detections
            detections_3d, cls_fea_3d = self._get_3d_detections(image_path, bev_image)

            # Step 2: Get 2D detections
            detections_2d, features_2d = self._get_2d_detections(img_rgb)

            # Step 3: Project 3D boxes to 2D
            calib = kitti_data_utils.Calibration(
                image_path.replace(".png", ".txt").replace("image_2", "calib")
            )
            boxes_3d_projected = compute_box2d(
                detections_3d, calib, img_rgb.shape, self.configs_3d.img_size
            )

            # Step 4: Calculate IoU and match boxes
            matched_pairs, unmatched_3d, unmatched_2d = self._match_boxes(
                boxes_3d_projected, detections_2d, detections_3d, features_2d, cls_fea_3d
            )

            # Step 5: Perform fusion
            fused_detections = self._fuse_detections(
                matched_pairs, unmatched_3d, detections_3d, cls_fea_3d
            )

            # Step 6: Apply NMS to fused results
            final_detections = self._apply_nms(fused_detections)

            # Store results
            result.boxes_3d = detections_3d[:, :7] if detections_3d is not None else None
            result.scores_3d = detections_3d[:, 6] if detections_3d is not None else None
            result.classes_3d = detections_3d[:, 7:].argmax(dim=1) if detections_3d is not None else None

            if detections_2d is not None:
                result.boxes_2d = detections_2d[:, :4]
                result.scores_2d = detections_2d[:, 4]
                result.classes_2d = detections_2d[:, 5]

            result.fused_boxes = final_detections[:, :7] if final_detections is not None else None
            result.fused_scores = final_detections[:, 6] if final_detections is not None else None
            result.fused_classes = final_detections[:, 7:].argmax(dim=1) if final_detections is not None else None

            self.results.append(result)
            print(
                f"Processed {image_path}: {len(final_detections) if final_detections is not None else 0} fused detections")

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            import traceback
            traceback.print_exc()

        return result

    def _get_3d_detections(self, image_path: str, bev_image: Optional[np.ndarray] = None) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Get 3D detections from Complex YOLOv4

        Args:
            image_path: Path to the image
            bev_image: Optional BEV image

        Returns:
            Tuple of (detections, classification features)
        """
        # Load BEV image if not provided
        if bev_image is None:
            # For inference, you would load or generate BEV here
            # This is a placeholder - adjust based on your data pipeline
            bev_image = np.zeros((608, 608, 3), dtype=np.float32)

        # Convert to tensor and preprocess
        bev_tensor = torch.from_numpy(bev_image).permute(2, 0, 1).unsqueeze(0).float()
        bev_tensor = bev_tensor.to(self.device_3d)

        # Inference
        with torch.no_grad():
            outputs = self.model3d(bev_tensor)

        # Process outputs
        if isinstance(outputs, tuple):
            detections = outputs[0]
        else:
            detections = outputs

        # Extract classification features
        cls_fea = detections[:, -3:] if detections.shape[1] > 7 else None

        return detections, cls_fea

    def _get_2d_detections(self, img_rgb: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get 2D detections from YOLOv3

        Args:
            img_rgb: RGB image array

        Returns:
            Tuple of (detections, features)
        """
        # Preprocess image
        resized_image, ratio, (dw, dh) = letterbox(img_rgb, (640, 640), stride=32, auto=True)
        resized_image = resized_image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        resized_image = np.ascontiguousarray(resized_image)

        # Convert to tensor
        img_tensor = torch.from_numpy(resized_image).to(self.device_2d)
        if self.configs_2d.half:
            img_tensor = img_tensor.half()
        else:
            img_tensor = img_tensor.float()
        img_tensor /= 255.0

        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # Inference
        with torch.no_grad():
            pred = self.model2d(img_tensor, augment=self.configs_2d.augment)

        # Add feature information
        self._add_feature_information(pred)

        # Apply NMS
        pred = non_max_suppression(
            pred, self.configs_2d.conf_thres, self.configs_2d.iou_thres,
            self.configs_2d.classes, self.configs_2d.agnostic_nms,
            max_det=self.configs_2d.max_det
        )

        # Process predictions
        if pred[0] is not None and len(pred[0]) > 0:
            # Apply sigmoid to last 3 features
            pred[0][:, -3:] = torch.sigmoid(pred[0][:, -3:])

            # Adjust boxes to original image coordinates
            pred[0][:, :4] = scale_boxes(img_tensor.shape[2:], pred[0][:, :4], img_rgb.shape).round()

            # Extract features
            features = pred[0][:, -3:] if pred[0].shape[1] > 6 else None

            return pred[0], features

        return None, None

    def _add_feature_information(self, pred):
        """Add feature information to 2D predictions"""
        if len(pred) > 1 and isinstance(pred[1], list):
            pre1 = pred[1][0].reshape(1, -1, 8)
            pre2 = pred[1][1].reshape(1, -1, 8)
            pre3 = pred[1][2].reshape(1, -1, 8)

            pre1 = pre1[:, :, -4:]
            pre2 = pre2[:, :, -4:]
            pre3 = pre3[:, :, -4:]

            pre123 = torch.cat((pre1, pre2, pre3), dim=1)
            pred[0] = torch.cat((pred[0], pre123), dim=2)

    def _match_boxes(self, boxes_3d_projected: np.ndarray, detections_2d: torch.Tensor,
                     detections_3d: torch.Tensor, features_2d: torch.Tensor,
                     cls_fea_3d: torch.Tensor) -> Tuple[List, List, List]:
        """
        Match 2D and 3D boxes using IoU

        Args:
            boxes_3d_projected: 3D boxes projected to 2D
            detections_2d: 2D detections
            detections_3d: 3D detections
            features_2d: 2D features
            cls_fea_3d: 3D classification features

        Returns:
            Tuple of (matched_pairs, unmatched_3d_indices, unmatched_2d_indices)
        """
        # Handle case where no 2D detections
        if detections_2d is None or len(detections_2d) == 0:
            # No matches, all 3D boxes are unmatched
            unmatched_3d = list(range(len(detections_3d))) if detections_3d is not None else []
            return [], unmatched_3d, []

        # Prepare 2D boxes
        d2_boxes = detections_2d[:, :4].cpu().numpy()

        # Initialize arrays for IoU calculation
        max_pairs = 9000000  # Adjust based on expected maximum
        overlaps = np.zeros((max_pairs, 4), dtype=d2_boxes.dtype)
        overlaps_ratio = np.zeros((max_pairs, 1), dtype=d2_boxes.dtype)
        tensor_index = np.zeros((max_pairs, 2), dtype=np.int64)

        overlaps[:, :] = -1
        tensor_index[:, :] = -1

        # Calculate distance to LiDAR for 3D boxes
        dis_to_lidar = torch.norm(detections_3d[:, :2], p=2, dim=1, keepdim=True) / 608.0

        # Calculate IoU
        iou_test, tensor_index, max_num, num_3dbox, num_2dbox, ratio_overlap = IOU3(
            boxes_3d_projected,
            d2_boxes,
            -1,
            detections_3d[:, 6:7].detach().cpu().numpy(),
            detections_2d[:, 4:5].detach().cpu().numpy(),
            dis_to_lidar.detach().cpu().numpy(),
            overlaps,
            tensor_index,
            overlaps_ratio
        )

        # Process matches
        matched_pairs = []
        if max_num > 0:
            for i in range(max_num):
                if i < tensor_index.shape[0] and tensor_index[i, 0] >= 0 and tensor_index[i, 1] >= 0:
                    matched_pairs.append({
                        '2d_idx': int(tensor_index[i, 0]),
                        '3d_idx': int(tensor_index[i, 1]),
                        'iou': float(iou_test[i, 0]),
                        'ratio': float(ratio_overlap[i, 0]) if ratio_overlap[i, 0] != -1 else 0.0
                    })

        # Find unmatched boxes
        all_3d_indices = set(range(len(detections_3d))) if detections_3d is not None else set()
        all_2d_indices = set(range(len(detections_2d)))

        matched_3d_indices = {pair['3d_idx'] for pair in matched_pairs}
        matched_2d_indices = {pair['2d_idx'] for pair in matched_pairs}

        unmatched_3d = list(all_3d_indices - matched_3d_indices)
        unmatched_2d = list(all_2d_indices - matched_2d_indices)

        return matched_pairs, unmatched_3d, unmatched_2d

    def _fuse_detections(self, matched_pairs: List[Dict], unmatched_3d: List[int],
                         detections_3d: torch.Tensor, cls_fea_3d: torch.Tensor) -> torch.Tensor:
        """
        Fuse matched 2D-3D detections

        Args:
            matched_pairs: List of matched 2D-3D pairs
            unmatched_3d: List of unmatched 3D detection indices
            detections_3d: 3D detections
            cls_fea_3d: 3D classification features

        Returns:
            Fused detections
        """
        if detections_3d is None:
            return None

        # Prepare tensors for fusion
        num_detections = len(detections_3d)
        fused_features = torch.zeros((num_detections, 5), device=self.device_3d)  # obj_score + cls_fea + uncertainty

        # Process matched pairs
        if matched_pairs:
            # Prepare matched features for TMC
            matched_2d_features = []
            matched_3d_features = []
            matched_indices = []

            for pair in matched_pairs:
                # Note: You'll need to access 2D features here
                # This depends on how features are stored in your pipeline
                idx_2d = pair['2d_idx']
                idx_3d = pair['3d_idx']

                # Placeholder - replace with actual feature extraction
                fea_2d = torch.randn(3, device=self.device_3d)  # Replace with actual 2D features
                fea_3d = cls_fea_3d[idx_3d] if cls_fea_3d is not None else torch.randn(3, device=self.device_3d)

                matched_2d_features.append(fea_2d)
                matched_3d_features.append(fea_3d)
                matched_indices.append(idx_3d)

            if matched_2d_features and matched_3d_features:
                # Convert to tensors
                fea_2d_tensor = torch.stack(matched_2d_features)
                fea_3d_tensor = torch.stack(matched_3d_features)

                # Prepare TMC input
                fea_dict = {0: fea_2d_tensor, 1: fea_3d_tensor}
                fake_targets = torch.randint(0, 3, (len(matched_indices),), device=self.device_3d)

                # Get TMC outputs
                with torch.no_grad():
                    evidences, evidence_a, _, u_a, _, _ = self.modelTMC(
                        fea_dict, fake_targets, epoch=1
                    )

                # Store fused features for matched detections
                for i, idx in enumerate(matched_indices):
                    fused_features[idx, 0] = evidence_a[i, 3]  # Use last feature as objectness score
                    fused_features[idx, 1:4] = evidence_a[i, :3]  # Classification features
                    fused_features[idx, 4] = u_a[i, 0]  # Uncertainty

        # Process unmatched 3D detections
        for idx in unmatched_3d:
            # For unmatched detections, use only 3D features
            if cls_fea_3d is not None and idx < len(cls_fea_3d):
                fused_features[idx, 0] = 0.5  # Default objectness score
                fused_features[idx, 1:4] = cls_fea_3d[idx]
                fused_features[idx, 4] = 1.0  # High uncertainty (no 2D match)

        # Combine with original 3D detections
        fused_detections = torch.cat([
            detections_3d[:, :6],  # 3D box parameters
            fused_features[:, 0:1],  # Fused objectness score
            fused_features[:, 1:4],  # Fused classification features
            fused_features[:, 4:5]  # Uncertainty
        ], dim=1)

        return fused_detections

    def _apply_nms(self, detections: torch.Tensor) -> torch.Tensor:
        """
        Apply NMS to fused detections

        Args:
            detections: Fused detections

        Returns:
            Detections after NMS
        """
        if detections is None or len(detections) == 0:
            return None

        # Apply confidence threshold
        conf_mask = detections[:, 6] > self.configs_fusion.fusion_threshold
        filtered_detections = detections[conf_mask]

        if len(filtered_detections) == 0:
            return None

        # Apply NMS
        with torch.no_grad():
            nms_detections = nms(filtered_detections,
                                 conf_thresh=self.configs_3d.conf_thresh,
                                 nms_thresh=self.configs_3d.nms_thresh)

        return nms_detections

    def visualize_results(self, result: DetectionResult, save_path: Optional[str] = None):
        """
        Visualize fusion results

        Args:
            result: DetectionResult to visualize
            save_path: Optional path to save visualization
        """
        if result.image_path is None:
            print("No image path in result")
            return

        # Load original image
        img = cv2.imread(result.image_path)
        if img is None:
            print(f"Could not load image: {result.image_path}")
            return

        # Create visualization
        vis_img = img.copy()

        # Draw 2D detections (green)
        if result.boxes_2d is not None:
            for box, score, cls_id in zip(result.boxes_2d, result.scores_2d, result.classes_2d):
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"2D: {cls_id}: {score:.2f}"
                cv2.putText(vis_img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw fused detections (red)
        if result.fused_boxes is not None:
            for box, score, cls_id in zip(result.fused_boxes, result.fused_scores, result.fused_classes):
                # Note: 3D boxes need projection to 2D for visualization
                # This is simplified - you'll need to project 3D boxes to 2D
                if len(box) >= 7:  # 3D box
                    # Placeholder: draw at center with fixed size
                    center_x, center_y = 320, 240  # Center of image
                    size = 50
                    x1, y1 = int(center_x - size / 2), int(center_y - size / 2)
                    x2, y2 = int(center_x + size / 2), int(center_y + size / 2)

                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    label = f"Fused: {cls_id}: {score:.2f}"
                    cv2.putText(vis_img, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display or save
        if self.configs_2d.view_img:
            cv2.imshow("Fusion Results", vis_img)
            cv2.waitKey(0)

        if save_path:
            cv2.imwrite(save_path, vis_img)
            print(f"Visualization saved to {save_path}")

    def save_results(self, result: DetectionResult, output_dir: str):
        """
        Save detection results to files

        Args:
            result: DetectionResult to save
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save as text file
        if result.image_path:
            base_name = os.path.splitext(os.path.basename(result.image_path))[0]
            txt_path = os.path.join(output_dir, f"{base_name}.txt")

            with open(txt_path, 'w') as f:
                f.write(f"Image: {result.image_path}\n")
                f.write(f"Timestamp: {result.timestamp}\n\n")

                # Save 2D detections
                if result.boxes_2d is not None:
                    f.write("2D Detections:\n")
                    for i, (box, score, cls_id) in enumerate(zip(result.boxes_2d,
                                                                 result.scores_2d,
                                                                 result.classes_2d)):
                        f.write(f"  Detection {i}: Class={cls_id}, Score={score:.4f}, "
                                f"Box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]\n")

                # Save fused detections
                if result.fused_boxes is not None:
                    f.write("\nFused Detections:\n")
                    for i, (box, score, cls_id) in enumerate(zip(result.fused_boxes,
                                                                 result.fused_scores,
                                                                 result.fused_classes)):
                        f.write(f"  Detection {i}: Class={cls_id}, Score={score:.4f}, "
                                f"Box 3D=[{box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}, "
                                f"{box[3]:.2f}, {box[4]:.2f}, {box[5]:.2f}, {box[6]:.2f}]\n")

            print(f"Results saved to {txt_path}")

    def process_dataset(self, dataset_path: str, output_dir: str):
        """
        Process an entire dataset

        Args:
            dataset_path: Path to dataset directory
            output_dir: Directory to save outputs
        """
        # Get list of images
        if os.path.isdir(dataset_path):
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(Path(dataset_path).glob(f'**/*{ext}'))
            image_files = [str(f) for f in image_files]
        else:
            # Single image
            image_files = [dataset_path]

        print(f"Found {len(image_files)} images to process")

        # Process each image
        for i, img_path in enumerate(tqdm(image_files, desc="Processing images")):
            print(f"\nProcessing image {i + 1}/{len(image_files)}: {img_path}")

            # Process image
            result = self.process_image(img_path)

            # Save results
            if self.configs_fusion.save_fusion_results:
                self.save_results(result, output_dir)

            # Visualize if requested
            if self.configs_2d.view_img or self.configs_3d.show_image:
                vis_path = os.path.join(output_dir, f"vis_{os.path.basename(img_path)}")
                self.visualize_results(result, vis_path)

        print(f"\nProcessing complete. Processed {len(self.results)} images.")
        return self.results

    def benchmark(self, dataset_path: str, num_iterations: int = 100):
        """
        Benchmark inference speed

        Args:
            dataset_path: Path to test image
            num_iterations: Number of iterations for benchmarking
        """
        print(f"\n{'=' * 50}")
        print("Running benchmark...")
        print(f"{'=' * 50}")

        # Load a test image
        if os.path.isdir(dataset_path):
            image_files = [f for f in Path(dataset_path).glob('*.png')][:1]
            test_image = str(image_files[0]) if image_files else None
        else:
            test_image = dataset_path

        if not test_image or not os.path.exists(test_image):
            print("No test image found for benchmarking")
            return

        # Warm-up
        print("Warm-up...")
        for _ in range(10):
            _ = self.process_image(test_image)

        # Benchmark
        print(f"Running {num_iterations} iterations...")
        import time

        times = []
        for i in range(num_iterations):
            start_time = time.time()
            result = self.process_image(test_image)
            end_time = time.time()
            times.append(end_time - start_time)

            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{num_iterations} iterations")

        # Calculate statistics
        times = np.array(times)
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1.0 / avg_time

        print(f"\nBenchmark Results:")
        print(f"  Average inference time: {avg_time * 1000:.2f} ms")
        print(f"  Standard deviation: {std_time * 1000:.2f} ms")
        print(f"  FPS: {fps:.2f}")
        print(f"  Min time: {np.min(times) * 1000:.2f} ms")
        print(f"  Max time: {np.max(times) * 1000:.2f} ms")

        return {
            'avg_time_ms': avg_time * 1000,
            'std_time_ms': std_time * 1000,
            'fps': fps,
            'min_time_ms': np.min(times) * 1000,
            'max_time_ms': np.max(times) * 1000
        }


def main():
    """Main inference function"""
    # Parse configurations
    configs_2d = InferenceConfig2D.parse()
    configs_3d = InferenceConfig3D.parse()
    configs_fusion = FusionConfig.parse()

    # Initialize fusion inference system
    fusion_system = FusionInference(configs_2d, configs_3d, configs_fusion)

    # Process input
    if configs_2d.source:
        # Create output directory
        output_dir = os.path.join(configs_2d.project, configs_2d.name)
        os.makedirs(output_dir, exist_ok=True)

        # Process dataset
        results = fusion_system.process_dataset(configs_2d.source, output_dir)

        # Print summary
        print(f"\n{'=' * 50}")
        print("Inference Summary:")
        print(f"{'=' * 50}")
        print(f"Total images processed: {len(results)}")

        total_2d = sum(len(r.boxes_2d) if r.boxes_2d is not None else 0 for r in results)
        total_3d = sum(len(r.boxes_3d) if r.boxes_3d is not None else 0 for r in results)
        total_fused = sum(len(r.fused_boxes) if r.fused_boxes is not None else 0 for r in results)

        print(f"Total 2D detections: {total_2d}")
        print(f"Total 3D detections: {total_3d}")
        print(f"Total fused detections: {total_fused}")
        print(f"Results saved to: {output_dir}")

    # Run benchmark if requested
    if configs_2d.source and len(fusion_system.results) > 0:
        fusion_system.benchmark(configs_2d.source)


if __name__ == '__main__':
    main()