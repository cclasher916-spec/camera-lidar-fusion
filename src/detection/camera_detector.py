"""Camera-based Object Detection using YOLOv8

This module implements camera-only object detection using the YOLOv8 model
with robustness evaluation under different weather conditions.
"""

from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import cv2
from loguru import logger

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    logger.warning("Ultralytics YOLO not available, using fallback detection")
    YOLO_AVAILABLE = False


class Detection:
    """Single detection result."""
    
    def __init__(self, bbox: List[float], confidence: float, class_id: int, class_name: str):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        
        # Calculate additional properties
        self.center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        self.width = bbox[2] - bbox[0]
        self.height = bbox[3] - bbox[1]
        self.area = self.width * self.height
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary format."""
        return {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'center': self.center,
            'width': self.width,
            'height': self.height,
            'area': self.area
        }


class CameraDetector:
    """Camera-based object detector using YOLOv8."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize camera detector.
        
        Args:
            config: Detection configuration
        """
        self.config = config
        self.model_name = config.get('model', 'yolov8n.pt')
        self.confidence_threshold = config.get('confidence', 0.5)
        self.iou_threshold = config.get('iou_threshold', 0.7)
        self.device = config.get('device', 'cpu')
        self.target_classes = config.get('classes', [0, 1, 2, 3, 5, 7])  # Common vehicle classes
        
        # Initialize model
        self.model = self._load_model()
        
        # Class names mapping (COCO classes)
        self.class_names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
            4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
            8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
            # Add more as needed
        }
        
        logger.info(f"Camera detector initialized with model {self.model_name}")
    
    def _load_model(self):
        """Load YOLOv8 model or fallback detector."""
        if YOLO_AVAILABLE:
            try:
                model = YOLO(self.model_name)
                model.to(self.device)
                return model
            except Exception as e:
                logger.warning(f"Failed to load YOLO model: {e}, using fallback")
                return None
        else:
            return None
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Detect objects in camera image.
        
        Args:
            image: Input image (H, W, C) in RGB format
            
        Returns:
            List of Detection objects
        """
        if self.model is not None and YOLO_AVAILABLE:
            return self._yolo_detect(image)
        else:
            return self._fallback_detect(image)
    
    def _yolo_detect(self, image: np.ndarray) -> List[Detection]:
        """Run YOLOv8 detection."""
        try:
            # Run inference
            results = self.model(
                image, 
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=self.target_classes,
                verbose=False
            )
            
            detections = []
            
            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        # Extract box information
                        bbox = boxes.xyxy[i].cpu().numpy().tolist()
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        class_name = self.class_names.get(class_id, f'class_{class_id}')
                        
                        # Create detection object
                        detection = Detection(bbox, confidence, class_id, class_name)
                        detections.append(detection)
            
            logger.debug(f"YOLOv8 detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return self._fallback_detect(image)
    
    def _fallback_detect(self, image: np.ndarray) -> List[Detection]:
        """Fallback detection using simple computer vision methods."""
        detections = []
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Simple edge-based detection (as fallback)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area > 1000 and area < 50000:  # Reasonable vehicle size
                x, y, w, h = cv2.boundingRect(contour)
                
                # Simple heuristics for vehicle-like shapes
                aspect_ratio = w / h
                if 0.5 < aspect_ratio < 3.0:  # Vehicle-like aspect ratio
                    bbox = [x, y, x + w, y + h]
                    confidence = min(0.8, area / 10000)  # Confidence based on size
                    
                    detection = Detection(
                        bbox=bbox,
                        confidence=confidence,
                        class_id=2,  # Assume car
                        class_name='car'
                    )
                    detections.append(detection)
        
        logger.debug(f"Fallback detector found {len(detections)} objects")
        return detections
    
    def detect_batch(self, images: List[np.ndarray]) -> List[List[Detection]]:
        """Detect objects in a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            List of detection lists, one per image
        """
        results = []
        for image in images:
            detections = self.detect(image)
            results.append(detections)
        return results
    
    def filter_detections(
        self, 
        detections: List[Detection], 
        min_confidence: float = None,
        target_classes: List[int] = None
    ) -> List[Detection]:
        """Filter detections based on criteria.
        
        Args:
            detections: Input detections
            min_confidence: Minimum confidence threshold
            target_classes: List of target class IDs
            
        Returns:
            Filtered detections
        """
        filtered = detections
        
        # Filter by confidence
        if min_confidence is not None:
            filtered = [d for d in filtered if d.confidence >= min_confidence]
        
        # Filter by class
        if target_classes is not None:
            filtered = [d for d in filtered if d.class_id in target_classes]
        
        return filtered
    
    def non_max_suppression(
        self, 
        detections: List[Detection], 
        iou_threshold: float = None
    ) -> List[Detection]:
        """Apply Non-Maximum Suppression to detections.
        
        Args:
            detections: Input detections
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Filtered detections after NMS
        """
        if not detections:
            return detections
        
        iou_threshold = iou_threshold or self.iou_threshold
        
        # Sort by confidence (descending)
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        keep = []
        
        while detections:
            # Take the detection with highest confidence
            current = detections.pop(0)
            keep.append(current)
            
            # Remove detections with high IoU overlap
            remaining = []
            for det in detections:
                iou = self._calculate_iou(current.bbox, det.bbox)
                if iou < iou_threshold:
                    remaining.append(det)
            
            detections = remaining
        
        return keep
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU value between 0 and 1
        """
        # Calculate intersection
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def visualize_detections(
        self, 
        image: np.ndarray, 
        detections: List[Detection]
    ) -> np.ndarray:
        """Visualize detections on image.
        
        Args:
            image: Input image
            detections: List of detections to visualize
            
        Returns:
            Image with visualized detections
        """
        vis_image = image.copy()
        
        for detection in detections:
            bbox = [int(x) for x in detection.bbox]
            
            # Draw bounding box
            cv2.rectangle(
                vis_image, 
                (bbox[0], bbox[1]), 
                (bbox[2], bbox[3]), 
                (255, 0, 0),  # Red color
                2
            )
            
            # Draw label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            cv2.rectangle(
                vis_image,
                (bbox[0], bbox[1] - label_size[1] - 10),
                (bbox[0] + label_size[0], bbox[1]),
                (255, 0, 0),
                -1
            )
            
            cv2.putText(
                vis_image,
                label,
                (bbox[0], bbox[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return vis_image
    
    def get_detection_stats(self, detections: List[Detection]) -> Dict[str, Any]:
        """Get statistics about detections.
        
        Args:
            detections: List of detections
            
        Returns:
            Statistics dictionary
        """
        if not detections:
            return {
                'total_count': 0,
                'class_counts': {},
                'confidence_stats': {},
                'size_stats': {}
            }
        
        # Count by class
        class_counts = {}
        for det in detections:
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
        
        # Confidence statistics
        confidences = [det.confidence for det in detections]
        confidence_stats = {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences)
        }
        
        # Size statistics
        areas = [det.area for det in detections]
        size_stats = {
            'mean_area': np.mean(areas),
            'std_area': np.std(areas),
            'min_area': np.min(areas),
            'max_area': np.max(areas)
        }
        
        return {
            'total_count': len(detections),
            'class_counts': class_counts,
            'confidence_stats': confidence_stats,
            'size_stats': size_stats
        }