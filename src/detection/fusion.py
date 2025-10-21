"""Multi-sensor Fusion System

Implements decision-level fusion between camera and LiDAR detections
with various fusion strategies and confidence weighting.
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from loguru import logger

from .camera_detector import Detection as CameraDetection
from .lidar_detector import LiDARDetection


@dataclass
class FusedDetection:
    """Fused detection result combining camera and LiDAR information."""
    
    # 2D information (from camera)
    bbox_2d: List[float]  # [x1, y1, x2, y2]
    class_name: str
    class_id: int
    
    # 3D information (from LiDAR)
    center_3d: np.ndarray  # [x, y, z]
    bbox_3d: Optional[np.ndarray] = None  # [x, y, z, w, l, h, ry]
    
    # Fusion metadata
    confidence: float = 1.0
    camera_confidence: float = 0.0
    lidar_confidence: float = 0.0
    fusion_method: str = "unknown"
    
    # Source information
    has_camera: bool = False
    has_lidar: bool = False
    camera_detection: Optional[CameraDetection] = None
    lidar_detection: Optional[LiDARDetection] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'bbox_2d': self.bbox_2d,
            'class_name': self.class_name,
            'class_id': self.class_id,
            'center_3d': self.center_3d.tolist() if isinstance(self.center_3d, np.ndarray) else self.center_3d,
            'bbox_3d': self.bbox_3d.tolist() if self.bbox_3d is not None else None,
            'confidence': self.confidence,
            'camera_confidence': self.camera_confidence,
            'lidar_confidence': self.lidar_confidence,
            'fusion_method': self.fusion_method,
            'has_camera': self.has_camera,
            'has_lidar': self.has_lidar
        }


class FusionSystem:
    """Multi-sensor fusion system for camera and LiDAR detections."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize fusion system.
        
        Args:
            config: Fusion configuration
        """
        self.config = config
        self.fusion_method = config.get('method', 'weighted_average')
        
        # Fusion weights
        weights = config.get('weights', {})
        self.camera_weight = weights.get('camera', 0.6)
        self.lidar_weight = weights.get('lidar', 0.4)
        
        # Distance-based weighting
        distance_config = config.get('distance_weighting', {})
        self.use_distance_weighting = distance_config.get('enabled', True)
        self.near_threshold = distance_config.get('near_threshold', 10.0)
        self.far_threshold = distance_config.get('far_threshold', 40.0)
        self.near_camera_weight = distance_config.get('near_camera_weight', 0.7)
        self.far_camera_weight = distance_config.get('far_camera_weight', 0.3)
        
        # Matching parameters
        matching_config = config.get('matching', {})
        self.max_distance_2d = matching_config.get('max_distance_2d', 50.0)
        self.max_distance_3d = matching_config.get('max_distance_3d', 2.0)
        self.iou_threshold = matching_config.get('iou_threshold', 0.3)
        
        logger.info(f"Fusion system initialized with method: {self.fusion_method}")
    
    def fuse_detections(
        self, 
        camera_detections: List[CameraDetection],
        lidar_detections: List[LiDARDetection],
        calibration: Dict[str, np.ndarray]
    ) -> List[FusedDetection]:
        """Fuse camera and LiDAR detections.
        
        Args:
            camera_detections: List of camera detections
            lidar_detections: List of LiDAR detections
            calibration: Camera-LiDAR calibration matrices
            
        Returns:
            List of fused detections
        """
        # Project LiDAR detections to image space
        lidar_2d_projections = self._project_lidar_to_image(lidar_detections, calibration)
        
        # Match detections between sensors
        matches, unmatched_camera, unmatched_lidar = self._match_detections(
            camera_detections, lidar_detections, lidar_2d_projections
        )
        
        # Fuse matched detections
        fused_detections = []
        
        # Process matched pairs
        for cam_idx, lidar_idx, match_score in matches:
            fused_det = self._fuse_matched_pair(
                camera_detections[cam_idx],
                lidar_detections[lidar_idx],
                match_score
            )
            fused_detections.append(fused_det)
        
        # Add unmatched camera detections
        for cam_idx in unmatched_camera:
            cam_det = camera_detections[cam_idx]
            fused_det = self._create_camera_only_detection(cam_det)
            fused_detections.append(fused_det)
        
        # Add unmatched LiDAR detections
        for lidar_idx in unmatched_lidar:
            lidar_det = lidar_detections[lidar_idx]
            lidar_2d = lidar_2d_projections.get(lidar_idx)
            if lidar_2d is not None:  # Only include if projectable to image
                fused_det = self._create_lidar_only_detection(lidar_det, lidar_2d)
                fused_detections.append(fused_det)
        
        logger.debug(
            f"Fusion: {len(matches)} matched, {len(unmatched_camera)} camera-only, "
            f"{len(unmatched_lidar)} lidar-only, {len(fused_detections)} total"
        )
        
        return fused_detections
    
    def _project_lidar_to_image(
        self, 
        lidar_detections: List[LiDARDetection],
        calibration: Dict[str, np.ndarray]
    ) -> Dict[int, List[float]]:
        """Project LiDAR detections to image coordinates.
        
        Args:
            lidar_detections: List of LiDAR detections
            calibration: Calibration matrices
            
        Returns:
            Dictionary mapping detection index to 2D bbox [x1, y1, x2, y2]
        """
        projections = {}
        
        for i, detection in enumerate(lidar_detections):
            bbox_2d = detection.project_to_image(calibration)
            if bbox_2d is not None:
                projections[i] = bbox_2d
        
        return projections
    
    def _match_detections(
        self, 
        camera_detections: List[CameraDetection],
        lidar_detections: List[LiDARDetection],
        lidar_2d_projections: Dict[int, List[float]]
    ) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
        """Match camera and LiDAR detections.
        
        Args:
            camera_detections: Camera detections
            lidar_detections: LiDAR detections
            lidar_2d_projections: LiDAR projections to image space
            
        Returns:
            Tuple of (matches, unmatched_camera, unmatched_lidar)
            matches: List of (camera_idx, lidar_idx, match_score)
        """
        matches = []
        used_camera = set()
        used_lidar = set()
        
        # Calculate matching scores between all pairs
        match_scores = []
        
        for cam_idx, cam_det in enumerate(camera_detections):
            for lidar_idx, lidar_det in enumerate(lidar_detections):
                if lidar_idx not in lidar_2d_projections:
                    continue
                
                lidar_2d = lidar_2d_projections[lidar_idx]
                score = self._calculate_match_score(cam_det, lidar_det, lidar_2d)
                
                if score > 0:  # Valid match
                    match_scores.append((cam_idx, lidar_idx, score))
        
        # Sort by match score (descending)
        match_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Greedily assign matches
        for cam_idx, lidar_idx, score in match_scores:
            if cam_idx not in used_camera and lidar_idx not in used_lidar:
                matches.append((cam_idx, lidar_idx, score))
                used_camera.add(cam_idx)
                used_lidar.add(lidar_idx)
        
        # Find unmatched detections
        unmatched_camera = [i for i in range(len(camera_detections)) if i not in used_camera]
        unmatched_lidar = [i for i in range(len(lidar_detections)) if i not in used_lidar]
        
        return matches, unmatched_camera, unmatched_lidar
    
    def _calculate_match_score(
        self, 
        camera_det: CameraDetection,
        lidar_det: LiDARDetection,
        lidar_2d: List[float]
    ) -> float:
        """Calculate matching score between camera and LiDAR detection.
        
        Args:
            camera_det: Camera detection
            lidar_det: LiDAR detection
            lidar_2d: LiDAR detection projected to 2D
            
        Returns:
            Match score (0 = no match, 1 = perfect match)
        """
        # Calculate 2D IoU
        iou = self._calculate_iou_2d(camera_det.bbox, lidar_2d)
        
        if iou < self.iou_threshold:
            return 0.0  # No match
        
        # Calculate 2D distance between centers
        cam_center = camera_det.center
        lidar_center = [(lidar_2d[0] + lidar_2d[2]) / 2, (lidar_2d[1] + lidar_2d[3]) / 2]
        
        distance_2d = np.sqrt(
            (cam_center[0] - lidar_center[0])**2 + 
            (cam_center[1] - lidar_center[1])**2
        )
        
        if distance_2d > self.max_distance_2d:
            return 0.0  # Too far apart
        
        # Combine IoU and distance scores
        distance_score = max(0, 1 - distance_2d / self.max_distance_2d)
        
        # Class compatibility (simplified)
        class_score = 1.0  # Assume all classes are compatible for now
        
        # Combined score
        match_score = (iou * 0.6 + distance_score * 0.3 + class_score * 0.1)
        
        return match_score
    
    def _calculate_iou_2d(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate 2D IoU between bounding boxes.
        
        Args:
            bbox1: First bbox [x1, y1, x2, y2]
            bbox2: Second bbox [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        # Intersection
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Areas
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _fuse_matched_pair(
        self, 
        camera_det: CameraDetection,
        lidar_det: LiDARDetection,
        match_score: float
    ) -> FusedDetection:
        """Fuse a matched camera-LiDAR detection pair.
        
        Args:
            camera_det: Camera detection
            lidar_det: LiDAR detection
            match_score: Matching score
            
        Returns:
            Fused detection
        """
        # Calculate distance-based weights if enabled
        if self.use_distance_weighting:
            distance_3d = np.linalg.norm(lidar_det.center)
            camera_weight, lidar_weight = self._get_distance_weights(distance_3d)
        else:
            camera_weight = self.camera_weight
            lidar_weight = self.lidar_weight
        
        # Fuse confidence scores
        if self.fusion_method == 'weighted_average':
            fused_confidence = (
                camera_det.confidence * camera_weight +
                lidar_det.confidence * lidar_weight
            )
        elif self.fusion_method == 'max_confidence':
            fused_confidence = max(camera_det.confidence, lidar_det.confidence)
        elif self.fusion_method == 'consensus':
            # Require both sensors to agree (minimum confidence)
            fused_confidence = min(camera_det.confidence, lidar_det.confidence) * match_score
        else:
            fused_confidence = (camera_det.confidence + lidar_det.confidence) / 2
        
        # Use camera for 2D information and class
        bbox_2d = camera_det.bbox
        class_name = camera_det.class_name
        class_id = camera_det.class_id
        
        # Use LiDAR for 3D information
        center_3d = lidar_det.center
        bbox_3d = lidar_det.bbox_3d
        
        return FusedDetection(
            bbox_2d=bbox_2d,
            class_name=class_name,
            class_id=class_id,
            center_3d=center_3d,
            bbox_3d=bbox_3d,
            confidence=fused_confidence,
            camera_confidence=camera_det.confidence,
            lidar_confidence=lidar_det.confidence,
            fusion_method=self.fusion_method,
            has_camera=True,
            has_lidar=True,
            camera_detection=camera_det,
            lidar_detection=lidar_det
        )
    
    def _create_camera_only_detection(self, camera_det: CameraDetection) -> FusedDetection:
        """Create fused detection from camera-only detection.
        
        Args:
            camera_det: Camera detection
            
        Returns:
            Fused detection
        """
        # Estimate 3D center from 2D bbox (very rough approximation)
        center_2d = camera_det.center
        estimated_distance = 10.0  # Default distance assumption
        center_3d = np.array([estimated_distance, 0, -1.5])  # Rough estimate
        
        return FusedDetection(
            bbox_2d=camera_det.bbox,
            class_name=camera_det.class_name,
            class_id=camera_det.class_id,
            center_3d=center_3d,
            bbox_3d=None,
            confidence=camera_det.confidence * 0.8,  # Penalize single-sensor detection
            camera_confidence=camera_det.confidence,
            lidar_confidence=0.0,
            fusion_method="camera_only",
            has_camera=True,
            has_lidar=False,
            camera_detection=camera_det,
            lidar_detection=None
        )
    
    def _create_lidar_only_detection(
        self, 
        lidar_det: LiDARDetection,
        lidar_2d: List[float]
    ) -> FusedDetection:
        """Create fused detection from LiDAR-only detection.
        
        Args:
            lidar_det: LiDAR detection
            lidar_2d: 2D projection of LiDAR detection
            
        Returns:
            Fused detection
        """
        # Estimate class from 3D properties (simplified)
        bbox_3d = lidar_det.bbox_3d
        if len(bbox_3d) >= 6:
            length, width = bbox_3d[4], bbox_3d[3]
            if length > 6 or width > 2.5:
                class_name, class_id = "truck", 7
            elif length > 4:
                class_name, class_id = "car", 2
            else:
                class_name, class_id = "car", 2  # Default
        else:
            class_name, class_id = "car", 2
        
        return FusedDetection(
            bbox_2d=lidar_2d,
            class_name=class_name,
            class_id=class_id,
            center_3d=lidar_det.center,
            bbox_3d=lidar_det.bbox_3d,
            confidence=lidar_det.confidence * 0.7,  # Penalize single-sensor detection
            camera_confidence=0.0,
            lidar_confidence=lidar_det.confidence,
            fusion_method="lidar_only",
            has_camera=False,
            has_lidar=True,
            camera_detection=None,
            lidar_detection=lidar_det
        )
    
    def _get_distance_weights(self, distance_3d: float) -> Tuple[float, float]:
        """Get distance-based fusion weights.
        
        Args:
            distance_3d: 3D distance to object
            
        Returns:
            Tuple of (camera_weight, lidar_weight)
        """
        if distance_3d <= self.near_threshold:
            # Close objects: favor camera
            camera_weight = self.near_camera_weight
        elif distance_3d >= self.far_threshold:
            # Far objects: favor LiDAR
            camera_weight = self.far_camera_weight
        else:
            # Intermediate distance: interpolate
            alpha = (distance_3d - self.near_threshold) / (self.far_threshold - self.near_threshold)
            camera_weight = self.near_camera_weight * (1 - alpha) + self.far_camera_weight * alpha
        
        lidar_weight = 1.0 - camera_weight
        
        return camera_weight, lidar_weight
    
    def get_fusion_stats(self, fused_detections: List[FusedDetection]) -> Dict[str, Any]:
        """Get statistics about fusion results.
        
        Args:
            fused_detections: List of fused detections
            
        Returns:
            Statistics dictionary
        """
        if not fused_detections:
            return {
                'total_count': 0,
                'fusion_method_counts': {},
                'confidence_stats': {},
                'sensor_coverage': {}
            }
        
        # Count by fusion method
        method_counts = {}
        for det in fused_detections:
            method = det.fusion_method
            method_counts[method] = method_counts.get(method, 0) + 1
        
        # Confidence statistics
        confidences = [det.confidence for det in fused_detections]
        confidence_stats = {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences)
        }
        
        # Sensor coverage
        camera_count = sum(1 for det in fused_detections if det.has_camera)
        lidar_count = sum(1 for det in fused_detections if det.has_lidar)
        both_count = sum(1 for det in fused_detections if det.has_camera and det.has_lidar)
        
        sensor_coverage = {
            'camera_detections': camera_count,
            'lidar_detections': lidar_count,
            'fused_detections': both_count,
            'camera_only': camera_count - both_count,
            'lidar_only': lidar_count - both_count
        }
        
        return {
            'total_count': len(fused_detections),
            'fusion_method_counts': method_counts,
            'confidence_stats': confidence_stats,
            'sensor_coverage': sensor_coverage
        }