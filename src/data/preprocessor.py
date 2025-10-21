"""Data Preprocessing Utilities

Handles preprocessing of camera images and LiDAR point clouds
including normalization, filtering, and coordinate transformations.
"""

from typing import Tuple, Dict, Any, Optional

import numpy as np
import cv2
from loguru import logger


class DataPreprocessor:
    """Preprocessor for camera and LiDAR data."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize preprocessor with configuration.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.image_size = tuple(config.get('image_size', (640, 480)))
        self.lidar_range = config.get('lidar_range', {'x': [-50, 50], 'y': [-25, 25], 'z': [-3, 5]})
        self.normalize_images = config.get('normalize_images', True)
        
        logger.info("Data preprocessor initialized")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess camera image.
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            Preprocessed image
        """
        # Resize image
        if image.shape[:2] != self.image_size[::-1]:  # OpenCV uses (W, H)
            image = cv2.resize(image, self.image_size)
        
        # Normalize if requested
        if self.normalize_images:
            image = self._normalize_image(image)
        
        return image
    
    def preprocess_lidar(self, points: np.ndarray) -> np.ndarray:
        """Preprocess LiDAR point cloud.
        
        Args:
            points: Input points (N, 4) [x, y, z, intensity]
            
        Returns:
            Filtered and processed points
        """
        # Filter by range
        filtered_points = self._filter_points_by_range(points)
        
        # Remove ground plane (simple height filter)
        filtered_points = self._remove_ground_points(filtered_points)
        
        # Normalize intensity
        filtered_points = self._normalize_intensity(filtered_points)
        
        return filtered_points
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        return image.astype(np.float32) / 255.0
    
    def _filter_points_by_range(self, points: np.ndarray) -> np.ndarray:
        """Filter points by spatial range."""
        x_min, x_max = self.lidar_range['x']
        y_min, y_max = self.lidar_range['y']
        z_min, z_max = self.lidar_range['z']
        
        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        )
        
        return points[mask]
    
    def _remove_ground_points(self, points: np.ndarray, ground_threshold: float = -1.4) -> np.ndarray:
        """Remove ground plane points based on height."""
        mask = points[:, 2] > ground_threshold
        return points[mask]
    
    def _normalize_intensity(self, points: np.ndarray) -> np.ndarray:
        """Normalize intensity values to [0, 1] range."""
        if points.shape[1] >= 4:
            intensity = points[:, 3]
            if intensity.max() > 1.0:
                points[:, 3] = intensity / 255.0
        return points
    
    def augment_image(self, image: np.ndarray, augmentation_type: str = "none") -> np.ndarray:
        """Apply image augmentation.
        
        Args:
            image: Input image
            augmentation_type: Type of augmentation to apply
            
        Returns:
            Augmented image
        """
        if augmentation_type == "brightness":
            return self._adjust_brightness(image, factor=0.8)
        elif augmentation_type == "contrast":
            return self._adjust_contrast(image, factor=1.2)
        elif augmentation_type == "noise":
            return self._add_noise(image, std=0.02)
        else:
            return image
    
    def _adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image brightness."""
        return np.clip(image * factor, 0, 255 if image.dtype == np.uint8 else 1.0)
    
    def _adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image contrast."""
        mean = np.mean(image)
        return np.clip((image - mean) * factor + mean, 0, 255 if image.dtype == np.uint8 else 1.0)
    
    def _add_noise(self, image: np.ndarray, std: float) -> np.ndarray:
        """Add Gaussian noise to image."""
        noise = np.random.normal(0, std, image.shape)
        return np.clip(image + noise, 0, 255 if image.dtype == np.uint8 else 1.0)
    
    def create_bird_eye_view(self, points: np.ndarray, resolution: float = 0.1) -> np.ndarray:
        """Create bird's eye view projection of LiDAR points.
        
        Args:
            points: LiDAR points (N, 4)
            resolution: Grid resolution in meters
            
        Returns:
            Bird's eye view image
        """
        x_min, x_max = self.lidar_range['x']
        y_min, y_max = self.lidar_range['y']
        
        # Calculate grid dimensions
        width = int((x_max - x_min) / resolution)
        height = int((y_max - y_min) / resolution)
        
        # Create empty grid
        bev = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Project points to grid
        x_coords = ((points[:, 0] - x_min) / resolution).astype(int)
        y_coords = ((points[:, 1] - y_min) / resolution).astype(int)
        
        # Filter valid coordinates
        valid_mask = (
            (x_coords >= 0) & (x_coords < width) &
            (y_coords >= 0) & (y_coords < height)
        )
        
        valid_x = x_coords[valid_mask]
        valid_y = y_coords[valid_mask]
        valid_z = points[valid_mask, 2]
        valid_intensity = points[valid_mask, 3]
        
        # Fill BEV image
        for i in range(len(valid_x)):
            # Height encoding (red channel)
            bev[valid_y[i], valid_x[i], 0] = min(255, int((valid_z[i] + 3) * 42.5))
            # Intensity encoding (green channel)
            bev[valid_y[i], valid_x[i], 1] = min(255, int(valid_intensity[i] * 255))
            # Density encoding (blue channel)
            bev[valid_y[i], valid_x[i], 2] = min(255, bev[valid_y[i], valid_x[i], 2] + 50)
        
        return bev
    
    def get_preprocessing_stats(self, image: np.ndarray, points: np.ndarray) -> Dict[str, Any]:
        """Get preprocessing statistics.
        
        Args:
            image: Original image
            points: Original points
            
        Returns:
            Statistics dictionary
        """
        processed_image = self.preprocess_image(image)
        processed_points = self.preprocess_lidar(points)
        
        return {
            'original_image_shape': image.shape,
            'processed_image_shape': processed_image.shape,
            'original_point_count': len(points),
            'processed_point_count': len(processed_points),
            'point_reduction_ratio': len(processed_points) / len(points) if len(points) > 0 else 0,
            'image_size_change': processed_image.shape != image.shape
        }