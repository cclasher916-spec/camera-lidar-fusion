"""KITTI Dataset Loading and Management

This module handles loading and preprocessing of KITTI dataset samples
including camera images, LiDAR point clouds, and calibration data.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import cv2
from loguru import logger

try:
    import open3d as o3d
except ImportError:
    logger.warning("Open3D not available, using alternative point cloud loading")
    o3d = None


class KITTIDataLoader:
    """KITTI dataset loader with support for camera and LiDAR data."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize KITTI data loader.
        
        Args:
            config: Configuration dictionary with dataset parameters
        """
        self.config = config
        self.dataset_path = Path(config.get('dataset_path', 'data/kitti'))
        self.sequence = config.get('sequence', '0001')
        self.max_samples = config.get('max_samples', 20)
        
        # Create sample data if not exists
        self._ensure_sample_data()
        
        logger.info(f"KITTI loader initialized for sequence {self.sequence}")
    
    def _ensure_sample_data(self):
        """Create sample data if KITTI dataset is not available."""
        if not self.dataset_path.exists():
            logger.info("KITTI dataset not found, creating sample data")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """Create synthetic sample data for demonstration."""
        sample_dir = self.dataset_path / "sample"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample images
        image_dir = sample_dir / "image_2"
        image_dir.mkdir(exist_ok=True)
        
        # Create sample LiDAR data
        velodyne_dir = sample_dir / "velodyne"
        velodyne_dir.mkdir(exist_ok=True)
        
        # Create sample calibration
        calib_dir = sample_dir / "calib"
        calib_dir.mkdir(exist_ok=True)
        
        for i in range(self.max_samples):
            # Create sample camera image
            img = self._create_sample_image(i)
            cv2.imwrite(str(image_dir / f"{i:06d}.png"), img)
            
            # Create sample LiDAR points
            points = self._create_sample_lidar(i)
            points.astype(np.float32).tofile(str(velodyne_dir / f"{i:06d}.bin"))
            
            # Create sample calibration (only need one)
            if i == 0:
                self._create_sample_calibration(calib_dir / "000000.txt")
        
        logger.info(f"Created {self.max_samples} sample data files")
    
    def _create_sample_image(self, idx: int) -> np.ndarray:
        """Create a synthetic camera image with vehicles."""
        # Create base image
        img = np.random.randint(100, 200, (376, 1241, 3), dtype=np.uint8)
        
        # Add road-like background
        img[300:, :] = [80, 80, 80]  # Road color
        img[200:300, :] = [135, 206, 235]  # Sky color
        
        # Add some "vehicles" (rectangles)
        vehicles = [
            (500 + idx * 10, 280, 80, 40),  # Moving vehicle
            (800, 290, 60, 35),
            (300, 295, 70, 38)
        ]
        
        for x, y, w, h in vehicles:
            if x < img.shape[1] and y < img.shape[0]:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), -1)
        
        return img
    
    def _create_sample_lidar(self, idx: int) -> np.ndarray:
        """Create synthetic LiDAR point cloud."""
        # Create ground plane
        x_ground = np.random.uniform(-20, 20, 5000)
        y_ground = np.random.uniform(-10, 10, 5000)
        z_ground = np.random.uniform(-1.8, -1.5, 5000)
        intensity_ground = np.random.uniform(0.1, 0.3, 5000)
        
        # Create vehicle-like clusters
        vehicles = [
            (5 + idx * 0.1, 0, -1, 200),  # Moving vehicle
            (15, 2, -1, 150),
            (10, -3, -1, 180)
        ]
        
        vehicle_points = []
        for vx, vy, vz, num_points in vehicles:
            # Create box-like point cluster
            x_veh = np.random.uniform(vx - 2, vx + 2, num_points)
            y_veh = np.random.uniform(vy - 1, vy + 1, num_points)
            z_veh = np.random.uniform(vz, vz + 1.5, num_points)
            intensity_veh = np.random.uniform(0.5, 0.9, num_points)
            
            vehicle_points.extend(zip(x_veh, y_veh, z_veh, intensity_veh))
        
        # Combine all points
        all_points = []
        for i in range(len(x_ground)):
            all_points.append([x_ground[i], y_ground[i], z_ground[i], intensity_ground[i]])
        
        all_points.extend(vehicle_points)
        
        return np.array(all_points, dtype=np.float32)
    
    def _create_sample_calibration(self, calib_path: Path):
        """Create sample calibration file."""
        # Standard KITTI calibration format
        calib_content = """P0: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 0.000000000000e+00 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P1: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.875744000000e+02 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P2: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 4.485728000000e+01 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.163791000000e-02 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.745884000000e-04
P3: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.395242000000e+02 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.199936000000e-02 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.729905000000e-04
R0_rect: 9.999239000000e-01 9.837760000000e-03 -7.445048000000e-03 -9.869795000000e-03 9.999421000000e-01 -4.278459000000e-03 7.402527000000e-03 4.351614000000e-03 9.999631000000e-01
Tr_velo_to_cam: 7.533745000000e-03 -9.999714000000e-01 -6.166020000000e-04 -4.069766000000e-03 1.480249000000e-02 7.280733000000e-04 -9.998902000000e-01 -7.631618000000e-02 9.998621000000e-01 7.523790000000e-03 1.480755000000e-02 -2.717806000000e-01
Tr_imu_to_velo: 9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03 -8.086759000000e-01 -7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02 3.195559000000e-01 2.024406000000e-03 1.482454000000e-02 9.998881000000e-01 -7.997231000000e-01"""
        
        with open(calib_path, 'w') as f:
            f.write(calib_content)
    
    def load_sample(self, sample_idx: int) -> Dict[str, Any]:
        """Load a complete sample with camera, LiDAR, and calibration data.
        
        Args:
            sample_idx: Index of sample to load
            
        Returns:
            Dictionary containing all sample data
        """
        sample_data = {
            'sample_id': sample_idx,
            'camera_image': self._load_camera_image(sample_idx),
            'lidar_points': self._load_lidar_points(sample_idx),
            'calibration': self._load_calibration(sample_idx)
        }
        
        # Add ground truth if available
        gt_path = self._get_ground_truth_path(sample_idx)
        if gt_path and gt_path.exists():
            sample_data['ground_truth'] = self._load_ground_truth(gt_path)
        else:
            # Create synthetic ground truth for demo
            sample_data['ground_truth'] = self._create_sample_ground_truth(sample_idx)
        
        logger.debug(f"Loaded sample {sample_idx}")
        return sample_data
    
    def _load_camera_image(self, sample_idx: int) -> np.ndarray:
        """Load camera image for given sample."""
        img_path = self.dataset_path / "sample" / "image_2" / f"{sample_idx:06d}.png"
        
        if img_path.exists():
            img = cv2.imread(str(img_path))
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            logger.warning(f"Image not found: {img_path}, creating sample")
            return self._create_sample_image(sample_idx)
    
    def _load_lidar_points(self, sample_idx: int) -> np.ndarray:
        """Load LiDAR point cloud for given sample."""
        lidar_path = self.dataset_path / "sample" / "velodyne" / f"{sample_idx:06d}.bin"
        
        if lidar_path.exists():
            points = np.fromfile(str(lidar_path), dtype=np.float32)
            return points.reshape(-1, 4)  # [x, y, z, intensity]
        else:
            logger.warning(f"LiDAR data not found: {lidar_path}, creating sample")
            return self._create_sample_lidar(sample_idx)
    
    def _load_calibration(self, sample_idx: int) -> Dict[str, np.ndarray]:
        """Load calibration matrices."""
        calib_path = self.dataset_path / "sample" / "calib" / "000000.txt"
        
        calibration = {}
        
        if calib_path.exists():
            with open(calib_path, 'r') as f:
                for line in f:
                    key, value = line.strip().split(':', 1)
                    calibration[key] = np.array([float(x) for x in value.split()])
        else:
            # Default calibration matrices
            calibration = self._get_default_calibration()
        
        # Reshape matrices
        if 'P2' in calibration:
            calibration['P2'] = calibration['P2'].reshape(3, 4)
        if 'R0_rect' in calibration:
            calibration['R0_rect'] = calibration['R0_rect'].reshape(3, 3)
        if 'Tr_velo_to_cam' in calibration:
            calibration['Tr_velo_to_cam'] = calibration['Tr_velo_to_cam'].reshape(3, 4)
        
        return calibration
    
    def _get_default_calibration(self) -> Dict[str, np.ndarray]:
        """Get default calibration matrices."""
        return {
            'P2': np.array([
                7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01,
                0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-02,
                0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-04
            ]),
            'R0_rect': np.array([
                9.999239e-01, 9.837760e-03, -7.445048e-03,
                -9.869795e-03, 9.999421e-01, -4.278459e-03,
                7.402527e-03, 4.351614e-03, 9.999631e-01
            ]),
            'Tr_velo_to_cam': np.array([
                7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03,
                1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02,
                9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01
            ])
        }
    
    def _get_ground_truth_path(self, sample_idx: int) -> Optional[Path]:
        """Get path to ground truth annotations."""
        gt_path = self.dataset_path / "sample" / "label_2" / f"{sample_idx:06d}.txt"
        return gt_path if gt_path.exists() else None
    
    def _load_ground_truth(self, gt_path: Path) -> list:
        """Load ground truth annotations."""
        annotations = []
        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 15:
                    annotation = {
                        'type': parts[0],
                        'bbox_2d': [float(x) for x in parts[4:8]],
                        'bbox_3d': [float(x) for x in parts[8:15]]
                    }
                    annotations.append(annotation)
        return annotations
    
    def _create_sample_ground_truth(self, sample_idx: int) -> list:
        """Create synthetic ground truth for demonstration."""
        # Create some sample bounding boxes
        return [
            {
                'type': 'Car',
                'bbox_2d': [500 + sample_idx * 10, 280, 580 + sample_idx * 10, 320],
                'bbox_3d': [5 + sample_idx * 0.1, 0, -1, 4, 1.5, 1.8, 0]
            },
            {
                'type': 'Car', 
                'bbox_2d': [800, 290, 860, 325],
                'bbox_3d': [15, 2, -1, 4, 1.5, 1.8, 0]
            }
        ]
    
    def get_sample_count(self) -> int:
        """Get total number of available samples."""
        return min(self.max_samples, len(list((self.dataset_path / "sample" / "image_2").glob("*.png"))))
    
    def get_sequence_info(self) -> Dict[str, Any]:
        """Get information about the loaded sequence."""
        return {
            'sequence': self.sequence,
            'dataset_path': str(self.dataset_path),
            'sample_count': self.get_sample_count(),
            'has_ground_truth': (self.dataset_path / "sample" / "label_2").exists()
        }