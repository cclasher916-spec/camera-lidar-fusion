"""Camera-LiDAR Calibration Utilities

Handles calibration data parsing and coordinate system transformations
between camera and LiDAR coordinate systems.
"""

from typing import Dict, Any, Tuple
import numpy as np
from loguru import logger


class CalibrationManager:
    """Manage camera-LiDAR calibration data and transformations."""
    
    def __init__(self, calibration_data: Dict[str, np.ndarray]):
        """Initialize calibration manager.
        
        Args:
            calibration_data: Dictionary containing calibration matrices
        """
        self.calibration_data = calibration_data
        self.validate_calibration_data()
        
        # Extract key matrices
        self.P2 = calibration_data.get('P2')  # Camera projection matrix
        self.R0_rect = calibration_data.get('R0_rect')  # Rectification matrix
        self.Tr_velo_to_cam = calibration_data.get('Tr_velo_to_cam')  # LiDAR to camera transform
        
        logger.info("Calibration manager initialized")
    
    def validate_calibration_data(self):
        """Validate that required calibration matrices are present and correctly shaped."""
        required_matrices = ['P2', 'R0_rect', 'Tr_velo_to_cam']
        expected_shapes = {
            'P2': (3, 4),
            'R0_rect': (3, 3),
            'Tr_velo_to_cam': (3, 4)
        }
        
        for matrix_name in required_matrices:
            if matrix_name not in self.calibration_data:
                logger.warning(f"Missing calibration matrix: {matrix_name}")
                continue
            
            matrix = self.calibration_data[matrix_name]
            expected_shape = expected_shapes[matrix_name]
            
            if matrix.shape != expected_shape:
                logger.warning(f"Unexpected shape for {matrix_name}: {matrix.shape}, expected {expected_shape}")
    
    def lidar_to_camera(self, points_lidar: np.ndarray) -> np.ndarray:
        """Transform points from LiDAR to camera coordinate system.
        
        Args:
            points_lidar: LiDAR points (N, 3) or (N, 4)
            
        Returns:
            Points in camera coordinate system (N, 3)
        """
        if self.Tr_velo_to_cam is None or self.R0_rect is None:
            logger.error("Missing required calibration matrices for LiDAR to camera transformation")
            return points_lidar[:, :3]
        
        # Take only XYZ coordinates
        points_xyz = points_lidar[:, :3]
        
        # Add homogeneous coordinate
        points_homo = np.hstack([points_xyz, np.ones((points_xyz.shape[0], 1))])
        
        # Transform from LiDAR to camera coordinate system
        points_cam = points_homo @ self.Tr_velo_to_cam.T
        
        # Apply rectification
        R0_homo = np.eye(4)
        R0_homo[:3, :3] = self.R0_rect
        
        points_cam_homo = np.hstack([points_cam, np.ones((points_cam.shape[0], 1))])
        points_rect = points_cam_homo @ R0_homo.T
        
        return points_rect[:, :3]
    
    def camera_to_image(self, points_camera: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project 3D camera points to 2D image coordinates.
        
        Args:
            points_camera: Points in camera coordinate system (N, 3)
            
        Returns:
            Tuple of (image_points (N, 2), valid_mask (N,))
        """
        if self.P2 is None:
            logger.error("Missing P2 matrix for camera to image projection")
            return np.zeros((points_camera.shape[0], 2)), np.zeros(points_camera.shape[0], dtype=bool)
        
        # Add homogeneous coordinate
        points_homo = np.hstack([points_camera, np.ones((points_camera.shape[0], 1))])
        
        # Project to image
        image_points_homo = points_homo @ self.P2.T
        
        # Normalize by depth
        valid_mask = image_points_homo[:, 2] > 0  # Points in front of camera
        
        image_points = np.zeros((points_camera.shape[0], 2))
        if np.any(valid_mask):
            image_points[valid_mask, 0] = image_points_homo[valid_mask, 0] / image_points_homo[valid_mask, 2]
            image_points[valid_mask, 1] = image_points_homo[valid_mask, 1] / image_points_homo[valid_mask, 2]
        
        return image_points, valid_mask
    
    def lidar_to_image(self, points_lidar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform LiDAR points directly to image coordinates.
        
        Args:
            points_lidar: LiDAR points (N, 3) or (N, 4)
            
        Returns:
            Tuple of (image_points (N, 2), valid_mask (N,))
        """
        # Transform to camera coordinates
        points_camera = self.lidar_to_camera(points_lidar)
        
        # Project to image
        return self.camera_to_image(points_camera)
    
    def get_camera_parameters(self) -> Dict[str, float]:
        """Extract camera intrinsic parameters from projection matrix.
        
        Returns:
            Dictionary containing camera parameters
        """
        if self.P2 is None:
            return {}
        
        # Extract intrinsic parameters from P2 matrix
        fx = self.P2[0, 0]
        fy = self.P2[1, 1]
        cx = self.P2[0, 2]
        cy = self.P2[1, 2]
        
        return {
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'baseline': -self.P2[0, 3] / fx if fx != 0 else 0  # Approximate baseline for stereo
        }
    
    def project_bbox_3d_to_2d(self, bbox_3d: np.ndarray) -> np.ndarray:
        """Project 3D bounding box to 2D image coordinates.
        
        Args:
            bbox_3d: 3D bounding box [x, y, z, w, l, h, ry]
            
        Returns:
            2D bounding box [x1, y1, x2, y2] or empty array if invalid
        """
        if len(bbox_3d) < 7:
            return np.array([])
        
        x, y, z, w, l, h, ry = bbox_3d[:7]
        
        # Create 3D box corners
        corners_3d = np.array([
            [-l/2, -w/2, 0], [l/2, -w/2, 0], [l/2, w/2, 0], [-l/2, w/2, 0],  # bottom
            [-l/2, -w/2, h], [l/2, -w/2, h], [l/2, w/2, h], [-l/2, w/2, h]   # top
        ])
        
        # Rotate corners
        cos_ry, sin_ry = np.cos(ry), np.sin(ry)
        rotation_matrix = np.array([
            [cos_ry, -sin_ry, 0],
            [sin_ry, cos_ry, 0],
            [0, 0, 1]
        ])
        
        rotated_corners = corners_3d @ rotation_matrix.T
        
        # Translate to world position
        rotated_corners[:, 0] += x
        rotated_corners[:, 1] += y
        rotated_corners[:, 2] += z
        
        # Project to image
        image_points, valid_mask = self.lidar_to_image(rotated_corners)
        
        if not np.any(valid_mask):
            return np.array([])  # No valid projections
        
        # Get 2D bounding box
        valid_points = image_points[valid_mask]
        
        x1, y1 = np.min(valid_points, axis=0)
        x2, y2 = np.max(valid_points, axis=0)
        
        return np.array([x1, y1, x2, y2])
    
    def is_point_in_image(self, image_point: np.ndarray, image_shape: Tuple[int, int]) -> bool:
        """Check if a projected point is within image boundaries.
        
        Args:
            image_point: 2D point [x, y]
            image_shape: Image shape (height, width)
            
        Returns:
            True if point is within image boundaries
        """
        height, width = image_shape
        x, y = image_point
        
        return 0 <= x < width and 0 <= y < height
    
    def filter_points_by_image(self, points_lidar: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """Filter LiDAR points to only include those that project into the image.
        
        Args:
            points_lidar: LiDAR points (N, 3) or (N, 4)
            image_shape: Image shape (height, width)
            
        Returns:
            Filtered LiDAR points
        """
        # Project to image
        image_points, valid_mask = self.lidar_to_image(points_lidar)
        
        # Check image boundaries
        height, width = image_shape
        in_image_mask = (
            valid_mask &
            (image_points[:, 0] >= 0) & (image_points[:, 0] < width) &
            (image_points[:, 1] >= 0) & (image_points[:, 1] < height)
        )
        
        return points_lidar[in_image_mask]
    
    def get_transformation_matrix(self) -> np.ndarray:
        """Get the complete transformation matrix from LiDAR to image coordinates.
        
        Returns:
            Combined transformation matrix (3, 4)
        """
        if self.P2 is None or self.R0_rect is None or self.Tr_velo_to_cam is None:
            return np.eye(3, 4)
        
        # Create homogeneous rectification matrix
        R0_homo = np.eye(4)
        R0_homo[:3, :3] = self.R0_rect
        
        # Create homogeneous velodyne transformation
        Tr_homo = np.eye(4)
        Tr_homo[:3, :] = self.Tr_velo_to_cam
        
        # Combine transformations: P2 * R0 * Tr
        combined = self.P2 @ R0_homo[:3, :] @ Tr_homo
        
        return combined
    
    def validate_transformation(self, test_points: np.ndarray) -> Dict[str, Any]:
        """Validate calibration by testing transformation on sample points.
        
        Args:
            test_points: Sample LiDAR points for validation
            
        Returns:
            Validation results
        """
        results = {
            'total_points': len(test_points),
            'valid_projections': 0,
            'avg_depth': 0.0,
            'transformation_errors': []
        }
        
        try:
            # Transform points
            camera_points = self.lidar_to_camera(test_points)
            image_points, valid_mask = self.camera_to_image(camera_points)
            
            results['valid_projections'] = np.sum(valid_mask)
            
            if np.any(valid_mask):
                valid_camera_points = camera_points[valid_mask]
                results['avg_depth'] = np.mean(valid_camera_points[:, 2])
            
            # Check for transformation consistency
            direct_transform = self.get_transformation_matrix()
            
            # Test a few points with both methods
            test_indices = np.random.choice(len(test_points), min(10, len(test_points)), replace=False)
            
            for idx in test_indices:
                point = test_points[idx]
                
                # Method 1: Step by step
                img_pt_1, valid_1 = self.lidar_to_image(point.reshape(1, -1))
                
                # Method 2: Direct transformation
                point_homo = np.append(point[:3], 1)
                img_pt_homo_2 = direct_transform @ point_homo
                
                if img_pt_homo_2[2] > 0:  # Valid depth
                    img_pt_2 = img_pt_homo_2[:2] / img_pt_homo_2[2]
                    
                    if valid_1[0]:
                        error = np.linalg.norm(img_pt_1[0] - img_pt_2)
                        results['transformation_errors'].append(error)
            
            if results['transformation_errors']:
                results['avg_transformation_error'] = np.mean(results['transformation_errors'])
                results['max_transformation_error'] = np.max(results['transformation_errors'])
            
        except Exception as e:
            logger.error(f"Calibration validation failed: {e}")
            results['error'] = str(e)
        
        return results