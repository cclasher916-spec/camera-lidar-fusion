"""LiDAR-based Object Detection using Clustering

This module implements LiDAR-only object detection using clustering algorithms
(primarily DBSCAN) to identify vehicle-like point clusters.
"""

from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from loguru import logger

try:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available, using fallback clustering")
    SKLEARN_AVAILABLE = False

try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    logger.warning("Open3D not available, using numpy-only processing")
    O3D_AVAILABLE = False


class LiDARDetection:
    """Single LiDAR detection result."""
    
    def __init__(
        self, 
        points: np.ndarray, 
        center: np.ndarray, 
        bbox_3d: np.ndarray,
        confidence: float = 1.0
    ):
        self.points = points  # Point cloud of the detected object
        self.center = center  # 3D center [x, y, z]
        self.bbox_3d = bbox_3d  # 3D bounding box [x, y, z, w, l, h, ry]
        self.confidence = confidence
        
        # Calculate additional properties
        self.num_points = len(points)
        self.volume = bbox_3d[3] * bbox_3d[4] * bbox_3d[5] if len(bbox_3d) >= 6 else 0
        self.density = self.num_points / max(self.volume, 0.1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary format."""
        return {
            'center': self.center.tolist(),
            'bbox_3d': self.bbox_3d.tolist(),
            'confidence': self.confidence,
            'num_points': self.num_points,
            'volume': self.volume,
            'density': self.density
        }
    
    def project_to_image(self, calibration: Dict[str, np.ndarray]) -> Optional[List[float]]:
        """Project 3D bounding box to 2D image coordinates.
        
        Args:
            calibration: Camera calibration matrices
            
        Returns:
            2D bounding box [x1, y1, x2, y2] or None if not visible
        """
        try:
            # Get 3D bounding box corners
            corners_3d = self._get_bbox_corners()
            
            # Transform to camera coordinates
            if 'Tr_velo_to_cam' in calibration and 'R0_rect' in calibration:
                # Transform from LiDAR to camera coordinate system
                corners_cam = self._transform_points(
                    corners_3d, calibration['Tr_velo_to_cam'], calibration['R0_rect']
                )
                
                # Project to image
                if 'P2' in calibration:
                    corners_2d = self._project_to_image(corners_cam, calibration['P2'])
                    
                    # Get 2D bounding box
                    x_coords = corners_2d[:, 0]
                    y_coords = corners_2d[:, 1]
                    
                    # Filter points in front of camera
                    valid_mask = corners_cam[:, 2] > 0
                    if np.any(valid_mask):
                        x_valid = x_coords[valid_mask]
                        y_valid = y_coords[valid_mask]
                        
                        bbox_2d = [
                            np.min(x_valid), np.min(y_valid),
                            np.max(x_valid), np.max(y_valid)
                        ]
                        return bbox_2d
        
        except Exception as e:
            logger.debug(f"Failed to project to image: {e}")
        
        return None
    
    def _get_bbox_corners(self) -> np.ndarray:
        """Get 8 corners of 3D bounding box."""
        x, y, z, w, l, h = self.bbox_3d[:6]
        ry = self.bbox_3d[6] if len(self.bbox_3d) > 6 else 0
        
        # Create corners in object coordinate system
        corners = np.array([
            [-l/2, -w/2, 0],    # Bottom corners
            [l/2, -w/2, 0],
            [l/2, w/2, 0],
            [-l/2, w/2, 0],
            [-l/2, -w/2, h],    # Top corners
            [l/2, -w/2, h],
            [l/2, w/2, h],
            [-l/2, w/2, h]
        ])
        
        # Rotate around z-axis
        cos_ry = np.cos(ry)
        sin_ry = np.sin(ry)
        rotation_matrix = np.array([
            [cos_ry, -sin_ry, 0],
            [sin_ry, cos_ry, 0],
            [0, 0, 1]
        ])
        
        rotated_corners = corners @ rotation_matrix.T
        
        # Translate to world position
        rotated_corners[:, 0] += x
        rotated_corners[:, 1] += y
        rotated_corners[:, 2] += z
        
        return rotated_corners
    
    def _transform_points(
        self, 
        points: np.ndarray, 
        velo_to_cam: np.ndarray, 
        rect: np.ndarray
    ) -> np.ndarray:
        """Transform points from LiDAR to camera coordinates."""
        # Add homogeneous coordinate
        points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
        
        # Transform to camera coordinates
        cam_points = points_homo @ velo_to_cam.T
        
        # Apply rectification
        rect_homo = np.eye(4)
        rect_homo[:3, :3] = rect
        
        cam_points_homo = np.hstack([cam_points, np.ones((cam_points.shape[0], 1))])
        rect_points = cam_points_homo @ rect_homo.T
        
        return rect_points[:, :3]
    
    def _project_to_image(self, points_3d: np.ndarray, projection_matrix: np.ndarray) -> np.ndarray:
        """Project 3D points to image coordinates."""
        # Add homogeneous coordinate
        points_homo = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
        
        # Project to image
        image_points = points_homo @ projection_matrix.T
        
        # Normalize by depth
        image_points[:, 0] /= image_points[:, 2]
        image_points[:, 1] /= image_points[:, 2]
        
        return image_points[:, :2]


class LiDARDetector:
    """LiDAR-based object detector using clustering."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LiDAR detector.
        
        Args:
            config: Detection configuration
        """
        self.config = config
        
        # Clustering parameters
        clustering_config = config.get('clustering', {})
        self.clustering_method = clustering_config.get('method', 'dbscan')
        self.eps = clustering_config.get('eps', 0.5)
        self.min_samples = clustering_config.get('min_samples', 10)
        
        # Filtering parameters
        filtering_config = config.get('filtering', {})
        self.min_points = filtering_config.get('min_points', 50)
        self.max_points = filtering_config.get('max_points', 5000)
        self.height_range = filtering_config.get('height_range', [-2, 3])
        
        # Size filtering
        size_config = filtering_config.get('size_filter', {})
        self.min_length = size_config.get('min_length', 1.0)
        self.max_length = size_config.get('max_length', 15.0)
        self.min_width = size_config.get('min_width', 0.5)
        self.max_width = size_config.get('max_width', 3.0)
        
        logger.info("LiDAR detector initialized with clustering method: {}".format(self.clustering_method))
    
    def detect(self, points: np.ndarray) -> List[LiDARDetection]:
        """Detect objects in LiDAR point cloud.
        
        Args:
            points: Input point cloud (N, 4) [x, y, z, intensity]
            
        Returns:
            List of LiDARDetection objects
        """
        # Preprocess points
        filtered_points = self._preprocess_points(points)
        
        if len(filtered_points) < self.min_samples:
            logger.debug("Insufficient points for clustering")
            return []
        
        # Cluster points
        clusters = self._cluster_points(filtered_points)
        
        # Convert clusters to detections
        detections = self._clusters_to_detections(clusters, filtered_points)
        
        logger.debug(f"LiDAR detected {len(detections)} objects from {len(clusters)} clusters")
        return detections
    
    def _preprocess_points(self, points: np.ndarray) -> np.ndarray:
        """Preprocess point cloud by filtering and normalization.
        
        Args:
            points: Input points (N, 4)
            
        Returns:
            Filtered points
        """
        # Remove points outside height range
        height_mask = (
            (points[:, 2] >= self.height_range[0]) & 
            (points[:, 2] <= self.height_range[1])
        )
        filtered_points = points[height_mask]
        
        # Remove points too close or too far
        distance = np.sqrt(filtered_points[:, 0]**2 + filtered_points[:, 1]**2)
        distance_mask = (distance >= 2.0) & (distance <= 50.0)
        filtered_points = filtered_points[distance_mask]
        
        return filtered_points
    
    def _cluster_points(self, points: np.ndarray) -> List[np.ndarray]:
        """Cluster points using specified clustering algorithm.
        
        Args:
            points: Filtered points (N, 4)
            
        Returns:
            List of cluster indices arrays
        """
        if self.clustering_method == 'dbscan':
            return self._dbscan_clustering(points)
        else:
            return self._simple_grid_clustering(points)
    
    def _dbscan_clustering(self, points: np.ndarray) -> List[np.ndarray]:
        """Perform DBSCAN clustering.
        
        Args:
            points: Input points (N, 4)
            
        Returns:
            List of cluster point indices
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, using fallback clustering")
            return self._simple_grid_clustering(points)
        
        try:
            # Use only XY coordinates for clustering (ground plane)
            xy_points = points[:, :2]
            
            # Perform DBSCAN clustering
            clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            cluster_labels = clustering.fit_predict(xy_points)
            
            # Group points by cluster
            clusters = []
            unique_labels = set(cluster_labels)
            
            for label in unique_labels:
                if label == -1:  # Skip noise points
                    continue
                
                cluster_mask = cluster_labels == label
                cluster_points = points[cluster_mask]
                
                # Filter by size
                if len(cluster_points) >= self.min_points and len(cluster_points) <= self.max_points:
                    clusters.append(cluster_points)
            
            return clusters
            
        except Exception as e:
            logger.error(f"DBSCAN clustering failed: {e}")
            return self._simple_grid_clustering(points)
    
    def _simple_grid_clustering(self, points: np.ndarray, grid_size: float = 1.0) -> List[np.ndarray]:
        """Simple grid-based clustering as fallback.
        
        Args:
            points: Input points (N, 4)
            grid_size: Size of grid cells
            
        Returns:
            List of cluster point arrays
        """
        # Create grid indices
        grid_x = (points[:, 0] / grid_size).astype(int)
        grid_y = (points[:, 1] / grid_size).astype(int)
        
        # Group points by grid cell
        grid_dict = {}
        for i, (gx, gy) in enumerate(zip(grid_x, grid_y)):
            key = (gx, gy)
            if key not in grid_dict:
                grid_dict[key] = []
            grid_dict[key].append(i)
        
        # Convert to clusters
        clusters = []
        for indices in grid_dict.values():
            if len(indices) >= self.min_points:
                cluster_points = points[indices]
                clusters.append(cluster_points)
        
        return clusters
    
    def _clusters_to_detections(self, clusters: List[np.ndarray], all_points: np.ndarray) -> List[LiDARDetection]:
        """Convert point clusters to detection objects.
        
        Args:
            clusters: List of point clusters
            all_points: All filtered points
            
        Returns:
            List of LiDARDetection objects
        """
        detections = []
        
        for cluster_points in clusters:
            if len(cluster_points) < self.min_points:
                continue
            
            # Calculate bounding box and center
            bbox_3d, center = self._calculate_bbox_3d(cluster_points)
            
            # Filter by size
            if not self._is_valid_size(bbox_3d):
                continue
            
            # Calculate confidence based on point density and size
            confidence = self._calculate_confidence(cluster_points, bbox_3d)
            
            # Create detection
            detection = LiDARDetection(
                points=cluster_points,
                center=center,
                bbox_3d=bbox_3d,
                confidence=confidence
            )
            
            detections.append(detection)
        
        return detections
    
    def _calculate_bbox_3d(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate 3D bounding box from point cluster.
        
        Args:
            points: Cluster points (N, 4)
            
        Returns:
            Tuple of (bbox_3d, center)
            bbox_3d: [x, y, z, width, length, height, rotation_y]
            center: [x, y, z]
        """
        # Calculate basic statistics
        min_coords = np.min(points[:, :3], axis=0)
        max_coords = np.max(points[:, :3], axis=0)
        
        # Center and dimensions
        center = (min_coords + max_coords) / 2
        dimensions = max_coords - min_coords
        
        # Estimate rotation using PCA (simplified)
        rotation_y = self._estimate_rotation(points[:, :2])
        
        # Create bounding box: [x, y, z, width, length, height, rotation_y]
        bbox_3d = np.array([
            center[0], center[1], center[2],
            dimensions[1], dimensions[0], dimensions[2],  # width, length, height
            rotation_y
        ])
        
        return bbox_3d, center
    
    def _estimate_rotation(self, xy_points: np.ndarray) -> float:
        """Estimate object rotation using PCA.
        
        Args:
            xy_points: 2D points (N, 2)
            
        Returns:
            Rotation angle in radians
        """
        if len(xy_points) < 3:
            return 0.0
        
        # Center points
        centered = xy_points - np.mean(xy_points, axis=0)
        
        # Calculate covariance matrix
        cov_matrix = np.cov(centered.T)
        
        # Get principal component (eigenvector with largest eigenvalue)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        principal_component = eigenvectors[:, np.argmax(eigenvalues)]
        
        # Calculate rotation angle
        rotation_y = np.arctan2(principal_component[1], principal_component[0])
        
        return rotation_y
    
    def _is_valid_size(self, bbox_3d: np.ndarray) -> bool:
        """Check if bounding box has valid vehicle-like dimensions.
        
        Args:
            bbox_3d: Bounding box [x, y, z, width, length, height, rotation]
            
        Returns:
            True if valid size
        """
        width, length, height = bbox_3d[3], bbox_3d[4], bbox_3d[5]
        
        # Check size constraints
        valid_length = self.min_length <= length <= self.max_length
        valid_width = self.min_width <= width <= self.max_width
        valid_height = 0.5 <= height <= 4.0  # Reasonable vehicle height
        
        return valid_length and valid_width and valid_height
    
    def _calculate_confidence(self, points: np.ndarray, bbox_3d: np.ndarray) -> float:
        """Calculate detection confidence based on point characteristics.
        
        Args:
            points: Cluster points
            bbox_3d: 3D bounding box
            
        Returns:
            Confidence score between 0 and 1
        """
        num_points = len(points)
        volume = bbox_3d[3] * bbox_3d[4] * bbox_3d[5]  # width * length * height
        
        # Point density score
        density = num_points / max(volume, 0.1)
        density_score = min(1.0, density / 10.0)  # Normalize to [0, 1]
        
        # Size score (prefer typical vehicle sizes)
        length, width, height = bbox_3d[4], bbox_3d[3], bbox_3d[5]
        typical_vehicle_size = (2 <= length <= 6) and (1 <= width <= 2.5) and (1 <= height <= 2.5)
        size_score = 0.8 if typical_vehicle_size else 0.4
        
        # Point count score
        count_score = min(1.0, num_points / 200.0)  # Normalize based on expected point count
        
        # Combined confidence
        confidence = (density_score * 0.4 + size_score * 0.4 + count_score * 0.2)
        
        return min(1.0, confidence)
    
    def visualize_detections_3d(self, points: np.ndarray, detections: List[LiDARDetection]) -> Any:
        """Visualize LiDAR detections in 3D.
        
        Args:
            points: All LiDAR points
            detections: List of detections
            
        Returns:
            Visualization object (Open3D if available)
        """
        if not O3D_AVAILABLE:
            logger.warning("Open3D not available, cannot create 3D visualization")
            return None
        
        try:
            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            
            # Color points by intensity if available
            if points.shape[1] >= 4:
                intensity = points[:, 3]
                colors = plt.cm.viridis(intensity / np.max(intensity))[:, :3]
                pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Create bounding boxes
            bbox_list = []
            for detection in detections:
                bbox_corners = detection._get_bbox_corners()
                
                # Create line set for bounding box
                lines = [
                    [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
                    [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
                    [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
                ]
                
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(bbox_corners)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines])  # Red
                
                bbox_list.append(line_set)
            
            return [pcd] + bbox_list
            
        except Exception as e:
            logger.error(f"3D visualization failed: {e}")
            return None
    
    def get_detection_stats(self, detections: List[LiDARDetection]) -> Dict[str, Any]:
        """Get statistics about LiDAR detections.
        
        Args:
            detections: List of detections
            
        Returns:
            Statistics dictionary
        """
        if not detections:
            return {
                'total_count': 0,
                'point_stats': {},
                'size_stats': {},
                'confidence_stats': {}
            }
        
        # Point count statistics
        point_counts = [det.num_points for det in detections]
        point_stats = {
            'mean_points': np.mean(point_counts),
            'std_points': np.std(point_counts),
            'min_points': np.min(point_counts),
            'max_points': np.max(point_counts)
        }
        
        # Size statistics
        volumes = [det.volume for det in detections]
        size_stats = {
            'mean_volume': np.mean(volumes),
            'std_volume': np.std(volumes),
            'min_volume': np.min(volumes),
            'max_volume': np.max(volumes)
        }
        
        # Confidence statistics
        confidences = [det.confidence for det in detections]
        confidence_stats = {
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences)
        }
        
        return {
            'total_count': len(detections),
            'point_stats': point_stats,
            'size_stats': size_stats,
            'confidence_stats': confidence_stats
        }