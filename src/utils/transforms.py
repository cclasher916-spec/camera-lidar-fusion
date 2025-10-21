"""Coordinate System Transformations

Utility functions for various coordinate system transformations
used in the camera-LiDAR fusion pipeline.
"""

from typing import Tuple, Union
import numpy as np
from loguru import logger


class CoordinateTransformer:
    """Handle various coordinate system transformations."""
    
    @staticmethod
    def cartesian_to_spherical(points: np.ndarray) -> np.ndarray:
        """Convert Cartesian coordinates to spherical coordinates.
        
        Args:
            points: Points in Cartesian coordinates (N, 3) [x, y, z]
            
        Returns:
            Points in spherical coordinates (N, 3) [r, theta, phi]
            r: radial distance
            theta: azimuthal angle (0 to 2π)
            phi: polar angle (0 to π)
        """
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y, x)
        phi = np.arccos(np.clip(z / np.maximum(r, 1e-8), -1, 1))
        
        # Ensure theta is in [0, 2π]
        theta = np.where(theta < 0, theta + 2*np.pi, theta)
        
        return np.column_stack([r, theta, phi])
    
    @staticmethod
    def spherical_to_cartesian(points: np.ndarray) -> np.ndarray:
        """Convert spherical coordinates to Cartesian coordinates.
        
        Args:
            points: Points in spherical coordinates (N, 3) [r, theta, phi]
            
        Returns:
            Points in Cartesian coordinates (N, 3) [x, y, z]
        """
        r, theta, phi = points[:, 0], points[:, 1], points[:, 2]
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        return np.column_stack([x, y, z])
    
    @staticmethod
    def points_to_homogeneous(points: np.ndarray) -> np.ndarray:
        """Convert points to homogeneous coordinates.
        
        Args:
            points: Points (N, D)
            
        Returns:
            Homogeneous points (N, D+1)
        """
        ones = np.ones((points.shape[0], 1))
        return np.hstack([points, ones])
    
    @staticmethod
    def homogeneous_to_points(points_homo: np.ndarray) -> np.ndarray:
        """Convert homogeneous points back to regular coordinates.
        
        Args:
            points_homo: Homogeneous points (N, D+1)
            
        Returns:
            Regular points (N, D)
        """
        return points_homo[:, :-1] / points_homo[:, -1:]
    
    @staticmethod
    def rotation_matrix_x(angle: float) -> np.ndarray:
        """Create rotation matrix around X-axis.
        
        Args:
            angle: Rotation angle in radians
            
        Returns:
            3x3 rotation matrix
        """
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        return np.array([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ])
    
    @staticmethod
    def rotation_matrix_y(angle: float) -> np.ndarray:
        """Create rotation matrix around Y-axis.
        
        Args:
            angle: Rotation angle in radians
            
        Returns:
            3x3 rotation matrix
        """
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        return np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
    
    @staticmethod
    def rotation_matrix_z(angle: float) -> np.ndarray:
        """Create rotation matrix around Z-axis.
        
        Args:
            angle: Rotation angle in radians
            
        Returns:
            3x3 rotation matrix
        """
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        return np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
    
    @staticmethod
    def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert Euler angles to rotation matrix.
        
        Args:
            roll: Roll angle (rotation around X-axis) in radians
            pitch: Pitch angle (rotation around Y-axis) in radians
            yaw: Yaw angle (rotation around Z-axis) in radians
            
        Returns:
            3x3 rotation matrix
        """
        R_x = CoordinateTransformer.rotation_matrix_x(roll)
        R_y = CoordinateTransformer.rotation_matrix_y(pitch)
        R_z = CoordinateTransformer.rotation_matrix_z(yaw)
        
        return R_z @ R_y @ R_x
    
    @staticmethod
    def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
        """Convert rotation matrix to Euler angles.
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            Tuple of (roll, pitch, yaw) in radians
        """
        # Extract angles (assuming ZYX order)
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
        
        return roll, pitch, yaw
    
    @staticmethod
    def create_transformation_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """Create 4x4 transformation matrix from rotation and translation.
        
        Args:
            rotation: 3x3 rotation matrix
            translation: 3D translation vector
            
        Returns:
            4x4 transformation matrix
        """
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = translation
        return T
    
    @staticmethod
    def transform_points(points: np.ndarray, transformation: np.ndarray) -> np.ndarray:
        """Apply transformation matrix to points.
        
        Args:
            points: Points (N, 3)
            transformation: 4x4 transformation matrix or 3x4 matrix
            
        Returns:
            Transformed points (N, 3)
        """
        # Convert to homogeneous coordinates
        points_homo = CoordinateTransformer.points_to_homogeneous(points)
        
        # Apply transformation
        if transformation.shape == (3, 4):
            # 3x4 transformation matrix
            transformed_homo = points_homo @ transformation.T
            return transformed_homo
        elif transformation.shape == (4, 4):
            # 4x4 transformation matrix
            transformed_homo = points_homo @ transformation.T
            return transformed_homo[:, :3]
        else:
            logger.error(f"Invalid transformation matrix shape: {transformation.shape}")
            return points
    
    @staticmethod
    def calculate_distance_matrix(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
        """Calculate pairwise distances between two sets of points.
        
        Args:
            points1: First set of points (N, D)
            points2: Second set of points (M, D)
            
        Returns:
            Distance matrix (N, M)
        """
        # Use broadcasting to compute all pairwise distances
        diff = points1[:, np.newaxis, :] - points2[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        return distances
    
    @staticmethod
    def find_nearest_neighbors(query_points: np.ndarray, reference_points: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors for each query point.
        
        Args:
            query_points: Query points (N, D)
            reference_points: Reference points (M, D)
            k: Number of nearest neighbors to find
            
        Returns:
            Tuple of (distances, indices)
            distances: (N, k) distances to nearest neighbors
            indices: (N, k) indices of nearest neighbors
        """
        distance_matrix = CoordinateTransformer.calculate_distance_matrix(query_points, reference_points)
        
        # Find k smallest distances for each query point
        nearest_indices = np.argsort(distance_matrix, axis=1)[:, :k]
        nearest_distances = np.sort(distance_matrix, axis=1)[:, :k]
        
        return nearest_distances, nearest_indices
    
    @staticmethod
    def project_points_to_plane(points: np.ndarray, plane_normal: np.ndarray, plane_point: np.ndarray) -> np.ndarray:
        """Project points onto a plane.
        
        Args:
            points: Points to project (N, 3)
            plane_normal: Normal vector of the plane (3,)
            plane_point: A point on the plane (3,)
            
        Returns:
            Projected points (N, 3)
        """
        # Normalize plane normal
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        
        # Calculate distance from each point to the plane
        point_to_plane = points - plane_point[np.newaxis, :]
        distances = np.dot(point_to_plane, plane_normal)
        
        # Project points onto plane
        projected_points = points - distances[:, np.newaxis] * plane_normal[np.newaxis, :]
        
        return projected_points
    
    @staticmethod
    def fit_plane_to_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit a plane to a set of 3D points using SVD.
        
        Args:
            points: 3D points (N, 3)
            
        Returns:
            Tuple of (plane_normal, plane_center)
        """
        if len(points) < 3:
            return np.array([0, 0, 1]), np.mean(points, axis=0)
        
        # Center the points
        center = np.mean(points, axis=0)
        centered_points = points - center
        
        # Use SVD to find the plane normal
        U, S, Vt = np.linalg.svd(centered_points)
        
        # The normal is the last column of V (or last row of Vt)
        plane_normal = Vt[-1, :]
        
        return plane_normal, center
    
    @staticmethod
    def compute_point_cloud_normals(points: np.ndarray, k_neighbors: int = 10) -> np.ndarray:
        """Estimate normal vectors for point cloud using local neighborhoods.
        
        Args:
            points: Point cloud (N, 3)
            k_neighbors: Number of neighbors to use for normal estimation
            
        Returns:
            Normal vectors (N, 3)
        """
        normals = np.zeros_like(points)
        
        for i, point in enumerate(points):
            # Find k nearest neighbors
            distances, indices = CoordinateTransformer.find_nearest_neighbors(
                point.reshape(1, -1), points, k_neighbors + 1  # +1 to exclude self
            )
            
            # Exclude the point itself (first in sorted list)
            neighbor_indices = indices[0, 1:]
            neighbors = points[neighbor_indices]
            
            # Fit plane to neighborhood
            if len(neighbors) >= 3:
                normal, _ = CoordinateTransformer.fit_plane_to_points(neighbors)
                normals[i] = normal
            else:
                normals[i] = np.array([0, 0, 1])  # Default upward normal
        
        return normals
    
    @staticmethod
    def remove_ground_plane(points: np.ndarray, distance_threshold: float = 0.1, 
                           height_threshold: float = -1.5) -> np.ndarray:
        """Remove ground plane points from point cloud.
        
        Args:
            points: Point cloud (N, 3) or (N, 4)
            distance_threshold: Maximum distance to plane for ground points
            height_threshold: Maximum height for potential ground points
            
        Returns:
            Filtered point cloud with ground points removed
        """
        if len(points) < 100:  # Not enough points for robust ground detection
            return points
        
        # Filter points by height first (rough ground estimation)
        potential_ground_mask = points[:, 2] < height_threshold + 0.5
        potential_ground_points = points[potential_ground_mask]
        
        if len(potential_ground_points) < 10:
            return points  # No clear ground plane
        
        # Fit plane to potential ground points
        try:
            plane_normal, plane_center = CoordinateTransformer.fit_plane_to_points(
                potential_ground_points[:, :3]
            )
            
            # Calculate distances to fitted plane
            point_to_plane = points[:, :3] - plane_center[np.newaxis, :]
            distances_to_plane = np.abs(np.dot(point_to_plane, plane_normal))
            
            # Remove points close to the ground plane
            ground_mask = (
                (distances_to_plane < distance_threshold) & 
                (points[:, 2] < height_threshold + 0.2)
            )
            
            return points[~ground_mask]
            
        except Exception as e:
            logger.debug(f"Ground plane removal failed: {e}")
            return points
    
    @staticmethod
    def downsample_point_cloud(points: np.ndarray, voxel_size: float = 0.1) -> np.ndarray:
        """Downsample point cloud using voxel grid.
        
        Args:
            points: Point cloud (N, 3) or (N, 4)
            voxel_size: Size of voxel grid
            
        Returns:
            Downsampled point cloud
        """
        if len(points) == 0:
            return points
        
        # Quantize points to voxel grid
        quantized = np.floor(points[:, :3] / voxel_size).astype(int)
        
        # Find unique voxels
        unique_voxels, unique_indices = np.unique(quantized, axis=0, return_index=True)
        
        # Return points corresponding to unique voxels
        return points[unique_indices]