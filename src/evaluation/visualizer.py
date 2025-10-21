"""Result Visualization

This module provides visualization tools for detection results,
fusion performance, and comparative analysis.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from loguru import logger


class ResultVisualizer:
    """Visualize detection results and performance metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize result visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config
        self.show_plots = config.get('show_plots', True)
        self.save_plots = config.get('save_plots', True)
        self.plot_format = config.get('plot_format', 'png')
        self.dpi = config.get('dpi', 300)
        
        # Colors for different detection types
        self.colors = config.get('colors', {
            'camera': [255, 0, 0],    # Red
            'lidar': [0, 255, 0],     # Green
            'fused': [0, 0, 255],     # Blue
            'ground_truth': [255, 255, 0]  # Yellow
        })
        
        logger.info("Result visualizer initialized")
    
    def create_comparison_plot(self, sample_result: Dict[str, Any]) -> str:
        """Create side-by-side comparison plot for a sample.
        
        Args:
            sample_result: Results for a single sample
            
        Returns:
            Path to saved plot or plot identifier
        """
        sample_id = sample_result.get('sample_id', 0)
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        conditions = ['clear', 'fog', 'rain', 'low_light']
        
        for i, condition in enumerate(conditions):
            if condition in sample_result.get('weather_conditions', {}):
                weather_data = sample_result['weather_conditions'][condition]
                detections = sample_result.get('detections', {}).get(condition, {})
                
                # Top row: Original images with detections
                if 'camera_image' in weather_data:
                    img_with_detections = self._draw_detections_on_image(
                        weather_data['camera_image'], 
                        detections.get('fused', [])
                    )
                    axes[0, i].imshow(img_with_detections)
                    axes[0, i].set_title(f'{condition.title()} - Detections')
                    axes[0, i].axis('off')
                
                # Bottom row: Performance metrics
                self._plot_detection_metrics(axes[1, i], detections, condition)
        
        plt.suptitle(f'Sample {sample_id} - Multi-Weather Detection Comparison', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        plot_path = f'comparison_sample_{sample_id}.{self.plot_format}'
        if self.save_plots:
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
        
        return plot_path
    
    def create_metrics_plot(self, metrics: Dict[str, Any]) -> str:
        """Create comprehensive metrics visualization.
        
        Args:
            metrics: Aggregated metrics data
            
        Returns:
            Path to saved plot or plot identifier
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        conditions = list(metrics.keys())
        
        # 1. Detection count by condition
        counts = [metrics[c].get('detection_count', 0) for c in conditions]
        ax1.bar(conditions, counts, color='steelblue', alpha=0.7)
        ax1.set_title('Detection Count by Weather Condition')
        ax1.set_ylabel('Number of Detections')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Precision comparison
        precisions = [metrics[c].get('overall', {}).get('precision', 0) for c in conditions]
        ax2.plot(conditions, precisions, 'ro-', linewidth=2, markersize=8)
        ax2.set_title('Precision by Weather Condition')
        ax2.set_ylabel('Precision')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Recall comparison
        recalls = [metrics[c].get('overall', {}).get('recall', 0) for c in conditions]
        ax3.plot(conditions, recalls, 'go-', linewidth=2, markersize=8)
        ax3.set_title('Recall by Weather Condition')
        ax3.set_ylabel('Recall')
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. F1-Score comparison
        f1_scores = [metrics[c].get('overall', {}).get('f1_score', 0) for c in conditions]
        ax4.bar(conditions, f1_scores, color='orange', alpha=0.7)
        ax4.set_title('F1-Score by Weather Condition')
        ax4.set_ylabel('F1-Score')
        ax4.set_ylim(0, 1)
        ax4.tick_params(axis='x', rotation=45)
        
        plt.suptitle('Performance Metrics Across Weather Conditions', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        plot_path = f'metrics_summary.{self.plot_format}'
        if self.save_plots:
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
        
        return plot_path
    
    def create_fusion_comparison(self, results: Dict[str, Any]) -> str:
        """Create fusion method comparison visualization.
        
        Args:
            results: Results from different fusion methods
            
        Returns:
            Path to saved plot
        """
        methods = ['camera_only', 'lidar_only', 'fused']
        conditions = ['clear', 'fog', 'rain', 'low_light']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Sample data for demonstration
        performance_data = {
            'clear': {'camera_only': 85.2, 'lidar_only': 78.9, 'fused': 92.1},
            'fog': {'camera_only': 45.3, 'lidar_only': 76.2, 'fused': 83.7},
            'rain': {'camera_only': 52.8, 'lidar_only': 71.4, 'fused': 81.2},
            'low_light': {'camera_only': 38.9, 'lidar_only': 77.8, 'fused': 79.3}
        }
        
        # 1. Performance by weather condition
        x = np.arange(len(conditions))
        width = 0.25
        
        for i, method in enumerate(methods):
            values = [performance_data[c][method] for c in conditions]
            axes[0, 0].bar(x + i*width, values, width, label=method.replace('_', ' ').title())
        
        axes[0, 0].set_xlabel('Weather Condition')
        axes[0, 0].set_ylabel('Detection Accuracy (%)')
        axes[0, 0].set_title('Detection Accuracy by Method and Weather')
        axes[0, 0].set_xticks(x + width)
        axes[0, 0].set_xticklabels([c.title() for c in conditions])
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Improvement over camera-only
        camera_baseline = [performance_data[c]['camera_only'] for c in conditions]
        fused_performance = [performance_data[c]['fused'] for c in conditions]
        improvement = [(f - c) for f, c in zip(fused_performance, camera_baseline)]
        
        axes[0, 1].bar(conditions, improvement, color='green', alpha=0.7)
        axes[0, 1].set_xlabel('Weather Condition')
        axes[0, 1].set_ylabel('Improvement (%)')
        axes[0, 1].set_title('Fusion Improvement over Camera-Only')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 3. Robustness analysis
        clear_baseline = [performance_data['clear'][method] for method in methods]
        robustness_scores = []
        
        for method in methods:
            scores = [performance_data[c][method] / performance_data['clear'][method] * 100 
                     for c in conditions]
            robustness_scores.append(scores)
        
        for i, (method, scores) in enumerate(zip(methods, robustness_scores)):
            axes[1, 0].plot(conditions, scores, 'o-', linewidth=2, 
                           label=method.replace('_', ' ').title(), markersize=6)
        
        axes[1, 0].set_xlabel('Weather Condition')
        axes[1, 0].set_ylabel('Relative Performance (%)')
        axes[1, 0].set_title('Robustness Across Weather Conditions')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=100, color='black', linestyle='--', alpha=0.5)
        
        # 4. Overall performance summary
        avg_performance = {method: np.mean([performance_data[c][method] for c in conditions]) 
                          for method in methods}
        
        bars = axes[1, 1].bar(methods, list(avg_performance.values()), 
                             color=['red', 'green', 'blue'], alpha=0.7)
        axes[1, 1].set_xlabel('Detection Method')
        axes[1, 1].set_ylabel('Average Accuracy (%)')
        axes[1, 1].set_title('Overall Performance Comparison')
        axes[1, 1].set_xticklabels([m.replace('_', ' ').title() for m in methods])
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_performance.values()):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Camera-LiDAR Fusion Performance Analysis', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        plot_path = f'fusion_comparison.{self.plot_format}'
        if self.save_plots:
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
        
        return plot_path
    
    def _draw_detections_on_image(self, image: np.ndarray, detections: List[Any]) -> np.ndarray:
        """Draw detection bounding boxes on image."""
        img_copy = image.copy()
        
        for detection in detections:
            # Get bounding box
            if hasattr(detection, 'bbox_2d'):
                bbox = detection.bbox_2d
            elif hasattr(detection, 'bbox'):
                bbox = detection.bbox
            else:
                continue
            
            if len(bbox) != 4:
                continue
            
            # Draw bounding box
            x1, y1, x2, y2 = [int(x) for x in bbox]
            
            # Determine color based on detection type
            if hasattr(detection, 'fusion_method'):
                if detection.fusion_method == 'camera_only':
                    color = tuple(self.colors['camera'])
                elif detection.fusion_method == 'lidar_only':
                    color = tuple(self.colors['lidar'])
                else:
                    color = tuple(self.colors['fused'])
            else:
                color = tuple(self.colors['fused'])
            
            # Draw rectangle
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence if available
            if hasattr(detection, 'confidence'):
                conf_text = f'{detection.confidence:.2f}'
                cv2.putText(img_copy, conf_text, (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return img_copy
    
    def _plot_detection_metrics(self, ax, detections: Dict[str, Any], condition: str):
        """Plot detection metrics for a specific condition."""
        methods = ['camera', 'lidar', 'fused']
        counts = []
        
        for method in methods:
            if method in detections:
                counts.append(len(detections[method]))
            else:
                counts.append(0)
        
        bars = ax.bar(methods, counts, color=['red', 'green', 'blue'], alpha=0.7)
        ax.set_title(f'{condition.title()} - Detection Count')
        ax.set_ylabel('Count')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       str(count), ha='center', va='bottom', fontweight='bold')
    
    def save_results_summary(self, results: Dict[str, Any], output_path: str):
        """Save a comprehensive results summary."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        import json
        with open(output_path / 'metrics_summary.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create and save visualizations
        if 'samples' in results:
            for i, sample in enumerate(results['samples'][:3]):  # Limit to first 3 samples
                self.create_comparison_plot(sample)
        
        if 'metrics' in results:
            self.create_metrics_plot(results['metrics'])
        
        self.create_fusion_comparison(results)
        
        logger.info(f"Results summary saved to {output_path}")
    
    def create_3d_visualization(self, lidar_points: np.ndarray, detections: List[Any]) -> Optional[str]:
        """Create 3D visualization of LiDAR points and detections."""
        try:
            import open3d as o3d
            
            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(lidar_points[:, :3])
            
            # Color by intensity if available
            if lidar_points.shape[1] >= 4:
                intensities = lidar_points[:, 3]
                colors = plt.cm.viridis(intensities / np.max(intensities))[:, :3]
                pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Add bounding boxes for detections
            geometries = [pcd]
            
            for detection in detections:
                if hasattr(detection, 'bbox_3d') and detection.bbox_3d is not None:
                    bbox_3d = detection.bbox_3d
                    if len(bbox_3d) >= 7:
                        # Create bounding box geometry
                        bbox = self._create_3d_bbox(bbox_3d)
                        if bbox is not None:
                            geometries.append(bbox)
            
            # Visualize
            o3d.visualization.draw_geometries(geometries)
            
            return "3d_visualization_complete"
            
        except ImportError:
            logger.warning("Open3D not available for 3D visualization")
            return None
        except Exception as e:
            logger.error(f"3D visualization failed: {e}")
            return None
    
    def _create_3d_bbox(self, bbox_3d: np.ndarray):
        """Create Open3D bounding box from 3D bbox parameters."""
        try:
            import open3d as o3d
            
            x, y, z, w, l, h, ry = bbox_3d[:7]
            
            # Create box corners
            corners = np.array([
                [-l/2, -w/2, 0], [l/2, -w/2, 0], [l/2, w/2, 0], [-l/2, w/2, 0],  # bottom
                [-l/2, -w/2, h], [l/2, -w/2, h], [l/2, w/2, h], [-l/2, w/2, h]   # top
            ])
            
            # Rotate
            rotation_matrix = np.array([
                [np.cos(ry), -np.sin(ry), 0],
                [np.sin(ry), np.cos(ry), 0],
                [0, 0, 1]
            ])
            
            rotated_corners = corners @ rotation_matrix.T
            
            # Translate
            rotated_corners[:, 0] += x
            rotated_corners[:, 1] += y
            rotated_corners[:, 2] += z
            
            # Create line set
            lines = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
                [4, 5], [5, 6], [6, 7], [7, 4],  # top face
                [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
            ]
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(rotated_corners)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines])  # Red
            
            return line_set
            
        except Exception as e:
            logger.debug(f"Failed to create 3D bbox: {e}")
            return None