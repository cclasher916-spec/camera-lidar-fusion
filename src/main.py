#!/usr/bin/env python3
"""
Main execution script for Camera-LiDAR Fusion System

This script orchestrates the entire fusion pipeline including:
- Data loading and preprocessing
- Weather simulation
- Single sensor detection
- Multi-sensor fusion
- Evaluation and visualization
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

import yaml
from loguru import logger

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from data.data_loader import KITTIDataLoader
from detection.camera_detector import CameraDetector
from detection.lidar_detector import LiDARDetector
from detection.fusion import FusionSystem
from simulation.weather_effects import WeatherSimulator
from evaluation.metrics import MetricsCalculator
from evaluation.visualizer import ResultVisualizer


class FusionPipeline:
    """Main pipeline for Camera-LiDAR fusion system."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the fusion pipeline with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        
        # Initialize components
        self.data_loader = KITTIDataLoader(self.config['data'])
        self.camera_detector = CameraDetector(self.config['detection']['camera'])
        self.lidar_detector = LiDARDetector(self.config['detection']['lidar'])
        self.fusion_system = FusionSystem(self.config['fusion'])
        self.weather_simulator = WeatherSimulator(self.config['weather'])
        self.metrics_calculator = MetricsCalculator(self.config['evaluation'])
        self.visualizer = ResultVisualizer(self.config['visualization'])
        
        logger.info("Fusion pipeline initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'data': {
                'dataset_path': 'data/kitti',
                'sequence': '0001',
                'max_samples': 20
            },
            'detection': {
                'camera': {
                    'model': 'yolov8n.pt',
                    'confidence': 0.5,
                    'iou_threshold': 0.7
                },
                'lidar': {
                    'eps': 0.5,
                    'min_samples': 10,
                    'min_points': 50
                }
            },
            'fusion': {
                'method': 'weighted_average',
                'camera_weight': 0.6,
                'lidar_weight': 0.4
            },
            'weather': {
                'conditions': ['clear', 'fog', 'rain', 'low_light']
            },
            'evaluation': {
                'save_results': True,
                'output_dir': 'results/outputs'
            },
            'visualization': {
                'show_plots': True,
                'save_plots': True
            }
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logger.remove()  # Remove default handler
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO"
        )
        logger.add(
            "logs/fusion_pipeline.log",
            rotation="10 MB",
            retention="1 week",
            level="DEBUG"
        )
    
    def run_demo(self, sample_indices: list = None) -> Dict[str, Any]:
        """Run complete demonstration pipeline.
        
        Args:
            sample_indices: List of sample indices to process
            
        Returns:
            Dictionary containing all results
        """
        logger.info("Starting fusion demo pipeline")
        
        if sample_indices is None:
            sample_indices = list(range(min(10, self.config['data']['max_samples'])))
        
        results = {
            'samples': [],
            'metrics': {},
            'visualizations': []
        }
        
        # Process each sample
        for idx in sample_indices:
            logger.info(f"Processing sample {idx}")
            sample_result = self._process_sample(idx)
            results['samples'].append(sample_result)
        
        # Calculate overall metrics
        logger.info("Calculating overall metrics")
        results['metrics'] = self._calculate_overall_metrics(results['samples'])
        
        # Generate visualizations
        logger.info("Generating visualizations")
        results['visualizations'] = self._generate_visualizations(results)
        
        # Save results
        if self.config['evaluation']['save_results']:
            self._save_results(results)
        
        logger.info("Demo pipeline completed successfully")
        return results
    
    def _process_sample(self, sample_idx: int) -> Dict[str, Any]:
        """Process a single sample through the entire pipeline."""
        # Load data
        sample_data = self.data_loader.load_sample(sample_idx)
        
        results = {
            'sample_id': sample_idx,
            'weather_conditions': {},
            'detections': {},
            'fusion_results': {},
            'metrics': {}
        }
        
        # Process under different weather conditions
        for condition in self.config['weather']['conditions']:
            logger.debug(f"Processing under {condition} conditions")
            
            # Apply weather effects
            modified_data = self.weather_simulator.apply_weather(
                sample_data, condition
            )
            
            # Run detections
            camera_detections = self.camera_detector.detect(
                modified_data['camera_image']
            )
            lidar_detections = self.lidar_detector.detect(
                modified_data['lidar_points']
            )
            
            # Fusion
            fused_detections = self.fusion_system.fuse_detections(
                camera_detections, lidar_detections, sample_data['calibration']
            )
            
            # Store results
            results['weather_conditions'][condition] = modified_data
            results['detections'][condition] = {
                'camera': camera_detections,
                'lidar': lidar_detections,
                'fused': fused_detections
            }
            
            # Calculate metrics for this condition
            if 'ground_truth' in sample_data:
                condition_metrics = self.metrics_calculator.calculate_metrics(
                    fused_detections, sample_data['ground_truth']
                )
                results['metrics'][condition] = condition_metrics
        
        return results
    
    def _calculate_overall_metrics(self, sample_results: list) -> Dict[str, Any]:
        """Calculate overall metrics across all samples."""
        return self.metrics_calculator.aggregate_metrics(sample_results)
    
    def _generate_visualizations(self, results: Dict[str, Any]) -> list:
        """Generate all visualizations."""
        visualizations = []
        
        # Generate comparison plots
        for sample_result in results['samples']:
            vis = self.visualizer.create_comparison_plot(sample_result)
            visualizations.append(vis)
        
        # Generate metrics summary
        metrics_vis = self.visualizer.create_metrics_plot(results['metrics'])
        visualizations.append(metrics_vis)
        
        return visualizations
    
    def _save_results(self, results: Dict[str, Any]):
        """Save results to output directory."""
        output_dir = Path(self.config['evaluation']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        import json
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(results['metrics'], f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Camera-LiDAR Fusion System"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        nargs="+", 
        default=None,
        help="Sample indices to process"
    )
    parser.add_argument(
        "--demo", 
        action="store_true",
        help="Run demo mode with sample data"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = FusionPipeline(args.config)
    
    # Run demo or full pipeline
    if args.demo:
        logger.info("Running in demo mode")
        results = pipeline.run_demo(args.samples)
        print("\n=== Demo Results ===")
        print(f"Processed {len(results['samples'])} samples")
        print(f"Overall metrics: {results['metrics']}")
    else:
        logger.info("Running full pipeline")
        results = pipeline.run_demo(args.samples)
    
    return results


if __name__ == "__main__":
    main()