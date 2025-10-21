"""Performance Metrics Calculation

This module implements various metrics for evaluating
detection performance and fusion effectiveness.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger


class MetricsCalculator:
    """Calculate performance metrics for object detection systems."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize metrics calculator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.iou_thresholds = config.get('iou_thresholds', [0.3, 0.5, 0.7])
        self.confidence_thresholds = config.get('confidence_thresholds', [0.1, 0.3, 0.5, 0.7, 0.9])
        
        logger.info("Metrics calculator initialized")
    
    def calculate_metrics(self, detections: List[Any], ground_truth: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive metrics for detections.
        
        Args:
            detections: List of detection objects
            ground_truth: List of ground truth annotations
            
        Returns:
            Dictionary of calculated metrics
        """
        if not detections and not ground_truth:
            return self._empty_metrics()
        
        metrics = {}
        
        # Basic counting metrics
        metrics['detection_count'] = len(detections)
        metrics['ground_truth_count'] = len(ground_truth)
        
        # IoU-based metrics
        for iou_threshold in self.iou_thresholds:
            iou_metrics = self._calculate_iou_metrics(detections, ground_truth, iou_threshold)
            metrics[f'iou_{iou_threshold}'] = iou_metrics
        
        # Confidence-based metrics
        for conf_threshold in self.confidence_thresholds:
            conf_metrics = self._calculate_confidence_metrics(detections, conf_threshold)
            metrics[f'conf_{conf_threshold}'] = conf_metrics
        
        # Overall performance metrics
        metrics['overall'] = self._calculate_overall_metrics(detections, ground_truth)
        
        return metrics
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure."""
        return {
            'detection_count': 0,
            'ground_truth_count': 0,
            'overall': {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'ap': 0.0
            }
        }
    
    def _calculate_iou_metrics(self, detections: List[Any], ground_truth: List[Dict[str, Any]], iou_threshold: float) -> Dict[str, Any]:
        """Calculate IoU-based metrics."""
        if not detections or not ground_truth:
            return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        
        # Match detections to ground truth
        matches = self._match_detections_to_gt(detections, ground_truth, iou_threshold)
        
        true_positives = len(matches)
        false_positives = len(detections) - true_positives
        false_negatives = len(ground_truth) - true_positives
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def _calculate_confidence_metrics(self, detections: List[Any], conf_threshold: float) -> Dict[str, Any]:
        """Calculate confidence-based metrics."""
        if not detections:
            return {'count': 0, 'avg_confidence': 0.0}
        
        # Filter by confidence threshold
        high_conf_detections = [d for d in detections if getattr(d, 'confidence', 0) >= conf_threshold]
        
        return {
            'count': len(high_conf_detections),
            'avg_confidence': np.mean([getattr(d, 'confidence', 0) for d in high_conf_detections]) if high_conf_detections else 0.0
        }
    
    def _calculate_overall_metrics(self, detections: List[Any], ground_truth: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall performance metrics."""
        if not detections or not ground_truth:
            return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'ap': 0.0}
        
        # Use default IoU threshold for overall metrics
        default_iou = 0.5
        matches = self._match_detections_to_gt(detections, ground_truth, default_iou)
        
        true_positives = len(matches)
        false_positives = len(detections) - true_positives
        false_negatives = len(ground_truth) - true_positives
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Simple AP calculation (area under PR curve)
        ap = self._calculate_average_precision(detections, ground_truth)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'ap': ap
        }
    
    def _match_detections_to_gt(self, detections: List[Any], ground_truth: List[Dict[str, Any]], iou_threshold: float) -> List[tuple]:
        """Match detections to ground truth based on IoU."""
        matches = []
        used_gt = set()
        
        # Sort detections by confidence (descending)
        sorted_detections = sorted(enumerate(detections), 
                                 key=lambda x: getattr(x[1], 'confidence', 0), 
                                 reverse=True)
        
        for det_idx, detection in sorted_detections:
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching ground truth
            for gt_idx, gt in enumerate(ground_truth):
                if gt_idx in used_gt:
                    continue
                
                iou = self._calculate_bbox_iou(detection, gt)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # If match found, add to matches
            if best_gt_idx >= 0:
                matches.append((det_idx, best_gt_idx, best_iou))
                used_gt.add(best_gt_idx)
        
        return matches
    
    def _calculate_bbox_iou(self, detection: Any, ground_truth: Dict[str, Any]) -> float:
        """Calculate IoU between detection and ground truth bounding boxes."""
        try:
            # Get bounding boxes
            if hasattr(detection, 'bbox_2d'):
                det_bbox = detection.bbox_2d
            elif hasattr(detection, 'bbox'):
                det_bbox = detection.bbox
            else:
                return 0.0
            
            gt_bbox = ground_truth.get('bbox_2d', [])
            
            if len(det_bbox) != 4 or len(gt_bbox) != 4:
                return 0.0
            
            # Calculate intersection
            x1 = max(det_bbox[0], gt_bbox[0])
            y1 = max(det_bbox[1], gt_bbox[1])
            x2 = min(det_bbox[2], gt_bbox[2])
            y2 = min(det_bbox[3], gt_bbox[3])
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
            
            intersection = (x2 - x1) * (y2 - y1)
            
            # Calculate union
            det_area = (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1])
            gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
            union = det_area + gt_area - intersection
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.debug(f"Error calculating IoU: {e}")
            return 0.0
    
    def _calculate_average_precision(self, detections: List[Any], ground_truth: List[Dict[str, Any]]) -> float:
        """Calculate Average Precision (AP)."""
        if not detections or not ground_truth:
            return 0.0
        
        # Sort detections by confidence
        sorted_detections = sorted(detections, key=lambda x: getattr(x, 'confidence', 0), reverse=True)
        
        # Calculate precision-recall curve
        precisions = []
        recalls = []
        
        for i in range(1, len(sorted_detections) + 1):
            top_i_detections = sorted_detections[:i]
            matches = self._match_detections_to_gt(top_i_detections, ground_truth, 0.5)
            
            tp = len(matches)
            fp = len(top_i_detections) - tp
            fn = len(ground_truth) - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Calculate AP using trapezoidal rule
        if len(recalls) < 2:
            return 0.0
        
        # Sort by recall
        sorted_pairs = sorted(zip(recalls, precisions))
        recalls_sorted = [pair[0] for pair in sorted_pairs]
        precisions_sorted = [pair[1] for pair in sorted_pairs]
        
        # Calculate area under curve
        ap = 0.0
        for i in range(1, len(recalls_sorted)):
            ap += (recalls_sorted[i] - recalls_sorted[i-1]) * precisions_sorted[i]
        
        return ap
    
    def aggregate_metrics(self, sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across multiple samples.
        
        Args:
            sample_results: List of per-sample results
            
        Returns:
            Aggregated metrics
        """
        if not sample_results:
            return {}
        
        aggregated = {}
        
        # Aggregate by weather condition
        conditions = set()
        for sample in sample_results:
            if 'metrics' in sample:
                conditions.update(sample['metrics'].keys())
        
        for condition in conditions:
            condition_metrics = []
            for sample in sample_results:
                if 'metrics' in sample and condition in sample['metrics']:
                    condition_metrics.append(sample['metrics'][condition])
            
            if condition_metrics:
                aggregated[condition] = self._average_metrics(condition_metrics)
        
        return aggregated
    
    def _average_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Average a list of metrics dictionaries."""
        if not metrics_list:
            return {}
        
        averaged = {}
        
        # Get all keys
        all_keys = set()
        for metrics in metrics_list:
            if isinstance(metrics, dict):
                all_keys.update(metrics.keys())
        
        for key in all_keys:
            values = []
            for metrics in metrics_list:
                if isinstance(metrics, dict) and key in metrics:
                    value = metrics[key]
                    if isinstance(value, (int, float)):
                        values.append(value)
                    elif isinstance(value, dict) and 'overall' in value:
                        # Handle nested metrics
                        if 'precision' in value['overall']:
                            values.append(value['overall']['precision'])
            
            if values:
                averaged[key] = np.mean(values)
        
        return averaged
    
    def compare_methods(self, results_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare different detection methods.
        
        Args:
            results_dict: Dictionary mapping method names to their results
            
        Returns:
            Comparison metrics
        """
        comparison = {}
        
        methods = list(results_dict.keys())
        
        # Extract key metrics for comparison
        for method in methods:
            method_results = results_dict[method]
            comparison[method] = {
                'avg_precision': self._extract_avg_precision(method_results),
                'avg_recall': self._extract_avg_recall(method_results),
                'detection_count': self._extract_detection_count(method_results)
            }
        
        # Calculate improvements
        if len(methods) >= 2:
            baseline_method = methods[0]
            for method in methods[1:]:
                comparison[f'{method}_vs_{baseline_method}'] = {
                    'precision_improvement': comparison[method]['avg_precision'] - comparison[baseline_method]['avg_precision'],
                    'recall_improvement': comparison[method]['avg_recall'] - comparison[baseline_method]['avg_recall']
                }
        
        return comparison
    
    def _extract_avg_precision(self, results: Dict[str, Any]) -> float:
        """Extract average precision from results."""
        precisions = []
        for condition_results in results.values():
            if isinstance(condition_results, dict) and 'overall' in condition_results:
                if 'precision' in condition_results['overall']:
                    precisions.append(condition_results['overall']['precision'])
        return np.mean(precisions) if precisions else 0.0
    
    def _extract_avg_recall(self, results: Dict[str, Any]) -> float:
        """Extract average recall from results."""
        recalls = []
        for condition_results in results.values():
            if isinstance(condition_results, dict) and 'overall' in condition_results:
                if 'recall' in condition_results['overall']:
                    recalls.append(condition_results['overall']['recall'])
        return np.mean(recalls) if recalls else 0.0
    
    def _extract_detection_count(self, results: Dict[str, Any]) -> float:
        """Extract average detection count from results."""
        counts = []
        for condition_results in results.values():
            if isinstance(condition_results, dict) and 'detection_count' in condition_results:
                counts.append(condition_results['detection_count'])
        return np.mean(counts) if counts else 0.0