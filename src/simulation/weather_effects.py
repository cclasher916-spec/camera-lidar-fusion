"""Weather Effects Simulation

This module implements physics-based weather simulation for testing
sensor robustness under adverse conditions.
"""

from typing import Dict, Any
import numpy as np
import cv2
from loguru import logger


class WeatherSimulator:
    """Simulate various weather conditions affecting camera and LiDAR sensors."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize weather simulator.
        
        Args:
            config: Weather simulation configuration
        """
        self.config = config
        self.effects_config = config.get('effects', {})
        
        # Default weather parameters
        self.default_effects = {
            'fog': {
                'blur_kernel': 5,
                'brightness_factor': 0.7,
                'contrast_factor': 0.8,
                'lidar_attenuation': 0.9
            },
            'rain': {
                'noise_std': 15,
                'brightness_factor': 0.8,
                'contrast_factor': 1.2,
                'streak_density': 100,
                'lidar_noise_factor': 1.1
            },
            'low_light': {
                'brightness_factor': 0.3,
                'gamma': 1.5,
                'noise_std': 10,
                'lidar_intensity_factor': 0.8
            }
        }
        
        logger.info("Weather simulator initialized")
    
    def apply_weather(self, sample_data: Dict[str, Any], condition: str) -> Dict[str, Any]:
        """Apply weather effects to sensor data.
        
        Args:
            sample_data: Original sensor data
            condition: Weather condition to apply
            
        Returns:
            Modified sensor data with weather effects
        """
        if condition == 'clear':
            return sample_data.copy()
        
        modified_data = sample_data.copy()
        
        # Get effect parameters
        effect_params = self.effects_config.get(condition, self.default_effects.get(condition, {}))
        
        if condition == 'fog':
            modified_data = self._apply_fog(modified_data, effect_params)
        elif condition == 'rain':
            modified_data = self._apply_rain(modified_data, effect_params)
        elif condition == 'low_light':
            modified_data = self._apply_low_light(modified_data, effect_params)
        else:
            logger.warning(f"Unknown weather condition: {condition}")
        
        return modified_data
    
    def _apply_fog(self, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply fog effects to sensor data."""
        # Camera effects
        if 'camera_image' in data:
            img = data['camera_image'].copy()
            
            # Gaussian blur
            kernel_size = params.get('blur_kernel', 5)
            if kernel_size > 0:
                img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            
            # Brightness reduction
            brightness_factor = params.get('brightness_factor', 0.7)
            img = np.clip(img * brightness_factor, 0, 255).astype(np.uint8)
            
            # Contrast reduction
            contrast_factor = params.get('contrast_factor', 0.8)
            img = np.clip((img - 128) * contrast_factor + 128, 0, 255).astype(np.uint8)
            
            data['camera_image'] = img
        
        # LiDAR effects
        if 'lidar_points' in data:
            points = data['lidar_points'].copy()
            
            # Attenuate distant points (fog scattering)
            attenuation = params.get('lidar_attenuation', 0.9)
            distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
            
            # Apply distance-based attenuation
            keep_prob = attenuation ** (distances / 10.0)  # Stronger attenuation at distance
            keep_mask = np.random.random(len(points)) < keep_prob
            
            data['lidar_points'] = points[keep_mask]
        
        return data
    
    def _apply_rain(self, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply rain effects to sensor data."""
        # Camera effects
        if 'camera_image' in data:
            img = data['camera_image'].copy().astype(np.float32)
            
            # Add noise
            noise_std = params.get('noise_std', 15)
            noise = np.random.normal(0, noise_std, img.shape)
            img = img + noise
            
            # Brightness change
            brightness_factor = params.get('brightness_factor', 0.8)
            img = img * brightness_factor
            
            # Contrast change
            contrast_factor = params.get('contrast_factor', 1.2)
            img = (img - 128) * contrast_factor + 128
            
            # Add rain streaks
            streak_density = params.get('streak_density', 100)
            self._add_rain_streaks(img, streak_density)
            
            # Clip and convert back
            img = np.clip(img, 0, 255).astype(np.uint8)
            data['camera_image'] = img
        
        # LiDAR effects
        if 'lidar_points' in data:
            points = data['lidar_points'].copy()
            
            # Add noise to point positions
            noise_factor = params.get('lidar_noise_factor', 1.1)
            position_noise = np.random.normal(0, 0.02, points[:, :3].shape)
            points[:, :3] += position_noise
            
            # Add false returns from raindrops
            false_returns = self._generate_rain_false_returns(points, 0.1)
            if len(false_returns) > 0:
                points = np.vstack([points, false_returns])
            
            data['lidar_points'] = points
        
        return data
    
    def _apply_low_light(self, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply low light effects to sensor data."""
        # Camera effects
        if 'camera_image' in data:
            img = data['camera_image'].copy().astype(np.float32)
            
            # Brightness reduction
            brightness_factor = params.get('brightness_factor', 0.3)
            img = img * brightness_factor
            
            # Gamma correction
            gamma = params.get('gamma', 1.5)
            img = np.power(img / 255.0, gamma) * 255.0
            
            # Add noise (sensor noise increases in low light)
            noise_std = params.get('noise_std', 10)
            noise = np.random.normal(0, noise_std, img.shape)
            img = img + noise
            
            # Clip and convert
            img = np.clip(img, 0, 255).astype(np.uint8)
            data['camera_image'] = img
        
        # LiDAR effects (minimal - active sensor)
        if 'lidar_points' in data:
            points = data['lidar_points'].copy()
            
            # Reduce intensity (some materials reflect less in certain conditions)
            intensity_factor = params.get('lidar_intensity_factor', 0.8)
            points[:, 3] *= intensity_factor
            
            data['lidar_points'] = points
        
        return data
    
    def _add_rain_streaks(self, img: np.ndarray, density: int):
        """Add rain streak effects to image."""
        height, width = img.shape[:2]
        
        for _ in range(density):
            # Random streak position
            x = np.random.randint(0, width)
            y_start = np.random.randint(0, height // 2)
            streak_length = np.random.randint(5, 20)
            
            # Draw streak
            for i in range(streak_length):
                y = y_start + i
                if y < height:
                    # Add brightness (rain reflects light)
                    img[y, x] = np.minimum(img[y, x] + 30, 255)
    
    def _generate_rain_false_returns(self, points: np.ndarray, density: float) -> np.ndarray:
        """Generate false LiDAR returns from raindrops."""
        num_false = int(len(points) * density)
        
        if num_false == 0:
            return np.array([]).reshape(0, 4)
        
        # Generate random positions in sensor range
        x = np.random.uniform(-20, 20, num_false)
        y = np.random.uniform(-10, 10, num_false)
        z = np.random.uniform(-2, 5, num_false)
        intensity = np.random.uniform(0.1, 0.3, num_false)  # Low intensity
        
        false_returns = np.column_stack([x, y, z, intensity])
        return false_returns
    
    def get_weather_stats(self, original_data: Dict[str, Any], modified_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get statistics about weather effects applied."""
        stats = {}
        
        # Camera statistics
        if 'camera_image' in original_data and 'camera_image' in modified_data:
            orig_img = original_data['camera_image']
            mod_img = modified_data['camera_image']
            
            stats['camera'] = {
                'brightness_change': np.mean(mod_img) - np.mean(orig_img),
                'contrast_change': np.std(mod_img) - np.std(orig_img)
            }
        
        # LiDAR statistics
        if 'lidar_points' in original_data and 'lidar_points' in modified_data:
            orig_points = original_data['lidar_points']
            mod_points = modified_data['lidar_points']
            
            stats['lidar'] = {
                'point_count_change': len(mod_points) - len(orig_points),
                'intensity_change': np.mean(mod_points[:, 3]) - np.mean(orig_points[:, 3])
            }
        
        return stats