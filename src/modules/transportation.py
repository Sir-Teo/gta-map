"""
Transportation generation module.
Handles road networks, highways, and transportation infrastructure.
"""
import random
import math
from typing import List, Tuple, Optional
import numpy as np
from noise import pnoise2

from ..core.map_data import MapData, Road
from ..config.settings import TransportationConfig


class TransportationGenerator:
    """
    Generates transportation networks including highways, arterials, and local roads.
    """
    
    def generate_highway_network(self, map_data: MapData, config: TransportationConfig):
        """
        Generate a network of highways connecting major areas of the map.
        
        Args:
            map_data: The map data container to populate
            config: Transportation generation configuration
        """
        # Create a few main highway paths across the map
        num_highways = random.randint(*config.highway_count_range)
        
        # Define highway entry/exit points around the edges of the map
        edge_points = self._generate_edge_points(map_data)
        
        # Generate highways between different edge points
        for i in range(min(num_highways, len(edge_points)//2)):
            if len(edge_points) < 2:
                break
                
            start_point = edge_points.pop(0)
            # Try to find an endpoint on a different edge
            end_idx = 0
            for j, ep in enumerate(edge_points):
                if ep[2] != start_point[2]:  # Different edge type
                    end_idx = j
                    break
            end_point = edge_points.pop(end_idx)
            
            # Create a curved highway path between these points
            highway = self._generate_curved_highway(start_point, end_point, f"highway_{i}", config)
            if highway:
                map_data.add_road(highway)
                
        # Also add a ring road/beltway if the map is large enough
        if map_data.width > 1000 and map_data.height > 1000 and random.random() > 0.3:
            ring_highway = self._generate_ring_highway(map_data, config)
            if ring_highway:
                map_data.add_road(ring_highway)
    
    def generate_arterial_grid(self, map_data: MapData, config: TransportationConfig):
        """
        Generate arterial roads in a grid pattern.
        
        Args:
            map_data: The map data container to populate
            config: Transportation generation configuration
        """
        # Generate vertical arterials
        x = config.arterial_spacing
        arterial_id = 0
        while x < map_data.width - config.arterial_spacing:
            arterial = self._generate_vertical_arterial(map_data, x, f"arterial_v_{arterial_id}", config)
            if arterial:
                map_data.add_road(arterial)
            x += config.arterial_spacing + random.randint(-50, 50)  # Add some variation
            arterial_id += 1
        
        # Generate horizontal arterials
        y = config.arterial_spacing
        while y < map_data.height - config.arterial_spacing:
            arterial = self._generate_horizontal_arterial(map_data, y, f"arterial_h_{arterial_id}", config)
            if arterial:
                map_data.add_road(arterial)
            y += config.arterial_spacing + random.randint(-50, 50)  # Add some variation
            arterial_id += 1
    
    def generate_local_roads(self, map_data: MapData, config: TransportationConfig):
        """
        Generate local roads and collectors.
        
        Args:
            map_data: The map data container to populate
            config: Transportation generation configuration
        """
        # Generate collector roads between arterials
        self._generate_collector_roads(map_data, config)
        
        # Generate local roads in districts
        for district in map_data.districts.values():
            if district.polygon:
                self._generate_district_roads(map_data, district, config)
    
    def _generate_edge_points(self, map_data: MapData) -> List[Tuple[float, float, str]]:
        """Generate potential highway entry/exit points around map edges."""
        edge_points = []
        
        # Add potential entry/exit points at the edges
        for i in range(4, map_data.width-1, int(map_data.width/6)):  # Top edge
            if np.any(map_data.land_mask[:20, i//map_data.grid_size]):
                edge_points.append((i, 0, 'top'))
        
        for i in range(4, map_data.height-1, int(map_data.height/6)):  # Right edge
            if np.any(map_data.land_mask[i//map_data.grid_size, -20:]):
                edge_points.append((map_data.width, i, 'right'))
        
        for i in range(4, map_data.width-1, int(map_data.width/6)):  # Bottom edge
            if np.any(map_data.land_mask[-20:, i//map_data.grid_size]):
                edge_points.append((i, map_data.height, 'bottom'))
        
        for i in range(4, map_data.height-1, int(map_data.height/6)):  # Left edge
            if np.any(map_data.land_mask[i//map_data.grid_size, :20]):
                edge_points.append((0, i, 'left'))
        
        # Shuffle the edge points to get random connections
        random.shuffle(edge_points)
        return edge_points
    
    def _generate_curved_highway(
        self, 
        start_point: Tuple[float, float, str], 
        end_point: Tuple[float, float, str], 
        highway_id: str,
        config: TransportationConfig
    ) -> Optional[Road]:
        """Generate a curved highway path between two points."""
        start_x, start_y, _ = start_point
        end_x, end_y, _ = end_point
        
        # Generate 1-3 control points for a natural curve
        num_controls = random.randint(1, 3)
        
        # Calculate control points for Bezier-like curve
        control_points = []
        for i in range(num_controls):
            t = (i + 1) / (num_controls + 1)  # Evenly distribute along path
            
            # Base interpolation point
            base_x = start_x * (1 - t) + end_x * t
            base_y = start_y * (1 - t) + end_y * t
            
            # Add perpendicular offset for curve
            dx = end_x - start_x
            dy = end_y - start_y
            length = math.sqrt(dx**2 + dy**2)
            
            if length > 0:
                # Perpendicular direction (rotate 90 degrees)
                perp_x = -dy / length
                perp_y = dx / length
                
                # Random offset distance
                offset_distance = random.uniform(-length * 0.2, length * 0.2)
                
                control_x = base_x + perp_x * offset_distance
                control_y = base_y + perp_y * offset_distance
                
                control_points.append((control_x, control_y))
        
        # Generate smooth path through control points
        all_points = [(start_x, start_y)] + control_points + [(end_x, end_y)]
        highway_points = self._smooth_path(all_points, 50)
        
        return Road(
            id=highway_id,
            points=highway_points,
            road_type='highway',
            width=config.road_styles['highway']['width'],
            color=config.road_styles['highway']['color']
        )
    
    def _generate_ring_highway(self, map_data: MapData, config: TransportationConfig) -> Optional[Road]:
        """Generate a ring highway around the city center."""
        # Create a circular/elliptical ring road
        center_x = map_data.width / 2
        center_y = map_data.height / 2
        
        # Ring dimensions
        radius_x = min(map_data.width, map_data.height) * 0.3
        radius_y = radius_x * random.uniform(0.8, 1.2)  # Make it slightly elliptical
        
        ring_points = []
        num_points = 32
        
        for i in range(num_points + 1):  # +1 to close the loop
            angle = (i / num_points) * 2 * math.pi
            
            # Add some noise for more natural shape
            noise_val = pnoise2(math.cos(angle) * 0.5, math.sin(angle) * 0.5, octaves=3)
            radius_modifier = 1 + noise_val * 0.2
            
            x = center_x + math.cos(angle) * radius_x * radius_modifier
            y = center_y + math.sin(angle) * radius_y * radius_modifier
            
            # Clamp to map bounds
            x = max(50, min(x, map_data.width - 50))
            y = max(50, min(y, map_data.height - 50))
            
            ring_points.append((x, y))
        
        return Road(
            id='ring_highway',
            points=ring_points,
            road_type='highway',
            width=config.road_styles['highway']['width'],
            color=config.road_styles['highway']['color']
        )
    
    def _generate_vertical_arterial(
        self, 
        map_data: MapData, 
        x: float, 
        arterial_id: str, 
        config: TransportationConfig
    ) -> Optional[Road]:
        """Generate a vertical arterial road."""
        points = []
        current_y = 0
        
        # Add curve variation to the arterial
        base_x = x
        
        while current_y <= map_data.height:
            # Add slight curve variation
            curve_offset = math.sin(current_y * 0.01) * 30 * config.road_curve_factor
            road_x = base_x + curve_offset
            
            # Clamp to valid bounds
            road_x = max(0, min(road_x, map_data.width))
            
            points.append((road_x, current_y))
            current_y += 20  # Step size
        
        if len(points) < 2:
            return None
        
        return Road(
            id=arterial_id,
            points=points,
            road_type='arterial',
            width=config.road_styles['arterial']['width'],
            color=config.road_styles['arterial']['color']
        )
    
    def _generate_horizontal_arterial(
        self, 
        map_data: MapData, 
        y: float, 
        arterial_id: str, 
        config: TransportationConfig
    ) -> Optional[Road]:
        """Generate a horizontal arterial road."""
        points = []
        current_x = 0
        
        # Add curve variation to the arterial
        base_y = y
        
        while current_x <= map_data.width:
            # Add slight curve variation
            curve_offset = math.sin(current_x * 0.01) * 30 * config.road_curve_factor
            road_y = base_y + curve_offset
            
            # Clamp to valid bounds
            road_y = max(0, min(road_y, map_data.height))
            
            points.append((current_x, road_y))
            current_x += 20  # Step size
        
        if len(points) < 2:
            return None
        
        return Road(
            id=arterial_id,
            points=points,
            road_type='arterial',
            width=config.road_styles['arterial']['width'],
            color=config.road_styles['arterial']['color']
        )
    
    def _generate_collector_roads(self, map_data: MapData, config: TransportationConfig):
        """Generate collector roads between arterials."""
        # This is a simplified implementation
        # In a more complex version, this would create a proper secondary road network
        collector_id = 0
        
        # Add some diagonal collectors for variety
        for i in range(3):
            start_x = random.uniform(0, map_data.width)
            start_y = random.uniform(0, map_data.height)
            end_x = random.uniform(0, map_data.width)
            end_y = random.uniform(0, map_data.height)
            
            collector_points = self._smooth_path([(start_x, start_y), (end_x, end_y)], 20)
            
            collector = Road(
                id=f"collector_{collector_id}",
                points=collector_points,
                road_type='collector',
                width=config.road_styles['collector']['width'],
                color=config.road_styles['collector']['color']
            )
            
            map_data.add_road(collector)
            collector_id += 1
    
    def _generate_district_roads(self, map_data: MapData, district, config: TransportationConfig):
        """Generate local roads within a district."""
        if not district.polygon:
            return
        
        # Generate a few local roads within the district
        bounds = district.polygon.bounds
        min_x, min_y, max_x, max_y = bounds
        
        # Add 2-4 local roads per district
        num_local_roads = random.randint(2, 4)
        
        for i in range(num_local_roads):
            # Random start and end points within district bounds
            start_x = random.uniform(min_x, max_x)
            start_y = random.uniform(min_y, max_y)
            end_x = random.uniform(min_x, max_x)
            end_y = random.uniform(min_y, max_y)
            
            local_points = self._smooth_path([(start_x, start_y), (end_x, end_y)], 10)
            
            local_road = Road(
                id=f"local_{district.id}_{i}",
                points=local_points,
                road_type='local',
                width=config.road_styles['local']['width'],
                color=config.road_styles['local']['color']
            )
            
            map_data.add_road(local_road)
    
    def _smooth_path(self, control_points: List[Tuple[float, float]], num_segments: int) -> List[Tuple[float, float]]:
        """Create a smooth path through control points using linear interpolation."""
        if len(control_points) < 2:
            return control_points
        
        smooth_points = []
        
        for i in range(len(control_points) - 1):
            start_x, start_y = control_points[i]
            end_x, end_y = control_points[i + 1]
            
            # Interpolate between this segment
            for j in range(num_segments if i < len(control_points) - 2 else num_segments + 1):
                t = j / num_segments
                x = start_x * (1 - t) + end_x * t
                y = start_y * (1 - t) + end_y * t
                smooth_points.append((x, y))
        
        return smooth_points 