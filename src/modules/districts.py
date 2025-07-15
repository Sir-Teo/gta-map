"""
District generation module.
Handles district placement, types, and boundary generation using Voronoi diagrams.
"""
import random
import math
from typing import List, Tuple, Optional
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point

from ..core.map_data import MapData, District
from ..config.settings import DistrictConfig


class DistrictGenerator:
    """
    Generates city districts with realistic placement and boundaries.
    """
    
    def place_districts(self, map_data: MapData, config: DistrictConfig):
        """
        Place districts across the map using intelligent placement algorithms.
        
        Args:
            map_data: The map data container to populate
            config: District generation configuration
        """
        # Find viable land points for district centers
        land_points = np.argwhere(map_data.land_mask)
        if len(land_points) == 0:
            print("Warning: No land to place districts on.")
            return

        # Prioritize flatter areas for certain districts
        gradient_x, gradient_y = np.gradient(map_data.heightmap)
        slope = np.sqrt(gradient_x**2 + gradient_y**2)
        flat_areas = np.argwhere((map_data.land_mask) & (slope < 0.01))
        
        # Ensure we have enough districts to include all important types
        district_count = max(config.district_count, len(config.key_districts))
        
        # Place important districts first
        centers = []
        districts_placed = []
        
        # Explicitly place key district types first
        for district_type in config.key_districts:
            point = self._find_suitable_district_location(
                map_data, district_type, land_points, flat_areas, centers, config
            )
            
            if point is not None:
                # Convert grid coordinates to world coordinates
                point_world = point * map_data.grid_size
                
                # Check if this point is far enough from existing centers
                if not self._is_too_close_to_existing(point_world, centers, config.minimum_district_distance):
                    centers.append(point_world)
                    districts_placed.append(district_type)
                
                # Stop if we've reached our district count
                if len(centers) >= district_count:
                    break
        
        # Fill remaining district slots with random types if needed
        self._fill_remaining_districts(
            map_data, land_points, centers, districts_placed, 
            district_count, config
        )
        
        # Generate Voronoi diagram to define district boundaries
        if len(centers) >= 3:  # Voronoi requires at least 3 points
            self._create_district_boundaries(map_data, centers, districts_placed, config)
    
    def _find_suitable_district_location(
        self, 
        map_data: MapData, 
        district_type: str, 
        land_points: np.ndarray,
        flat_areas: np.ndarray,
        existing_centers: List[np.ndarray],
        config: DistrictConfig
    ) -> Optional[np.ndarray]:
        """Find a suitable location for a specific district type."""
        
        # Special placement logic for different district types
        if district_type in ['downtown', 'commercial']:
            # Downtown and commercial districts should be central
            center_area = np.array([map_data.grid_width/2, map_data.grid_height/2])
            distances = np.sum((land_points - center_area) ** 2, axis=1)
            closest_points = land_points[np.argsort(distances)[:20]]
            if len(closest_points) > 0:
                return closest_points[np.random.choice(len(closest_points))]
        
        elif district_type == 'airport':
            # Airports need large flat areas
            if len(flat_areas) > 0:
                # Sort by distance from edge for easier access
                edge_dist = np.minimum(
                    np.minimum(flat_areas[:, 0], map_data.grid_width - flat_areas[:, 0]),
                    np.minimum(flat_areas[:, 1], map_data.grid_height - flat_areas[:, 1])
                )
                airport_candidates = flat_areas[edge_dist < map_data.grid_width/4]
                if len(airport_candidates) > 0:
                    return airport_candidates[np.random.choice(len(airport_candidates))]
                else:
                    return flat_areas[np.random.choice(len(flat_areas))]
        
        elif district_type == 'beach':
            # Beaches should be near water at beach level
            beach_points = np.argwhere((map_data.heightmap > 0.35) & 
                                      (map_data.heightmap < 0.4))
            if len(beach_points) > 0:
                return beach_points[np.random.choice(len(beach_points))]
        
        elif district_type == 'port':
            # Ports should be right next to water
            water_edges = self._find_water_edges(map_data)
            if water_edges:
                return np.array(water_edges[np.random.choice(len(water_edges))])
        
        elif district_type == 'hills':
            # Hills districts prefer higher elevation
            high_areas = np.argwhere((map_data.land_mask) & (map_data.heightmap > 0.6))
            if len(high_areas) > 0:
                return high_areas[np.random.choice(len(high_areas))]
        
        # Default placement if special placement failed
        if len(land_points) > 0:
            return land_points[np.random.choice(len(land_points))]
        
        return None
    
    def _find_water_edges(self, map_data: MapData) -> List[List[int]]:
        """Find land points adjacent to water for port placement."""
        water_edges = []
        for i in range(1, map_data.grid_height-1):
            for j in range(1, map_data.grid_width-1):
                if map_data.land_mask[i, j]:
                    # Check if adjacent to water
                    if (not map_data.land_mask[i-1, j] or not map_data.land_mask[i+1, j] or
                        not map_data.land_mask[i, j-1] or not map_data.land_mask[i, j+1]):
                        water_edges.append([i, j])
        return water_edges
    
    def _is_too_close_to_existing(
        self, 
        point: np.ndarray, 
        existing_centers: List[np.ndarray], 
        min_distance: float
    ) -> bool:
        """Check if a point is too close to existing district centers."""
        for existing_center in existing_centers:
            if np.linalg.norm(existing_center - point) < min_distance:
                return True
        return False
    
    def _fill_remaining_districts(
        self,
        map_data: MapData,
        land_points: np.ndarray,
        centers: List[np.ndarray],
        districts_placed: List[str],
        target_count: int,
        config: DistrictConfig
    ):
        """Fill remaining district slots with random types."""
        remaining_districts = target_count - len(centers)
        if remaining_districts <= 0:
            return
        
        random_types = config.key_districts.copy()
        random.shuffle(random_types)
        
        for i in range(remaining_districts):
            district_type = random_types[i % len(random_types)]
            
            # Find a point that's not too close to existing centers
            attempts = 0
            found_point = False
            while attempts < 20 and not found_point:
                idx = np.random.choice(len(land_points))
                point = land_points[idx]
                point_world = point * map_data.grid_size
                
                # Check distance from existing centers
                if not self._is_too_close_to_existing(point_world, centers, 250):
                    centers.append(point_world)
                    districts_placed.append(district_type)
                    found_point = True
                
                attempts += 1
    
    def _create_district_boundaries(
        self,
        map_data: MapData,
        centers: List[np.ndarray],
        district_types: List[str],
        config: DistrictConfig
    ):
        """Create district boundaries using Voronoi diagrams."""
        vor = Voronoi([c.tolist() for c in centers])
        
        # Extract district vertices from Voronoi diagram
        for i, (center, district_type) in enumerate(zip(centers, district_types)):
            vertices = []
            
            # Find the region for this district center
            point_region_index = vor.point_region[i]
            region = vor.regions[point_region_index]
            
            if -1 not in region and len(region) > 0:
                for vertex_idx in region:
                    if vertex_idx >= 0 and vertex_idx < len(vor.vertices):
                        vertex = vor.vertices[vertex_idx].tolist()
                        # Clip to map boundaries
                        vertex[0] = max(0, min(vertex[0], map_data.width))
                        vertex[1] = max(0, min(vertex[1], map_data.height))
                        vertices.append(tuple(vertex))
            
            # Create district object
            district_id = f"{district_type}_{i}"
            district = District(
                id=district_id,
                center=tuple(center),
                district_type=district_type,
                vertices=vertices,
                radius=random.uniform(200, 400),
                color=config.colors.get(district_type, "#ffffff")
            )
            
            # Create polygon if we have enough vertices
            if len(vertices) >= 3:
                try:
                    district.polygon = Polygon(vertices)
                except Exception:
                    district.polygon = None
            
            map_data.add_district(district) 