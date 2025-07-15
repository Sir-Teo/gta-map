"""
Terrain generation module.
Handles heightmap generation, land/water definition, and water body creation.
"""
import math
import random
from typing import List, Tuple, Optional
import numpy as np
from noise import pnoise2
from shapely.geometry import Polygon, LineString, Point

from ..core.map_data import MapData, WaterBody
from ..config.settings import TerrainConfig, WaterConfig


class TerrainGenerator:
    """
    Generates terrain features including heightmaps and water bodies.
    """
    
    def generate_heightmap(self, map_data: MapData, config: TerrainConfig):
        """
        Generate a realistic heightmap with distinct mountain ranges and terrain features.
        
        Args:
            map_data: The map data container to populate
            config: Terrain generation configuration
        """
        map_data.heightmap = np.zeros((map_data.grid_height, map_data.grid_width))
        
        # Use multiple frequency Perlin noise for natural terrain
        scale = map_data.grid_width * config.terrain_scale
        mountain_scale = scale * config.mountain_scale_multiplier
        
        # Generate 1-3 distinct mountain ranges
        mountain_count = random.randint(*config.mountain_count_range)
        mountain_centers = []
        
        for _ in range(mountain_count):
            # Place mountain ranges away from the edges
            mx = random.uniform(map_data.width * 0.2, map_data.width * 0.8)
            my = random.uniform(map_data.height * 0.2, map_data.height * 0.8)
            mountain_centers.append((mx, my))
        
        # Generate the base terrain
        for y in range(map_data.grid_height):
            for x in range(map_data.grid_width):
                nx = x / map_data.grid_width - 0.5
                ny = y / map_data.grid_height - 0.5
                
                # Base continent shape (large features)
                e = (1.0 * pnoise2(1 * nx * scale, 1 * ny * scale, octaves=4, persistence=0.5, lacunarity=2.0) +
                     0.5 * pnoise2(2 * nx * scale, 2 * ny * scale, octaves=4, persistence=0.5, lacunarity=2.0) +
                     0.25 * pnoise2(4 * nx * scale, 4 * ny * scale, octaves=4, persistence=0.5, lacunarity=2.0))
                e /= (1 + 0.5 + 0.25)
                
                # Create a coastal falloff
                d = min(1, (math.sqrt(nx**2 + ny**2) / math.sqrt(0.5**2 + 0.5**2))**0.5)
                e = (e + 1) / 2  # map to 0-1
                e = e * (1 - d * 0.5)  # lower edges to form island
                
                # Add mountain ranges
                mountain_influence = 0
                for mx, my in mountain_centers:
                    # Convert mountain center to grid coordinates
                    grid_mx = mx / map_data.grid_size
                    grid_my = my / map_data.grid_size
                    
                    # Distance to mountain center
                    dist = math.sqrt((x - grid_mx)**2 + (y - grid_my)**2)
                    max_dist = min(map_data.grid_width, map_data.grid_height) * 0.3
                    
                    if dist < max_dist:
                        # Create a distance-based falloff for mountain height
                        falloff = 1 - (dist / max_dist)**2
                        
                        # Add some noise to the mountain
                        mountain_noise = pnoise2(x * mountain_scale * 0.1, y * mountain_scale * 0.1, 
                                              octaves=6, persistence=0.6, lacunarity=2.2) 
                        
                        # Higher persistence and octaves make more detailed mountains
                        mountain_height = falloff * (mountain_noise * 0.5 + 0.5) * 0.4
                        mountain_influence = max(mountain_influence, mountain_height)
                
                # Combine base terrain with mountains
                map_data.heightmap[y, x] = min(0.95, e + mountain_influence)  # Cap at 0.95 to avoid pure white
    
    def define_land_and_water(self, map_data: MapData, config: TerrainConfig):
        """
        Define which areas are land vs water based on the heightmap.
        
        Args:
            map_data: The map data container to populate
            config: Terrain generation configuration
        """
        map_data.land_mask = map_data.heightmap > config.water_level
    
    def generate_water_bodies(self, map_data: MapData, config: WaterConfig):
        """
        Generate natural water features like rivers and lakes.
        
        Args:
            map_data: The map data container to populate
            config: Water generation configuration
        """
        # Generate rivers
        num_rivers = random.randint(*config.river_count_range)
        for i in range(num_rivers):
            river = self._generate_river(map_data, config, f"river_{i}")
            if river:
                map_data.add_water_body(river)
        
        # Generate lakes
        num_lakes = random.randint(*config.lake_count_range)
        for i in range(num_lakes):
            lake = self._generate_lake(map_data, config, f"lake_{i}")
            if lake:
                map_data.add_water_body(lake)
    
    def _generate_river(self, map_data: MapData, config: WaterConfig, river_id: str) -> Optional[WaterBody]:
        """Generate a more visible and realistic winding river."""
        # Choose a starting edge - prefer top (mountains) or sides for more natural flow
        edge = random.choice(['top', 'top', 'right', 'left', 'bottom'])
        
        # Generate start point and initial direction
        if edge == 'top':
            start_x = random.randint(int(map_data.width * 0.2), int(map_data.width * 0.8))
            start_y = 0
            direction = (random.uniform(-0.3, 0.3), 1)  # Flow downward with slight variance
        elif edge == 'right':
            start_x = map_data.width
            start_y = random.randint(int(map_data.height * 0.2), int(map_data.height * 0.8))
            direction = (-1, random.uniform(-0.3, 0.3))  # Flow leftward with variance
        elif edge == 'bottom':
            start_x = random.randint(int(map_data.width * 0.2), int(map_data.width * 0.8))
            start_y = map_data.height
            direction = (random.uniform(-0.3, 0.3), -1)  # Flow upward with variance
        else:  # left
            start_x = 0
            start_y = random.randint(int(map_data.height * 0.2), int(map_data.height * 0.8))
            direction = (1, random.uniform(-0.3, 0.3))  # Flow rightward with variance
        
        # Normalize direction vector
        magnitude = (direction[0]**2 + direction[1]**2)**0.5
        direction = (direction[0]/magnitude, direction[1]/magnitude)
        
        # Generate river path
        river_points = [(start_x, start_y)]
        x, y = start_x, start_y
        
        # River properties
        river_width = random.uniform(*config.river_width_range)
        river_length = random.randint(int(max(map_data.width, map_data.height) * 0.5), 
                                   max(map_data.width, map_data.height))
        
        # Parameters for more natural river curves
        perlin_scale = 0.005
        perlin_strength = 70
        terrain_influence = 0.5
        
        for i in range(river_length):
            # Get current grid position
            grid_x = min(int(x / map_data.grid_size), map_data.grid_width-2)
            grid_y = min(int(y / map_data.grid_size), map_data.grid_height-2)
            
            # Terrain influence - rivers tend to flow downhill
            if 0 <= grid_x < map_data.grid_width-1 and 0 <= grid_y < map_data.grid_height-1:
                dx_terrain = map_data.heightmap[grid_y, grid_x+1] - map_data.heightmap[grid_y, grid_x]
                dy_terrain = map_data.heightmap[grid_y+1, grid_x] - map_data.heightmap[grid_y, grid_x]
                
                # Normalize terrain gradient if non-zero
                terrain_mag = math.sqrt(dx_terrain**2 + dy_terrain**2)
                if terrain_mag > 0.001:
                    dx_terrain /= terrain_mag
                    dy_terrain /= terrain_mag
            else:
                dx_terrain, dy_terrain = 0, 0
            
            # Add Perlin noise influence for natural meandering
            perlin_val = pnoise2(x * perlin_scale, y * perlin_scale, octaves=3, 
                               persistence=0.6, lacunarity=2.0)
            noise_angle = perlin_val * 2 * math.pi
            
            # Blend original direction, terrain influence, and noise
            nx = direction[0] * (0.4) - dx_terrain * terrain_influence + math.cos(noise_angle) * (0.2)
            ny = direction[1] * (0.4) - dy_terrain * terrain_influence + math.sin(noise_angle) * (0.2)
            
            # Renormalize direction
            mag = math.sqrt(nx**2 + ny**2)
            if mag > 0.001:
                nx, ny = nx/mag, ny/mag
            
            # Update direction for next iteration
            direction = (nx, ny)
            
            # Take a step
            step_size = 15
            x += nx * step_size
            y += ny * step_size
            
            # Check if we've gone outside the map or reached water
            if x < 0 or x >= map_data.width or y < 0 or y >= map_data.height:
                break
            
            river_points.append((x, y))
        
        # Only create river if we have enough points
        if len(river_points) > 10:
            # Update land mask to reflect the river
            self._update_land_mask_for_river(map_data, river_points, river_width)
            
            # Create river water body
            return WaterBody(
                id=river_id,
                water_type='river',
                geometry=LineString(river_points),
                color='#0077cc',
                width=river_width
            )
        
        return None
    
    def _generate_lake(self, map_data: MapData, config: WaterConfig, lake_id: str) -> Optional[WaterBody]:
        """Generate a lake in a low area of the map."""
        # Find areas with low elevation but above water level
        potential_areas = np.argwhere((map_data.heightmap > 0.35) & 
                                     (map_data.heightmap < 0.45))
        
        if len(potential_areas) == 0:
            return None
        
        # Select a random point in these areas
        idx = np.random.choice(len(potential_areas))
        center_y, center_x = potential_areas[idx]
        
        # Convert to map coordinates
        center_x = center_x * map_data.grid_size
        center_y = center_y * map_data.grid_size
        
        # Generate an irregular lake shape
        num_points = random.randint(8, 15)
        lake_radius = random.uniform(*config.lake_radius_range)
        lake_points = []
        
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            # Vary the radius with perlin noise
            radius_var = pnoise2(math.cos(angle), math.sin(angle), octaves=2, 
                               persistence=0.5, lacunarity=2.0)
            radius = lake_radius * (1 + radius_var * 0.5)  # +/- 50% variation
            x = center_x + math.cos(angle) * radius
            y = center_y + math.sin(angle) * radius
            lake_points.append((x, y))
        
        # Close the loop
        lake_points.append(lake_points[0])
        
        # Create the lake
        try:
            lake_polygon = Polygon(lake_points)
            if lake_polygon.is_valid and lake_polygon.area > 100:
                # Update land mask for the lake
                self._update_land_mask_for_lake(map_data, lake_polygon)
                
                return WaterBody(
                    id=lake_id,
                    water_type='lake',
                    geometry=lake_polygon,
                    color='#0066cc'
                )
        except Exception:
            pass  # Skip invalid polygons
        
        return None
    
    def _update_land_mask_for_river(self, map_data: MapData, river_points: List[Tuple[float, float]], width: float):
        """Update the land mask to include river area as water."""
        for i in range(len(river_points)-1):
            x1, y1 = river_points[i]
            x2, y2 = river_points[i+1]
            for t in np.linspace(0, 1, 30):
                x = int(x1 * (1-t) + x2 * t)
                y = int(y1 * (1-t) + y2 * t)
                
                # Convert to grid coordinates
                grid_x = min(int(x / map_data.grid_size), map_data.grid_width-1)
                grid_y = min(int(y / map_data.grid_size), map_data.grid_height-1)
                
                # Set a wider circle around each point as water
                if 0 <= grid_x < map_data.grid_width and 0 <= grid_y < map_data.grid_height:
                    river_grid_width = max(2, int(width / map_data.grid_size / 2))
                    for dx in range(-river_grid_width, river_grid_width+1):
                        for dy in range(-river_grid_width, river_grid_width+1):
                            if dx*dx + dy*dy <= river_grid_width*river_grid_width:
                                nx, ny = grid_x + dx, grid_y + dy
                                if 0 <= nx < map_data.grid_width and 0 <= ny < map_data.grid_height:
                                    map_data.land_mask[ny, nx] = False
    
    def _update_land_mask_for_lake(self, map_data: MapData, lake_polygon: Polygon):
        """Update the land mask to include lake area as water."""
        min_x, min_y, max_x, max_y = lake_polygon.bounds
        min_grid_x = max(0, int(min_x / map_data.grid_size))
        min_grid_y = max(0, int(min_y / map_data.grid_size))
        max_grid_x = min(map_data.grid_width-1, int(max_x / map_data.grid_size))
        max_grid_y = min(map_data.grid_height-1, int(max_y / map_data.grid_size))
        
        for grid_y in range(min_grid_y, max_grid_y+1):
            for grid_x in range(min_grid_x, max_grid_x+1):
                # Check if grid cell center is in lake polygon
                point = Point(grid_x * map_data.grid_size + map_data.grid_size/2, 
                             grid_y * map_data.grid_size + map_data.grid_size/2)
                if lake_polygon.contains(point):
                    map_data.land_mask[grid_y, grid_x] = False 