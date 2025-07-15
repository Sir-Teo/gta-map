"""
Advanced terrain generation module.
Creates realistic terrains with proper mountain ranges, valleys, drainage systems, and natural biomes.
"""
import math
import random
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
from noise import pnoise2, snoise2
from shapely.geometry import Polygon, LineString, Point
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist, squareform

from ..core.map_data import MapData, WaterBody
from ..config.settings import TerrainConfig, WaterConfig


class AdvancedTerrainGenerator:
    """
    Advanced terrain generation with realistic geological and hydrological features.
    Creates natural-looking terrain that follows real-world patterns.
    """
    
    def __init__(self):
        self.mountain_ranges = []
        self.major_rivers = []
        self.drainage_basins = []
        self.biome_map = None
    
    def generate_heightmap(self, map_data: MapData, config: TerrainConfig):
        """
        Generate a realistic heightmap with proper geological features.
        Uses advanced noise techniques and geological modeling.
        """
        print("  → Creating base geological structure...")
        map_data.heightmap = np.zeros((map_data.grid_height, map_data.grid_width))
        
        # Create tectonic structure first
        self._generate_tectonic_structure(map_data, config)
        
        # Add mountain ranges with realistic formation patterns
        self._generate_mountain_ranges(map_data, config)
        
        # Create valleys and lowlands
        self._generate_valleys_and_lowlands(map_data, config)
        
        # Add coastal features and continental shelf
        self._generate_coastal_features(map_data, config)
        
        # Apply erosion simulation for more natural look
        self._simulate_erosion(map_data, config)
        
        # Smooth the terrain for more natural transitions
        map_data.heightmap = gaussian_filter(map_data.heightmap, sigma=1.0)
        
        print("  → Geological structure complete")
    
    def define_land_and_water(self, map_data: MapData, config: TerrainConfig):
        """
        Define land and water areas based on heightmap and add continental features.
        """
        print("  → Defining land and water boundaries...")
        map_data.land_mask = map_data.heightmap > config.water_level
        
        # Create more natural coastlines with bays and peninsulas
        self._create_natural_coastline(map_data, config)
        
        print("  → Land/water boundaries defined")
    
    def generate_water_bodies(self, map_data: MapData, config: WaterConfig):
        """
        Generate realistic river networks and water bodies following natural patterns.
        """
        print("  → Generating river networks...")
        
        # Generate primary river network based on topography
        self._generate_drainage_network(map_data, config)
        
        # Generate lakes in appropriate locations
        self._generate_natural_lakes(map_data, config)
        
        # Add coastal features like bays and inlets
        self._generate_coastal_waters(map_data, config)
        
        print(f"  → Generated {len(self.major_rivers)} major river systems")
    
    def generate_biomes(self, map_data: MapData):
        """
        Generate natural biomes based on elevation, water proximity, and climate.
        """
        print("  → Generating natural biomes...")
        
        self.biome_map = np.full((map_data.grid_height, map_data.grid_width), 'plains', dtype=object)
        
        # Generate biomes based on elevation and other factors
        for y in range(map_data.grid_height):
            for x in range(map_data.grid_width):
                elevation = map_data.heightmap[y, x]
                
                # Distance to water
                water_distance = self._calculate_water_distance(x, y, map_data)
                
                # Determine biome
                if elevation < 0.35:  # Water level
                    self.biome_map[y, x] = 'water'
                elif elevation < 0.4:  # Coastal
                    if water_distance < 5:
                        self.biome_map[y, x] = 'beach'
                    else:
                        self.biome_map[y, x] = 'coastal_plains'
                elif elevation < 0.55:  # Plains and lowlands
                    if water_distance < 3:
                        self.biome_map[y, x] = 'wetlands'
                    elif water_distance < 10:
                        self.biome_map[y, x] = 'agricultural'
                    else:
                        self.biome_map[y, x] = 'plains'
                elif elevation < 0.7:  # Hills
                    if water_distance > 15:
                        self.biome_map[y, x] = 'forest'
                    else:
                        self.biome_map[y, x] = 'hills'
                elif elevation < 0.85:  # Mountains
                    self.biome_map[y, x] = 'mountain_forest'
                else:  # High mountains
                    self.biome_map[y, x] = 'mountain_peaks'
        
        # Add some large forest areas
        self._generate_forest_patches(map_data)
        
        print("  → Biome generation complete")
    
    def _generate_tectonic_structure(self, map_data: MapData, config: TerrainConfig):
        """Create large-scale tectonic features."""
        # Create continental base shape
        center_x, center_y = map_data.grid_width // 2, map_data.grid_height // 2
        
        for y in range(map_data.grid_height):
            for x in range(map_data.grid_width):
                # Distance from center
                dx = (x - center_x) / map_data.grid_width
                dy = (y - center_y) / map_data.grid_height
                distance = math.sqrt(dx**2 + dy**2)
                
                # Create continental shelf with noise
                base_elevation = 0.6 * (1 - distance**1.5)
                
                # Add large-scale tectonic noise
                tectonic_noise = (
                    pnoise2(x * 0.002, y * 0.002, octaves=3, persistence=0.7, lacunarity=2.0) * 0.3 +
                    pnoise2(x * 0.001, y * 0.001, octaves=2, persistence=0.5, lacunarity=3.0) * 0.2
                )
                
                map_data.heightmap[y, x] = base_elevation + tectonic_noise
    
    def _generate_mountain_ranges(self, map_data: MapData, config: TerrainConfig):
        """Generate realistic mountain ranges with proper geological structure."""
        num_ranges = random.randint(*config.mountain_count_range)
        
        for i in range(num_ranges):
            # Create mountain range spine
            range_length = random.randint(200, 400)
            range_width = random.randint(50, 120)
            
            # Random starting point and direction with bounds checking
            margin = min(range_width, map_data.grid_width // 4, map_data.grid_height // 4)
            start_x = random.randint(margin, max(margin + 1, map_data.grid_width - margin))
            start_y = random.randint(margin, max(margin + 1, map_data.grid_height - margin))
            direction = random.uniform(0, 2 * math.pi)
            
            # Generate mountain spine points
            spine_points = []
            for j in range(range_length // 10):
                # Add some curvature to the range
                curve = pnoise2(j * 0.1, i * 100, octaves=2) * 0.5
                
                x = start_x + j * 10 * math.cos(direction + curve)
                y = start_y + j * 10 * math.sin(direction + curve)
                
                if 0 <= x < map_data.grid_width and 0 <= y < map_data.grid_height:
                    spine_points.append((int(x), int(y)))
            
            self.mountain_ranges.append(spine_points)
            
            # Apply mountain elevation around spine
            for sx, sy in spine_points:
                for dy in range(-range_width//2, range_width//2):
                    for dx in range(-range_width//2, range_width//2):
                        x, y = sx + dx, sy + dy
                        if 0 <= x < map_data.grid_width and 0 <= y < map_data.grid_height:
                            distance = math.sqrt(dx**2 + dy**2)
                            if distance < range_width//2:
                                # Mountain height with realistic falloff
                                falloff = 1 - (distance / (range_width//2))**2
                                
                                # Add detailed mountain noise
                                mountain_detail = (
                                    pnoise2(x * 0.05, y * 0.05, octaves=6, persistence=0.6) * 0.4 +
                                    pnoise2(x * 0.1, y * 0.1, octaves=4, persistence=0.4) * 0.2
                                )
                                
                                mountain_height = falloff * (0.4 + mountain_detail * 0.3)
                                map_data.heightmap[y, x] += mountain_height
    
    def _generate_valleys_and_lowlands(self, map_data: MapData, config: TerrainConfig):
        """Create valleys and lowland areas between mountain ranges."""
        for y in range(map_data.grid_height):
            for x in range(map_data.grid_width):
                # Create valley systems with negative noise
                valley_noise = pnoise2(x * 0.01, y * 0.01, octaves=4, persistence=0.5)
                
                if valley_noise < -0.3:  # Create valleys in low noise areas
                    valley_depth = abs(valley_noise + 0.3) * 0.3
                    map_data.heightmap[y, x] -= valley_depth
                    
                    # Ensure valleys don't go below sea level in wrong places
                    if map_data.heightmap[y, x] < 0.2:
                        map_data.heightmap[y, x] = 0.2
    
    def _generate_coastal_features(self, map_data: MapData, config: TerrainConfig):
        """Generate realistic coastal features like bays and peninsulas."""
        # Apply coastal erosion pattern
        for y in range(map_data.grid_height):
            for x in range(map_data.grid_width):
                # Distance to edge
                edge_distance = min(x, y, map_data.grid_width - x - 1, map_data.grid_height - y - 1)
                edge_factor = min(1.0, edge_distance / 50.0)
                
                # Coastal noise for bays and peninsulas
                coastal_noise = pnoise2(x * 0.02, y * 0.02, octaves=3, persistence=0.6)
                
                # Apply coastal effect
                if edge_distance < 100:
                    coastal_effect = (1 - edge_factor) * coastal_noise * 0.4
                    map_data.heightmap[y, x] += coastal_effect
    
    def _simulate_erosion(self, map_data: MapData, config: TerrainConfig):
        """Apply simplified erosion simulation for more natural terrain."""
        heightmap = map_data.heightmap.copy()
        
        # Simple erosion simulation
        for _ in range(3):  # Multiple erosion passes
            new_heightmap = heightmap.copy()
            
            for y in range(1, map_data.grid_height - 1):
                for x in range(1, map_data.grid_width - 1):
                    # Get neighboring heights
                    neighbors = [
                        heightmap[y-1, x], heightmap[y+1, x],
                        heightmap[y, x-1], heightmap[y, x+1]
                    ]
                    
                    avg_neighbor = sum(neighbors) / len(neighbors)
                    current = heightmap[y, x]
                    
                    # Erosion: high areas lose material, low areas gain some
                    if current > avg_neighbor:
                        erosion = (current - avg_neighbor) * 0.1
                        new_heightmap[y, x] -= erosion * 0.3
                    
            heightmap = new_heightmap
        
        map_data.heightmap = heightmap
    
    def _create_natural_coastline(self, map_data: MapData, config: TerrainConfig):
        """Create more natural, irregular coastlines."""
        # Add coastal noise to make irregular shorelines
        for y in range(map_data.grid_height):
            for x in range(map_data.grid_width):
                if abs(map_data.heightmap[y, x] - config.water_level) < 0.05:
                    # Near water level - add coastal irregularity
                    coastal_noise = pnoise2(x * 0.03, y * 0.03, octaves=4, persistence=0.6)
                    map_data.heightmap[y, x] += coastal_noise * 0.03
        
        # Recalculate land mask
        map_data.land_mask = map_data.heightmap > config.water_level
    
    def _generate_drainage_network(self, map_data: MapData, config: WaterConfig):
        """Generate realistic river networks following topographic flow."""
        # Find mountain peaks as river sources
        sources = self._find_river_sources(map_data)
        
        for i, source in enumerate(sources[:config.river_count_range[1]]):
            river = self._trace_river_from_source(map_data, source, f"river_{i}")
            if river and len(river.points) > 20:
                map_data.add_water_body(river)
                self.major_rivers.append(river)
                
                # Update land mask for river
                self._carve_river_channel(map_data, river)
    
    def _find_river_sources(self, map_data: MapData) -> List[Tuple[int, int]]:
        """Find good river source locations (mountain peaks, ridges)."""
        sources = []
        
        # Look for local maxima in mountainous areas
        for y in range(2, map_data.grid_height - 2):
            for x in range(2, map_data.grid_width - 2):
                if map_data.heightmap[y, x] > 0.7:  # High elevation
                    # Check if it's a local maximum
                    neighbors = [
                        map_data.heightmap[y-1, x-1], map_data.heightmap[y-1, x], map_data.heightmap[y-1, x+1],
                        map_data.heightmap[y, x-1], map_data.heightmap[y, x+1],
                        map_data.heightmap[y+1, x-1], map_data.heightmap[y+1, x], map_data.heightmap[y+1, x+1]
                    ]
                    
                    if map_data.heightmap[y, x] > max(neighbors) + 0.05:
                        sources.append((x, y))
        
        return sources[:8]  # Limit number of major rivers
    
    def _trace_river_from_source(self, map_data: MapData, source: Tuple[int, int], river_id: str) -> Optional[WaterBody]:
        """Trace a river path from source following steepest descent."""
        start_x, start_y = source
        river_points = [(start_x * map_data.grid_size, start_y * map_data.grid_size)]
        
        current_x, current_y = start_x, start_y
        visited = set()
        
        while True:
            visited.add((current_x, current_y))
            
            # Find steepest descent direction
            best_direction = None
            best_elevation = map_data.heightmap[current_y, current_x]
            
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                new_x, new_y = current_x + dx, current_y + dy
                
                if (0 <= new_x < map_data.grid_width and 
                    0 <= new_y < map_data.grid_height and 
                    (new_x, new_y) not in visited):
                    
                    elevation = map_data.heightmap[new_y, new_x]
                    if elevation < best_elevation:
                        best_elevation = elevation
                        best_direction = (dx, dy)
            
            if best_direction is None:
                break  # Reached a minimum
            
            current_x += best_direction[0]
            current_y += best_direction[1]
            
            # Convert to map coordinates
            map_x = current_x * map_data.grid_size
            map_y = current_y * map_data.grid_size
            river_points.append((map_x, map_y))
            
            # Stop if we reach water level or map edge
            if (map_data.heightmap[current_y, current_x] <= 0.35 or
                current_x <= 0 or current_x >= map_data.grid_width - 1 or
                current_y <= 0 or current_y >= map_data.grid_height - 1):
                break
            
            # Prevent infinite loops
            if len(river_points) > 500:
                break
        
        if len(river_points) > 10:
            river_width = random.uniform(15, 40)
            return WaterBody(
                id=river_id,
                water_type='river',
                geometry=LineString(river_points),
                color='#0077cc',
                width=river_width
            )
        
        return None
    
    def _carve_river_channel(self, map_data: MapData, river: WaterBody):
        """Carve river channel into the terrain and update land mask."""
        if not isinstance(river.geometry, LineString):
            return
        
        points = list(river.geometry.coords)
        width = river.width or 20
        
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            
            # Convert to grid coordinates
            gx1, gy1 = int(x1 / map_data.grid_size), int(y1 / map_data.grid_size)
            gx2, gy2 = int(x2 / map_data.grid_size), int(y2 / map_data.grid_size)
            
            # Interpolate between points
            steps = max(abs(gx2 - gx1), abs(gy2 - gy1), 1)
            for step in range(steps + 1):
                t = step / steps if steps > 0 else 0
                gx = int(gx1 + t * (gx2 - gx1))
                gy = int(gy1 + t * (gy2 - gy1))
                
                # Carve river channel
                grid_width = max(2, int(width / map_data.grid_size))
                for dy in range(-grid_width, grid_width + 1):
                    for dx in range(-grid_width, grid_width + 1):
                        rx, ry = gx + dx, gy + dy
                        if 0 <= rx < map_data.grid_width and 0 <= ry < map_data.grid_height:
                            distance = math.sqrt(dx**2 + dy**2)
                            if distance <= grid_width:
                                # Carve the channel
                                carve_depth = (1 - distance / grid_width) * 0.1
                                map_data.heightmap[ry, rx] -= carve_depth
                                
                                # Update land mask
                                if map_data.heightmap[ry, rx] <= 0.35:
                                    map_data.land_mask[ry, rx] = False
    
    def _generate_natural_lakes(self, map_data: MapData, config: WaterConfig):
        """Generate lakes in natural depressions and valley floors."""
        num_lakes = random.randint(*config.lake_count_range)
        
        for i in range(num_lakes):
            # Find a good depression for a lake
            lake_center = self._find_lake_location(map_data)
            if lake_center:
                lake = self._create_natural_lake(map_data, lake_center, f"lake_{i}", config)
                if lake:
                    map_data.add_water_body(lake)
    
    def _find_lake_location(self, map_data: MapData) -> Optional[Tuple[int, int]]:
        """Find a natural depression suitable for a lake."""
        for _ in range(50):  # Try up to 50 random locations
            x = random.randint(10, map_data.grid_width - 10)
            y = random.randint(10, map_data.grid_height - 10)
            
            elevation = map_data.heightmap[y, x]
            
            # Check if it's in a suitable elevation range and is a local minimum
            if 0.4 < elevation < 0.6:
                neighbors = []
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        if 0 <= x + dx < map_data.grid_width and 0 <= y + dy < map_data.grid_height:
                            neighbors.append(map_data.heightmap[y + dy, x + dx])
                
                if elevation < sum(neighbors) / len(neighbors) - 0.02:  # Local minimum
                    return (x, y)
        
        return None
    
    def _create_natural_lake(self, map_data: MapData, center: Tuple[int, int], lake_id: str, config: WaterConfig) -> Optional[WaterBody]:
        """Create a natural-looking lake at the specified location."""
        cx, cy = center
        base_radius = random.uniform(*config.lake_radius_range) / map_data.grid_size
        
        # Generate irregular lake shape
        lake_points = []
        num_points = random.randint(12, 20)
        
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            
            # Vary radius using noise for natural shape
            radius_noise = pnoise2(math.cos(angle) * 2, math.sin(angle) * 2, octaves=3)
            radius = base_radius * (1 + radius_noise * 0.4)
            
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            
            # Convert to map coordinates
            map_x = x * map_data.grid_size
            map_y = y * map_data.grid_size
            lake_points.append((map_x, map_y))
        
        if len(lake_points) >= 3:
            lake_polygon = Polygon(lake_points)
            
            # Update terrain for lake
            for dy in range(-int(base_radius) - 2, int(base_radius) + 3):
                for dx in range(-int(base_radius) - 2, int(base_radius) + 3):
                    x, y = cx + dx, cy + dy
                    if 0 <= x < map_data.grid_width and 0 <= y < map_data.grid_height:
                        point = Point(x * map_data.grid_size, y * map_data.grid_size)
                        if lake_polygon.contains(point) or lake_polygon.distance(point) < 5:
                            map_data.heightmap[y, x] = 0.32  # Below water level
                            map_data.land_mask[y, x] = False
            
            return WaterBody(
                id=lake_id,
                water_type='lake',
                geometry=lake_polygon,
                color='#0066bb'
            )
        
        return None
    
    def _generate_coastal_waters(self, map_data: MapData, config: WaterConfig):
        """Generate coastal water features like bays and inlets."""
        # This would add coastal features - for now, keep it simple
        pass
    
    def _calculate_water_distance(self, x: int, y: int, map_data: MapData) -> float:
        """Calculate distance to nearest water."""
        if not map_data.land_mask[y, x]:
            return 0
        
        # Simple distance calculation to water
        min_dist = float('inf')
        for dy in range(-20, 21):
            for dx in range(-20, 21):
                nx, ny = x + dx, y + dy
                if 0 <= nx < map_data.grid_width and 0 <= ny < map_data.grid_height:
                    if not map_data.land_mask[ny, nx]:
                        dist = math.sqrt(dx**2 + dy**2)
                        min_dist = min(min_dist, dist)
        
        return min_dist if min_dist != float('inf') else 20
    
    def _generate_forest_patches(self, map_data: MapData):
        """Generate large forest areas in suitable locations."""
        # Create 3-6 large forest areas
        num_forests = random.randint(3, 6)
        
        for _ in range(num_forests):
            # Find a good location for forest
            for _ in range(20):  # Try multiple locations
                fx = random.randint(20, map_data.grid_width - 20)
                fy = random.randint(20, map_data.grid_height - 20)
                
                elevation = map_data.heightmap[fy, fx]
                if 0.45 < elevation < 0.75 and map_data.land_mask[fy, fx]:
                    # Create forest patch
                    forest_size = random.randint(30, 60)
                    
                    for dy in range(-forest_size//2, forest_size//2):
                        for dx in range(-forest_size//2, forest_size//2):
                            x, y = fx + dx, fy + dy
                            if 0 <= x < map_data.grid_width and 0 <= y < map_data.grid_height:
                                distance = math.sqrt(dx**2 + dy**2)
                                if distance < forest_size//2:
                                    # Use noise to create irregular forest edge
                                    edge_noise = pnoise2(x * 0.1, y * 0.1, octaves=2)
                                    if distance + edge_noise * 10 < forest_size//2:
                                        if (self.biome_map[y, x] in ['plains', 'hills'] and 
                                            map_data.land_mask[y, x]):
                                            self.biome_map[y, x] = 'forest'
                    break


# Backward compatibility with existing system
class TerrainGenerator:
    """Legacy terrain generator - now uses advanced system internally."""
    
    def __init__(self):
        self.advanced_generator = AdvancedTerrainGenerator()
    
    def generate_heightmap(self, map_data: MapData, config: TerrainConfig):
        """Generate heightmap using advanced system."""
        self.advanced_generator.generate_heightmap(map_data, config)
    
    def define_land_and_water(self, map_data: MapData, config: TerrainConfig):
        """Define land and water using advanced system."""
        self.advanced_generator.define_land_and_water(map_data, config)
    
    def generate_water_bodies(self, map_data: MapData, config: WaterConfig):
        """Generate water bodies using advanced system."""
        self.advanced_generator.generate_water_bodies(map_data, config)
        
        # Also generate biomes
        self.advanced_generator.generate_biomes(map_data) 