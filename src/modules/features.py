"""
Enhanced features generation module.
Handles natural features, parks, points of interest, and biome-aware feature placement.
"""
import random
import math
from typing import List, Tuple, Optional, Dict
import numpy as np
from noise import pnoise2
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union

from ..core.map_data import MapData, POI, Park
from ..config.settings import POIConfig, BiomeConfig


class AdvancedFeaturesGenerator:
    """
    Advanced features generator that creates natural biomes and realistic feature placement.
    """
    
    def __init__(self):
        self.national_parks = []
        self.agricultural_zones = []
        self.forest_preserves = []
        self.natural_landmarks = []
    
    def generate_parks(self, map_data: MapData, config):
        """
        Generate parks and green spaces throughout the map, including national parks.
        """
        print("  → Generating parks and green spaces...")
        
        # Generate national parks in mountainous/forested areas
        self._generate_national_parks(map_data, config)
        
        # Generate urban parks in cities
        self._generate_urban_parks(map_data, config)
        
        # Generate smaller neighborhood parks
        self._generate_neighborhood_parks(map_data, config)
        
        print(f"  → Created {len(map_data.parks)} parks and green spaces")
    
    def generate_pois(self, map_data: MapData, config: POIConfig):
        """
        Generate points of interest with biome and settlement awareness.
        """
        print("  → Placing points of interest...")
        
        # Generate settlement-based POIs
        self._generate_settlement_pois(map_data, config)
        
        # Generate natural landmarks
        self._generate_natural_landmarks(map_data, config)
        
        # Generate transportation hubs
        self._generate_transportation_hubs(map_data, config)
        
        print(f"  → Placed {len(map_data.pois)} points of interest")
    
    def generate_agricultural_zones(self, map_data: MapData, biome_config: BiomeConfig):
        """
        Generate agricultural areas in suitable locations.
        """
        print("  → Creating agricultural zones...")
        
        if not biome_config.enable_agricultural_areas:
            return
        
        # Find suitable areas for agriculture (flat, good water access, not in cities)
        suitable_areas = self._find_agricultural_areas(map_data)
        
        # Create agricultural zones
        target_coverage = biome_config.agricultural_coverage
        map_area = map_data.width * map_data.height
        target_area = map_area * target_coverage
        
        current_area = 0
        agricultural_id = 0
        
        for area_center, suitability in suitable_areas:
            if current_area >= target_area:
                break
            
            # Create agricultural zone
            zone_size = random.uniform(300, 800) * suitability
            agricultural_zone = self._create_agricultural_zone(
                map_data, area_center, zone_size, f"agricultural_zone_{agricultural_id}"
            )
            
            if agricultural_zone:
                map_data.add_park(agricultural_zone)  # Agricultural zones stored as special parks
                self.agricultural_zones.append(agricultural_zone)
                current_area += agricultural_zone.polygon.area
                agricultural_id += 1
        
        print(f"  → Created {len(self.agricultural_zones)} agricultural zones")
    
    def generate_natural_preserves(self, map_data: MapData, biome_config: BiomeConfig):
        """
        Generate nature preserves and protected areas.
        """
        print("  → Establishing nature preserves...")
        
        # Create forest preserves
        self._create_forest_preserves(map_data, biome_config)
        
        # Create wetland preserves
        self._create_wetland_preserves(map_data, biome_config)
        
        # Create mountain preserves
        self._create_mountain_preserves(map_data, biome_config)
        
        print(f"  → Established {len(self.forest_preserves)} nature preserves")
    
    def _generate_national_parks(self, map_data: MapData, config):
        """Generate large national parks in scenic areas."""
        
        # Find areas suitable for national parks (mountains, forests, unique terrain)
        park_locations = self._find_national_park_locations(map_data)
        
        # Create 1-3 national parks depending on map size
        map_area = map_data.width * map_data.height
        num_parks = max(1, min(3, int(map_area / (1500 * 1500))))
        
        for i in range(min(num_parks, len(park_locations))):
            location, terrain_type = park_locations[i]
            
            # Create large national park
            park = self._create_national_park(
                map_data, location, terrain_type, f"national_park_{i}"
            )
            
            if park:
                map_data.add_park(park)
                self.national_parks.append(park)
    
    def _generate_urban_parks(self, map_data: MapData, config):
        """Generate parks within urban areas."""
        
        urban_park_id = 0
        
        # Generate parks in each major district
        for district in map_data.districts.values():
            if district.district_type in ['downtown', 'commercial', 'residential']:
                
                # Chance of having a park in this district
                park_chance = {
                    'downtown': 0.8,
                    'commercial': 0.6,
                    'residential': 0.9
                }.get(district.district_type, 0.5)
                
                if random.random() < park_chance:
                    park = self._create_urban_park(
                        map_data, district, f"urban_park_{urban_park_id}"
                    )
                    
                    if park:
                        map_data.add_park(park)
                        urban_park_id += 1
    
    def _generate_neighborhood_parks(self, map_data: MapData, config):
        """Generate small neighborhood parks."""
        
        # Generate scattered small parks
        num_small_parks = random.randint(5, 15)
        neighborhood_park_id = 0
        
        for _ in range(num_small_parks):
            # Find a random location not in water or too close to existing parks
            location = self._find_small_park_location(map_data)
            
            if location:
                park = self._create_neighborhood_park(
                    map_data, location, f"neighborhood_park_{neighborhood_park_id}"
                )
                
                if park:
                    map_data.add_park(park)
                    neighborhood_park_id += 1
    
    def _generate_settlement_pois(self, map_data: MapData, config: POIConfig):
        """Generate POIs based on settlement locations and types."""
        
        poi_id = 0
        
        for district in map_data.districts.values():
            district_pois = self._get_district_appropriate_pois(district.district_type)
            
            # Generate POIs for this district
            for poi_type in district_pois:
                if random.random() < district_pois[poi_type]['probability']:
                    
                    poi_location = self._find_poi_location_in_district(
                        map_data, district, poi_type, config
                    )
                    
                    if poi_location:
                        poi = self._create_poi(
                            poi_location, poi_type, f"{poi_type}_{poi_id}", config
                        )
                        
                        if poi:
                            map_data.add_poi(poi)
                            poi_id += 1
    
    def _generate_natural_landmarks(self, map_data: MapData, config: POIConfig):
        """Generate natural landmarks like scenic viewpoints, waterfalls, etc."""
        
        landmark_id = 0
        
        # Generate mountain viewpoints
        viewpoints = self._find_scenic_viewpoints(map_data)
        for viewpoint in viewpoints[:3]:  # Limit to 3 viewpoints
            poi = POI(
                id=f"viewpoint_{landmark_id}",
                name=f"Scenic Overlook {landmark_id + 1}",
                poi_type='viewpoint',
                x=viewpoint[0] - 25,
                y=viewpoint[1] - 25,
                width=50,
                height=50,
                color='#8B7D6B',
                properties={'elevation': 'high', 'scenic': True}
            )
            map_data.add_poi(poi)
            self.natural_landmarks.append(poi)
            landmark_id += 1
        
        # Generate waterfalls near rivers
        waterfalls = self._find_waterfall_locations(map_data)
        for waterfall in waterfalls[:2]:  # Limit to 2 waterfalls
            poi = POI(
                id=f"waterfall_{landmark_id}",
                name=f"Waterfall {landmark_id + 1}",
                poi_type='waterfall',
                x=waterfall[0] - 30,
                y=waterfall[1] - 30,
                width=60,
                height=60,
                color='#4682B4',
                properties={'natural': True, 'water_feature': True}
            )
            map_data.add_poi(poi)
            self.natural_landmarks.append(poi)
            landmark_id += 1
    
    def _generate_transportation_hubs(self, map_data: MapData, config: POIConfig):
        """Generate transportation-related POIs."""
        
        # Generate airports
        airport_districts = [d for d in map_data.districts.values() if d.district_type == 'airport']
        for i, district in enumerate(airport_districts):
            airport_poi = POI(
                id=f"airport_{i}",
                name=f"Airport {i + 1}",
                poi_type='airport',
                x=district.center[0] - 75,
                y=district.center[1] - 75,
                width=150,
                height=150,
                color='#CC33CC',
                properties={'transportation': True, 'international': i == 0}
            )
            map_data.add_poi(airport_poi)
        
        # Generate ports
        port_districts = [d for d in map_data.districts.values() if d.district_type == 'port']
        for i, district in enumerate(port_districts):
            port_poi = POI(
                id=f"port_{i}",
                name=f"Port {i + 1}",
                poi_type='port',
                x=district.center[0] - 60,
                y=district.center[1] - 60,
                width=120,
                height=120,
                color='#3366CC',
                properties={'transportation': True, 'maritime': True}
            )
            map_data.add_poi(port_poi)
    
    def _find_agricultural_areas(self, map_data: MapData) -> List[Tuple[Tuple[float, float], float]]:
        """Find areas suitable for agriculture."""
        suitable_areas = []
        
        # Sample potential locations
        for _ in range(100):  # Try 100 random locations
            x = random.randint(20, map_data.grid_width - 20)
            y = random.randint(20, map_data.grid_height - 20)
            
            if not map_data.land_mask[y, x]:
                continue
            
            elevation = map_data.heightmap[y, x]
            
            # Check agricultural suitability
            suitability = self._calculate_agricultural_suitability(map_data, x, y, elevation)
            
            if suitability > 0.4:  # Minimum threshold
                map_x = x * map_data.grid_size
                map_y = y * map_data.grid_size
                suitable_areas.append(((map_x, map_y), suitability))
        
        # Sort by suitability
        suitable_areas.sort(key=lambda x: x[1], reverse=True)
        return suitable_areas
    
    def _calculate_agricultural_suitability(self, map_data: MapData, x: int, y: int, elevation: float) -> float:
        """Calculate agricultural suitability for a location."""
        score = 0.0
        
        # Prefer moderate elevations
        if 0.4 < elevation < 0.65:
            score += 0.4
        elif 0.35 < elevation < 0.75:
            score += 0.2
        else:
            score -= 0.2
        
        # Prefer flat areas
        flatness = self._calculate_area_flatness(map_data, x, y, radius=3)
        score += flatness * 0.3
        
        # Prefer areas near water but not too close
        water_distance = self._calculate_water_distance(map_data, x, y)
        if 3 < water_distance < 10:
            score += 0.3
        elif water_distance <= 3:
            score += 0.1  # Too close to water
        
        # Avoid urban areas
        too_close_to_urban = self._check_urban_proximity(map_data, x, y)
        if too_close_to_urban:
            score -= 0.5
        
        # Prefer existing plains/agricultural biomes if available
        if hasattr(map_data, 'biome_map') and map_data.biome_map is not None:
            if 0 <= y < map_data.biome_map.shape[0] and 0 <= x < map_data.biome_map.shape[1]:
                biome = map_data.biome_map[y, x]
                if biome in ['plains', 'agricultural', 'coastal_plains']:
                    score += 0.3
        
        return max(0.0, min(1.0, score))
    
    def _calculate_area_flatness(self, map_data: MapData, x: int, y: int, radius: int) -> float:
        """Calculate flatness of area around a point."""
        elevations = []
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < map_data.grid_width and 0 <= ny < map_data.grid_height:
                    elevations.append(map_data.heightmap[ny, nx])
        
        if not elevations:
            return 0.0
        
        # Calculate standard deviation (lower = flatter)
        mean_elevation = sum(elevations) / len(elevations)
        variance = sum((e - mean_elevation)**2 for e in elevations) / len(elevations)
        std_dev = math.sqrt(variance)
        
        # Convert to flatness score
        flatness = max(0.0, 1.0 - std_dev * 8)
        return flatness
    
    def _calculate_water_distance(self, map_data: MapData, x: int, y: int) -> float:
        """Calculate distance to nearest water."""
        min_dist = float('inf')
        
        for dy in range(-15, 16):
            for dx in range(-15, 16):
                nx, ny = x + dx, y + dy
                if 0 <= nx < map_data.grid_width and 0 <= ny < map_data.grid_height:
                    if not map_data.land_mask[ny, nx]:
                        dist = math.sqrt(dx**2 + dy**2)
                        min_dist = min(min_dist, dist)
        
        return min_dist if min_dist != float('inf') else 20
    
    def _check_urban_proximity(self, map_data: MapData, x: int, y: int) -> bool:
        """Check if location is too close to urban areas."""
        map_x = x * map_data.grid_size
        map_y = y * map_data.grid_size
        
        for district in map_data.districts.values():
            if district.district_type in ['downtown', 'commercial', 'residential', 'industrial']:
                distance = math.sqrt(
                    (map_x - district.center[0])**2 + (map_y - district.center[1])**2
                )
                if distance < 400:  # Too close to urban area
                    return True
        
        return False
    
    def _create_agricultural_zone(self, map_data: MapData, center: Tuple[float, float], size: float, zone_id: str) -> Optional[Park]:
        """Create an agricultural zone."""
        
        # Create agricultural area shape (more rectangular than circular)
        center_x, center_y = center
        
        # Agricultural areas tend to be more geometric
        width = size * random.uniform(0.8, 1.2)
        height = size * random.uniform(0.6, 1.0)
        
        # Create rectangular agricultural area with some noise
        vertices = []
        corners = [
            (center_x - width/2, center_y - height/2),
            (center_x + width/2, center_y - height/2),
            (center_x + width/2, center_y + height/2),
            (center_x - width/2, center_y + height/2)
        ]
        
        for i, (x, y) in enumerate(corners):
            # Add small variations to make it more natural
            noise_x = pnoise2(x * 0.01, y * 0.01, octaves=2) * 50
            noise_y = pnoise2(x * 0.01 + 100, y * 0.01 + 100, octaves=2) * 50
            
            vertices.append((x + noise_x, y + noise_y))
        
        try:
            polygon = Polygon(vertices)
            if polygon.is_valid and polygon.area > 1000:
                return Park(
                    id=zone_id,
                    name=f"Agricultural Zone {zone_id.split('_')[-1]}",
                    polygon=polygon,
                    park_type="agricultural",
                    color="#DAA520"
                )
        except Exception:
            pass
        
        return None
    
    def _find_national_park_locations(self, map_data: MapData) -> List[Tuple[Tuple[float, float], str]]:
        """Find suitable locations for national parks."""
        locations = []
        
        # Look for mountainous areas
        for y in range(10, map_data.grid_height - 10, 8):
            for x in range(10, map_data.grid_width - 10, 8):
                
                if not map_data.land_mask[y, x]:
                    continue
                
                elevation = map_data.heightmap[y, x]
                
                # National parks prefer interesting terrain
                if elevation > 0.6:  # Mountainous
                    # Check if area is suitable (large, undeveloped)
                    if self._check_national_park_suitability(map_data, x, y):
                        map_x = x * map_data.grid_size
                        map_y = y * map_data.grid_size
                        
                        terrain_type = 'mountain' if elevation > 0.75 else 'forest'
                        locations.append(((map_x, map_y), terrain_type))
        
        # Sort by elevation (prefer higher areas)
        locations.sort(key=lambda loc: self._get_elevation_at_point(map_data, loc[0]), reverse=True)
        return locations
    
    def _check_national_park_suitability(self, map_data: MapData, x: int, y: int) -> bool:
        """Check if area is suitable for a national park."""
        
        # Check if far enough from urban areas
        map_x = x * map_data.grid_size
        map_y = y * map_data.grid_size
        
        for district in map_data.districts.values():
            distance = math.sqrt(
                (map_x - district.center[0])**2 + (map_y - district.center[1])**2
            )
            if distance < 600:  # Too close to urban development
                return False
        
        # Check if area is large enough and has varied terrain
        terrain_variety = 0
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                nx, ny = x + dx, y + dy
                if 0 <= nx < map_data.grid_width and 0 <= ny < map_data.grid_height:
                    if map_data.land_mask[ny, nx]:
                        terrain_variety += 1
        
        return terrain_variety > 50  # Sufficient land area
    
    def _get_elevation_at_point(self, map_data: MapData, point: Tuple[float, float]) -> float:
        """Get elevation at a specific map point."""
        x, y = point
        grid_x = int(x / map_data.grid_size)
        grid_y = int(y / map_data.grid_size)
        
        if 0 <= grid_x < map_data.grid_width and 0 <= grid_y < map_data.grid_height:
            return map_data.heightmap[grid_y, grid_x]
        
        return 0.0
    
    def _create_national_park(self, map_data: MapData, location: Tuple[float, float], terrain_type: str, park_id: str) -> Optional[Park]:
        """Create a large national park."""
        
        center_x, center_y = location
        
        # National parks are large and irregular
        base_radius = random.uniform(400, 800)
        
        # Generate organic park boundary
        vertices = []
        num_points = random.randint(12, 20)
        
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            
            # Vary radius for organic shape
            radius_variation = pnoise2(
                math.cos(angle) * 3, math.sin(angle) * 3, octaves=3
            ) * 0.4 + 1.0
            
            radius = base_radius * radius_variation
            
            x = center_x + math.cos(angle) * radius
            y = center_y + math.sin(angle) * radius
            
            vertices.append((x, y))
        
        try:
            polygon = Polygon(vertices)
            if polygon.is_valid and polygon.area > 50000:
                
                park_name = f"{terrain_type.title()} National Park"
                park_color = '#228B22' if terrain_type == 'forest' else '#8FBC8F'
                
                return Park(
                    id=park_id,
                    name=park_name,
                    polygon=polygon,
                    park_type="national",
                    color=park_color
                )
        except Exception:
            pass
        
        return None
    
    def _create_urban_park(self, map_data: MapData, district, park_id: str) -> Optional[Park]:
        """Create a park within an urban district."""
        
        if not district.polygon:
            return None
        
        # Urban parks are smaller and more regular
        center_x, center_y = district.center
        
        # Offset from district center
        offset_x = random.uniform(-district.radius * 0.3, district.radius * 0.3)
        offset_y = random.uniform(-district.radius * 0.3, district.radius * 0.3)
        
        park_center_x = center_x + offset_x
        park_center_y = center_y + offset_y
        
        # Create park shape
        park_size = random.uniform(80, 150)
        vertices = []
        
        # Urban parks tend to be more geometric
        num_points = random.randint(6, 8)
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            
            radius_variation = random.uniform(0.8, 1.2)
            radius = park_size * radius_variation
            
            x = park_center_x + math.cos(angle) * radius
            y = park_center_y + math.sin(angle) * radius
            
            vertices.append((x, y))
        
        try:
            polygon = Polygon(vertices)
            if polygon.is_valid and polygon.area > 2000:
                return Park(
                    id=park_id,
                    name=f"City Park {park_id.split('_')[-1]}",
                    polygon=polygon,
                    park_type="urban",
                    color="#32CD32"
                )
        except Exception:
            pass
        
        return None
    
    def _find_small_park_location(self, map_data: MapData) -> Optional[Tuple[float, float]]:
        """Find a location for a small neighborhood park."""
        
        for _ in range(20):  # Try 20 random locations
            x = random.randint(50, map_data.width - 50)
            y = random.randint(50, map_data.height - 50)
            
            grid_x = int(x / map_data.grid_size)
            grid_y = int(y / map_data.grid_size)
            
            # Check if on land
            if (0 <= grid_x < map_data.grid_width and 
                0 <= grid_y < map_data.grid_height and
                map_data.land_mask[grid_y, grid_x]):
                
                # Check if not too close to existing parks
                too_close = False
                for park in map_data.parks.values():
                    park_center = park.center
                    distance = math.sqrt((x - park_center[0])**2 + (y - park_center[1])**2)
                    if distance < 200:
                        too_close = True
                        break
                
                if not too_close:
                    return (x, y)
        
        return None
    
    def _create_neighborhood_park(self, map_data: MapData, location: Tuple[float, float], park_id: str) -> Optional[Park]:
        """Create a small neighborhood park."""
        
        center_x, center_y = location
        park_size = random.uniform(40, 80)
        
        # Simple circular/oval shape for neighborhood parks
        vertices = []
        num_points = 8
        
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            
            radius_variation = random.uniform(0.8, 1.2)
            radius = park_size * radius_variation
            
            x = center_x + math.cos(angle) * radius
            y = center_y + math.sin(angle) * radius
            
            vertices.append((x, y))
        
        try:
            polygon = Polygon(vertices)
            if polygon.is_valid:
                return Park(
                    id=park_id,
                    name=f"Neighborhood Park {park_id.split('_')[-1]}",
                    polygon=polygon,
                    park_type="neighborhood",
                    color="#90EE90"
                )
        except Exception:
            pass
        
        return None
    
    def _get_district_appropriate_pois(self, district_type: str) -> Dict[str, Dict]:
        """Get POI types appropriate for each district type."""
        
        poi_types = {
            'downtown': {
                'hospital': {'probability': 0.8},
                'university': {'probability': 0.6},
                'mall': {'probability': 0.7}
            },
            'commercial': {
                'mall': {'probability': 0.9},
                'hospital': {'probability': 0.4}
            },
            'industrial': {
                'hospital': {'probability': 0.3}
            },
            'airport': {
                'airport': {'probability': 1.0}
            },
            'port': {
                'port': {'probability': 1.0}
            }
        }
        
        return poi_types.get(district_type, {})
    
    def _find_poi_location_in_district(self, map_data: MapData, district, poi_type: str, config: POIConfig) -> Optional[Tuple[float, float]]:
        """Find a suitable location for a POI within a district."""
        
        if not district.polygon:
            return None
        
        # Try to find a suitable location within the district
        for _ in range(10):
            # Random point within district bounds
            minx, miny, maxx, maxy = district.polygon.bounds
            
            x = random.uniform(minx, maxx)
            y = random.uniform(miny, maxy)
            
            if district.polygon.contains(Point(x, y)):
                # Check if location meets POI requirements
                if self._check_poi_requirements(map_data, (x, y), poi_type, config):
                    return (x, y)
        
        return None
    
    def _check_poi_requirements(self, map_data: MapData, location: Tuple[float, float], poi_type: str, config: POIConfig) -> bool:
        """Check if location meets requirements for POI type."""
        
        x, y = location
        grid_x = int(x / map_data.grid_size)
        grid_y = int(y / map_data.grid_size)
        
        # Check if on land
        if (not (0 <= grid_x < map_data.grid_width and 
                 0 <= grid_y < map_data.grid_height) or
            not map_data.land_mask[grid_y, grid_x]):
            return False
        
        # Check POI-specific requirements
        poi_config = config.poi_types.get(poi_type, {})
        
        if poi_config.get('requires_flat', False):
            # Check if area is flat enough
            elevation = map_data.heightmap[grid_y, grid_x]
            if elevation > 0.7:  # Too high/steep
                return False
        
        if poi_config.get('requires_water', False):
            # Check if near water
            water_distance = self._calculate_water_distance(map_data, grid_x, grid_y)
            if water_distance > 5:  # Too far from water
                return False
        
        return True
    
    def _create_poi(self, location: Tuple[float, float], poi_type: str, poi_id: str, config: POIConfig) -> Optional[POI]:
        """Create a POI at the specified location."""
        
        poi_config = config.poi_types.get(poi_type, {})
        size = poi_config.get('size', (50, 50))
        color = poi_config.get('color', '#CC33CC')
        
        width, height = size
        if isinstance(size[0], tuple):  # Size range
            width = random.uniform(size[0][0], size[0][1])
            height = random.uniform(size[1][0], size[1][1])
        
        x, y = location
        
        return POI(
            id=poi_id,
            name=f"{poi_type.title()} {poi_id.split('_')[-1]}",
            poi_type=poi_type,
            x=x - width/2,
            y=y - height/2,
            width=width,
            height=height,
            color=color,
            properties={'generated': True}
        )
    
    def _find_scenic_viewpoints(self, map_data: MapData) -> List[Tuple[float, float]]:
        """Find locations suitable for scenic viewpoints."""
        viewpoints = []
        
        # Look for high elevation areas with good views
        for y in range(5, map_data.grid_height - 5, 10):
            for x in range(5, map_data.grid_width - 5, 10):
                
                if not map_data.land_mask[y, x]:
                    continue
                
                elevation = map_data.heightmap[y, x]
                
                if elevation > 0.7:  # High elevation
                    # Check if it's higher than surrounding area
                    surrounding_avg = 0
                    count = 0
                    
                    for dy in range(-5, 6):
                        for dx in range(-5, 6):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < map_data.grid_width and 0 <= ny < map_data.grid_height:
                                surrounding_avg += map_data.heightmap[ny, nx]
                                count += 1
                    
                    if count > 0:
                        surrounding_avg /= count
                        
                        if elevation > surrounding_avg + 0.1:  # Significantly higher
                            map_x = x * map_data.grid_size
                            map_y = y * map_data.grid_size
                            viewpoints.append((map_x, map_y))
        
        return viewpoints
    
    def _find_waterfall_locations(self, map_data: MapData) -> List[Tuple[float, float]]:
        """Find locations suitable for waterfalls (near rivers with elevation changes)."""
        waterfalls = []
        
        # Look for rivers with significant elevation changes
        for water_body in map_data.water_bodies.values():
            if water_body.water_type == 'river' and hasattr(water_body.geometry, 'coords'):
                coords = list(water_body.geometry.coords)
                
                # Check elevation changes along the river
                for i in range(len(coords) - 5):
                    point1 = coords[i]
                    point2 = coords[i + 5]
                    
                    # Get elevations
                    grid_x1 = int(point1[0] / map_data.grid_size)
                    grid_y1 = int(point1[1] / map_data.grid_size)
                    grid_x2 = int(point2[0] / map_data.grid_size)
                    grid_y2 = int(point2[1] / map_data.grid_size)
                    
                    if (0 <= grid_x1 < map_data.grid_width and 0 <= grid_y1 < map_data.grid_height and
                        0 <= grid_x2 < map_data.grid_width and 0 <= grid_y2 < map_data.grid_height):
                        
                        elev1 = map_data.heightmap[grid_y1, grid_x1]
                        elev2 = map_data.heightmap[grid_y2, grid_x2]
                        
                        # Look for significant elevation drop
                        if elev1 > elev2 + 0.1:  # Waterfall opportunity
                            waterfalls.append(point1)
                            break  # One waterfall per river
        
        return waterfalls
    
    def _create_forest_preserves(self, map_data: MapData, biome_config: BiomeConfig):
        """Create forest nature preserves."""
        # Implementation for forest preserves
        pass
    
    def _create_wetland_preserves(self, map_data: MapData, biome_config: BiomeConfig):
        """Create wetland preserves."""
        # Implementation for wetland preserves
        pass
    
    def _create_mountain_preserves(self, map_data: MapData, biome_config: BiomeConfig):
        """Create mountain preserves."""
        # Implementation for mountain preserves
        pass


# Backward compatibility with existing system
class FeaturesGenerator:
    """Legacy features generator - now uses advanced system internally."""
    
    def __init__(self):
        self.advanced_generator = AdvancedFeaturesGenerator()
    
    def generate_parks(self, map_data: MapData, config):
        """Generate parks using advanced system."""
        self.advanced_generator.generate_parks(map_data, config)
        
        # Also generate agricultural zones and natural preserves
        if hasattr(config, 'biomes'):
            self.advanced_generator.generate_agricultural_zones(map_data, config.biomes)
            self.advanced_generator.generate_natural_preserves(map_data, config.biomes)
    
    def generate_pois(self, map_data: MapData, config: POIConfig):
        """Generate POIs using advanced system."""
        self.advanced_generator.generate_pois(map_data, config) 