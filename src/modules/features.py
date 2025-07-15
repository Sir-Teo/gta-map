"""
Features generation module.
Handles POIs (Points of Interest), landmarks, and parks generation.
"""
import random
import math
from typing import List, Tuple, Optional
import numpy as np
from shapely.geometry import Polygon, Point

from ..core.map_data import MapData, POI, Park
from ..config.settings import POIConfig, MapConfig


class FeaturesGenerator:
    """
    Generates POIs, landmarks, and parks throughout the city.
    """
    
    def generate_pois(self, map_data: MapData, config: POIConfig):
        """
        Generate Points of Interest (landmarks) across the map.
        
        Args:
            map_data: The map data container to populate
            config: POI generation configuration
        """
        poi_id = 0
        
        # Generate each type of POI
        for poi_type, properties in config.poi_types.items():
            # Determine how many of this POI type to generate
            if poi_type == 'airport':
                count = 1  # Usually only one airport
            elif poi_type in ['stadium', 'university', 'mall']:
                count = random.randint(1, 2)  # 1-2 of these
            elif poi_type in ['hospital']:
                count = random.randint(2, 4)  # More hospitals needed
            elif poi_type == 'port':
                count = 1 if self._has_water_access(map_data) else 0
            else:
                count = random.randint(1, 3)
            
            # Generate POIs of this type
            locations = self._find_poi_locations(map_data, poi_type, properties, count)
            
            for i, (x, y) in enumerate(locations):
                poi = POI(
                    id=f"{poi_type}_{poi_id}",
                    name=self._generate_poi_name(poi_type, i),
                    poi_type=poi_type,
                    x=x - properties['size'][0]/2,  # Center the POI
                    y=y - properties['size'][1]/2,
                    width=properties['size'][0],
                    height=properties['size'][1],
                    color=properties['color'],
                    properties=properties.copy()
                )
                
                map_data.add_poi(poi)
                poi_id += 1
        
        print(f"✓ {len(map_data.pois)} POIs generated.")
    
    def generate_parks(self, map_data: MapData, config: MapConfig):
        """
        Generate parks and green spaces throughout the city.
        
        Args:
            map_data: The map data container to populate
            config: Full map generation configuration
        """
        park_id = 0
        
        # Generate different types of parks
        park_types = [
            ('central_park', 1, (200, 300)),      # Large central park
            ('neighborhood', 3, (80, 150)),       # Medium neighborhood parks
            ('pocket', 5, (30, 60)),              # Small pocket parks
        ]
        
        for park_type, count, size_range in park_types:
            for i in range(count):
                park = self._generate_park(map_data, park_type, size_range, f"park_{park_id}")
                if park:
                    map_data.add_park(park)
                    park_id += 1
        
        print(f"✓ {len(map_data.parks)} parks generated.")
    
    def _find_poi_locations(
        self, 
        map_data: MapData, 
        poi_type: str, 
        properties: dict, 
        count: int
    ) -> List[Tuple[float, float]]:
        """
        Find suitable locations for POIs based on their requirements.
        
        Args:
            map_data: Map data for location finding
            poi_type: Type of POI to place
            properties: POI properties including requirements
            count: Number of locations needed
            
        Returns:
            List of (x, y) coordinates for POI placement
        """
        suitable_locations = []
        width, height = properties['size']
        
        # Get existing POI locations for buffer checking
        existing_pois = [(poi.x + poi.width/2, poi.y + poi.height/2) 
                        for poi in map_data.pois.values()]
        
        # Special handling for different POI types
        if poi_type == 'airport':
            # Airports need large flat areas away from city center
            suitable_locations = self._find_airport_locations(map_data, width, height)
        
        elif poi_type == 'port':
            # Ports need water access
            suitable_locations = self._find_port_locations(map_data, width, height)
        
        else:
            # Other POIs can be in urban areas
            suitable_locations = self._find_urban_poi_locations(
                map_data, poi_type, width, height
            )
        
        # Filter by minimum distance between POIs
        filtered_locations = []
        min_distance = properties['buffer']
        
        for loc in suitable_locations:
            too_close = False
            for existing in existing_pois + filtered_locations:
                dist = math.sqrt((loc[0] - existing[0])**2 + (loc[1] - existing[1])**2)
                if dist < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                filtered_locations.append(loc)
                if len(filtered_locations) >= count:
                    break
        
        return filtered_locations[:count]
    
    def _find_airport_locations(self, map_data: MapData, width: float, height: float) -> List[Tuple[float, float]]:
        """Find suitable locations for airports."""
        locations = []
        
        # Look for flat areas away from city center
        if map_data.heightmap is not None:
            gradient_x, gradient_y = np.gradient(map_data.heightmap)
            slope = np.sqrt(gradient_x**2 + gradient_y**2)
            flat_areas = np.argwhere((map_data.land_mask) & (slope < 0.01))
            
            center_x, center_y = map_data.width/2, map_data.height/2
            
            for grid_y, grid_x in flat_areas:
                x = grid_x * map_data.grid_size
                y = grid_y * map_data.grid_size
                
                # Check distance from center (prefer edges)
                dist_from_center = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist_from_center > min(map_data.width, map_data.height) * 0.3:
                    # Check if there's enough space
                    if (x + width < map_data.width and y + height < map_data.height):
                        locations.append((x + width/2, y + height/2))
        
        return locations[:5]  # Return at most 5 potential locations
    
    def _find_port_locations(self, map_data: MapData, width: float, height: float) -> List[Tuple[float, float]]:
        """Find suitable locations for ports near water."""
        locations = []
        
        # Find water edges
        for i in range(1, map_data.grid_height-1):
            for j in range(1, map_data.grid_width-1):
                if map_data.land_mask[i, j]:
                    # Check if adjacent to water
                    adjacent_to_water = (
                        not map_data.land_mask[i-1, j] or 
                        not map_data.land_mask[i+1, j] or
                        not map_data.land_mask[i, j-1] or 
                        not map_data.land_mask[i, j+1]
                    )
                    
                    if adjacent_to_water:
                        x = j * map_data.grid_size
                        y = i * map_data.grid_size
                        
                        # Check if there's enough space
                        if (x + width < map_data.width and y + height < map_data.height):
                            locations.append((x + width/2, y + height/2))
        
        return locations[:3]  # Return at most 3 potential locations
    
    def _find_urban_poi_locations(
        self, 
        map_data: MapData, 
        poi_type: str, 
        width: float, 
        height: float
    ) -> List[Tuple[float, float]]:
        """Find locations for urban POIs within city blocks."""
        locations = []
        
        # District preferences for different POIs
        district_preferences = {
            'stadium': ['downtown', 'commercial'],
            'hospital': ['residential', 'commercial'],
            'university': ['residential', 'suburban'],
            'mall': ['commercial', 'suburban']
        }
        
        preferred_districts = district_preferences.get(poi_type, ['residential', 'commercial'])
        
        # Look for suitable blocks
        for block in map_data.city_blocks.values():
            if block.district_type in preferred_districts:
                if block.polygon.area > width * height * 1.5:  # Ensure block is big enough
                    centroid = block.polygon.centroid
                    
                    # Check if on land
                    if self._is_on_land(centroid.x, centroid.y, map_data):
                        locations.append((centroid.x, centroid.y))
        
        return locations
    
    def _generate_park(
        self, 
        map_data: MapData, 
        park_type: str, 
        size_range: Tuple[float, float], 
        park_id: str
    ) -> Optional[Park]:
        """Generate a single park."""
        size = random.uniform(*size_range)
        
        # Find a suitable location
        max_attempts = 20
        for _ in range(max_attempts):
            # Random location
            center_x = random.uniform(size, map_data.width - size)
            center_y = random.uniform(size, map_data.height - size)
            
            # Check if on land
            if self._is_on_land(center_x, center_y, map_data):
                # Generate park shape (roughly circular with some variation)
                park_polygon = self._generate_park_shape(center_x, center_y, size)
                
                # Check for conflicts with existing features
                if not self._park_conflicts_with_features(park_polygon, map_data):
                    return Park(
                        id=park_id,
                        name=self._generate_park_name(park_type, park_id),
                        polygon=park_polygon,
                        park_type=park_type,
                        color='#33cc33'
                    )
        
        return None
    
    def _generate_park_shape(self, center_x: float, center_y: float, size: float) -> Polygon:
        """Generate an irregular park shape."""
        num_points = random.randint(8, 12)
        points = []
        
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            # Add some randomness to the radius
            radius = size * random.uniform(0.7, 1.3)
            x = center_x + math.cos(angle) * radius
            y = center_y + math.sin(angle) * radius
            points.append((x, y))
        
        # Close the polygon
        points.append(points[0])
        
        try:
            return Polygon(points)
        except:
            # Fallback to simple circle if irregular shape fails
            return Point(center_x, center_y).buffer(size)
    
    def _park_conflicts_with_features(self, park_polygon: Polygon, map_data: MapData) -> bool:
        """Check if park conflicts with existing features."""
        # Check conflicts with POIs
        for poi in map_data.pois.values():
            if park_polygon.intersects(poi.polygon):
                return True
        
        # Check conflicts with major buildings (simplified)
        # In a more complex system, we'd check against all buildings
        building_count = 0
        for building in map_data.buildings.values():
            if park_polygon.intersects(building.polygon):
                building_count += 1
                if building_count > 2:  # Allow small overlaps
                    return True
        
        return False
    
    def _has_water_access(self, map_data: MapData) -> bool:
        """Check if the map has water bodies for port placement."""
        return len(map_data.water_bodies) > 0 or not map_data.land_mask.all()
    
    def _is_on_land(self, x: float, y: float, map_data: MapData) -> bool:
        """Check if a point is on land."""
        grid_x = int(x / map_data.grid_size)
        grid_y = int(y / map_data.grid_size)
        
        if (0 <= grid_x < map_data.grid_width and 
            0 <= grid_y < map_data.grid_height):
            return map_data.land_mask[grid_y, grid_x]
        
        return False
    
    def _generate_poi_name(self, poi_type: str, index: int) -> str:
        """Generate names for POIs."""
        name_templates = {
            'airport': ['International Airport', 'Regional Airport', 'Metro Airport'],
            'stadium': ['City Stadium', 'Sports Complex', 'Arena'],
            'hospital': ['General Hospital', 'Medical Center', 'Community Hospital'],
            'university': ['State University', 'City College', 'Technical Institute'],
            'mall': ['Shopping Center', 'Plaza', 'Mall'],
            'port': ['Harbor', 'Port', 'Marine Terminal']
        }
        
        templates = name_templates.get(poi_type, [poi_type.title()])
        base_name = random.choice(templates)
        
        if index > 0:
            return f"{base_name} {index + 1}"
        return base_name
    
    def _generate_park_name(self, park_type: str, park_id: str) -> str:
        """Generate names for parks."""
        park_names = {
            'central_park': ['Central Park', 'City Park', 'Grand Park'],
            'neighborhood': ['Community Park', 'Neighborhood Green', 'Local Park'],
            'pocket': ['Pocket Park', 'Mini Green', 'Small Park']
        }
        
        templates = park_names.get(park_type, ['Park'])
        base_name = random.choice(templates)
        
        # Add some variety to the names
        if random.random() < 0.3:
            suffixes = ['Gardens', 'Commons', 'Square', 'Green']
            base_name += f" {random.choice(suffixes)}"
        
        return base_name 