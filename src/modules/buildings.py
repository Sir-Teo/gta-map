"""
Building generation module.
Handles building placement, sizing, and distribution within city blocks.
"""
import random
from typing import List, Tuple, Optional
from shapely.geometry import Polygon, Point

from ..core.map_data import MapData, Building
from ..config.settings import BuildingConfig


class BuildingGenerator:
    """
    Generates buildings within city blocks based on district types and density.
    """
    
    def populate_city_blocks(self, map_data: MapData, config: BuildingConfig):
        """
        Populate city blocks with buildings based on district type and density.
        
        Args:
            map_data: The map data container to populate
            config: Building generation configuration
        """
        building_id = 0
        
        for block in map_data.city_blocks.values():
            # Get density for this district type
            density = config.density_map.get(block.district_type, config.density_map.get('default', 0.0005))
            
            # Calculate number of buildings based on block area and density
            num_buildings = max(1, int(block.polygon.area * density))
            
            # Get size ranges for this district type
            size_config = config.size_ranges.get(block.district_type, config.size_ranges.get('default', {'width': (10, 30), 'length': (10, 30)}))
            
            # Generate buildings for this block
            buildings_placed = 0
            attempts = 0
            max_attempts = num_buildings * 3  # Prevent infinite loops
            
            while buildings_placed < num_buildings and attempts < max_attempts:
                building = self._generate_building_in_block(
                    block, 
                    f"building_{building_id}", 
                    size_config,
                    map_data
                )
                
                if building:
                    # Check for conflicts with existing buildings
                    if not self._conflicts_with_existing(building, map_data):
                        map_data.add_building(building)
                        block.buildings.append(building)
                        buildings_placed += 1
                        building_id += 1
                
                attempts += 1
        
        print(f"✓ {len(map_data.buildings)} buildings placed across {len(map_data.city_blocks)} city blocks.")
    
    def _generate_building_in_block(
        self,
        block,
        building_id: str,
        size_config: dict,
        map_data: MapData
    ) -> Optional[Building]:
        """
        Generate a single building within a city block.
        
        Args:
            block: The city block to place the building in
            building_id: Unique identifier for the building
            size_config: Size configuration for this district type
            map_data: Map data for validation
            
        Returns:
            Building or None if placement failed
        """
        polygon = block.polygon
        min_x, min_y, max_x, max_y = polygon.bounds
        
        # Generate building dimensions based on district type
        width = random.uniform(*size_config['width'])
        length = random.uniform(*size_config['length'])
        
        # Try to place building within block bounds
        max_attempts = 20
        for _ in range(max_attempts):
            # Random position within block bounds, with some margin
            margin = max(width, length) / 2
            x = random.uniform(min_x + margin, max_x - width - margin)
            y = random.uniform(min_y + margin, max_y - length - margin)
            
            # Check if building center is within the block polygon
            building_center = Point(x + width/2, y + length/2)
            if polygon.contains(building_center):
                # Validate that building is on land
                if self._is_on_land(x + width/2, y + length/2, map_data):
                    # Generate building height based on district type
                    height = self._generate_building_height(block.district_type)
                    
                    return Building(
                        id=building_id,
                        x=x,
                        y=y,
                        width=width,
                        length=length,
                        building_type=self._get_building_type(block.district_type),
                        district_type=block.district_type,
                        height=height
                    )
        
        return None
    
    def _conflicts_with_existing(self, new_building: Building, map_data: MapData) -> bool:
        """
        Check if a new building conflicts with existing buildings.
        
        Args:
            new_building: The building to check
            map_data: Map data containing existing buildings
            
        Returns:
            True if there's a conflict, False otherwise
        """
        new_poly = new_building.polygon
        
        # Add small buffer to prevent buildings from touching
        buffer_distance = 5.0
        buffered_poly = new_poly.buffer(buffer_distance)
        
        for existing_building in map_data.buildings.values():
            if buffered_poly.intersects(existing_building.polygon):
                return True
        
        return False
    
    def _is_on_land(self, x: float, y: float, map_data: MapData) -> bool:
        """
        Check if a point is on land (not water).
        
        Args:
            x, y: Coordinates to check
            map_data: Map data containing land mask
            
        Returns:
            True if on land, False if on water
        """
        # Convert to grid coordinates
        grid_x = int(x / map_data.grid_size)
        grid_y = int(y / map_data.grid_size)
        
        # Check bounds
        if (0 <= grid_x < map_data.grid_width and 
            0 <= grid_y < map_data.grid_height):
            return map_data.land_mask[grid_y, grid_x]
        
        return False
    
    def _generate_building_height(self, district_type: str) -> float:
        """
        Generate building height based on district type.
        
        Args:
            district_type: Type of district
            
        Returns:
            Building height in meters
        """
        height_ranges = {
            'downtown': (30, 150),      # Skyscrapers
            'commercial': (15, 50),     # Mid-rise commercial
            'residential': (8, 25),     # Apartments and houses
            'industrial': (10, 30),     # Industrial buildings
            'suburban': (5, 15),        # Houses
            'airport': (15, 40),        # Airport terminals
            'port': (10, 25),          # Port facilities
        }
        
        height_range = height_ranges.get(district_type, (8, 20))
        return random.uniform(*height_range)
    
    def _get_building_type(self, district_type: str) -> str:
        """
        Determine building type based on district type.
        
        Args:
            district_type: Type of district
            
        Returns:
            Building type string
        """
        building_types = {
            'downtown': ['office', 'mixed_use', 'hotel', 'retail'],
            'commercial': ['retail', 'office', 'restaurant', 'service'],
            'residential': ['apartment', 'house', 'townhouse', 'condo'],
            'industrial': ['factory', 'warehouse', 'processing', 'logistics'],
            'suburban': ['house', 'townhouse', 'local_retail'],
            'airport': ['terminal', 'hangar', 'cargo', 'service'],
            'port': ['warehouse', 'terminal', 'crane', 'office'],
            'hills': ['house', 'luxury_home'],
            'beach': ['hotel', 'restaurant', 'house'],
            'park': ['visitor_center', 'maintenance'],
        }
        
        available_types = building_types.get(district_type, ['generic'])
        return random.choice(available_types) 