"""
Urban planning module.
Handles city block definition, zoning, and urban structure.
"""
from typing import List, Dict, Any
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import polygonize

from ..core.map_data import MapData, CityBlock
from ..config.settings import MapConfig


class UrbanPlanner:
    """
    Handles urban planning including city block definition and zoning.
    """
    
    def define_city_blocks(self, map_data: MapData, config: MapConfig):
        """
        Detect city blocks as polygons enclosed by roads.
        
        Args:
            map_data: The map data container to populate
            config: Full map generation configuration
        """
        # Collect all road linestrings
        all_lines = []
        for road in map_data.roads.values():
            if len(road.points) > 1:
                try:
                    linestring = LineString(road.points)
                    if linestring.is_valid:
                        all_lines.append(linestring)
                except Exception:
                    continue  # Skip invalid linestrings

        if not all_lines:
            print("No valid roads found to form city blocks.")
            return

        # Use polygonize to find closed loops in the road network
        try:
            polygons = list(polygonize(all_lines))
            print(f"✓ {len(polygons)} potential city blocks found.")
            
            # If no polygons found, create some basic rectangular blocks
            if len(polygons) == 0:
                print("No polygons from roads, creating basic city blocks...")
                polygons = self._create_basic_city_blocks(map_data)
                print(f"✓ Created {len(polygons)} basic city blocks.")
                
        except Exception as e:
            print(f"Error in polygonize: {e}")
            print("Creating basic city blocks as fallback...")
            polygons = self._create_basic_city_blocks(map_data)
            print(f"✓ Created {len(polygons)} basic city blocks.")

        # Create district lookup for assigning block types
        district_lookup = self._create_district_lookup(map_data)
        
        # Process valid polygons and create city blocks
        block_id = 0
        for poly in polygons:
            if isinstance(poly, Polygon) and poly.is_valid and poly.area > 1000:  # Minimum block size
                # Determine district type for this block
                district_type = self._determine_block_district_type(poly, district_lookup)
                
                # Create city block
                city_block = CityBlock(
                    id=f"block_{block_id}",
                    polygon=poly,
                    district_type=district_type
                )
                
                map_data.add_city_block(city_block)
                block_id += 1
        
        print(f"✓ {len(map_data.city_blocks)} valid city blocks defined.")
    
    def plan_zoning(self, map_data: MapData, config: MapConfig):
        """
        Plan zoning and urban structure based on districts and blocks.
        
        Args:
            map_data: The map data container
            config: Full map generation configuration
        """
        # This method can be extended to:
        # - Refine block classifications
        # - Plan utility corridors
        # - Define building height restrictions
        # - Plan green space requirements
        # - etc.
        
        # For now, we'll do basic zoning validation
        self._validate_zoning(map_data, config)
    
    def _create_district_lookup(self, map_data: MapData) -> Dict[str, Polygon]:
        """Create a lookup dictionary of district polygons."""
        district_lookup = {}
        for district in map_data.districts.values():
            if district.polygon and district.polygon.is_valid:
                district_lookup[district.district_type] = district.polygon
        return district_lookup
    
    def _determine_block_district_type(self, block_polygon: Polygon, district_lookup: Dict[str, Polygon]) -> str:
        """Determine which district type a city block belongs to."""
        block_centroid = block_polygon.centroid
        
        # Check which district contains the block centroid
        for district_type, district_polygon in district_lookup.items():
            try:
                if district_polygon.contains(block_centroid):
                    return district_type
            except Exception:
                continue
        
        # If not contained in any district, find the closest one
        min_distance = float('inf')
        closest_district_type = 'residential'  # Default
        
        for district_type, district_polygon in district_lookup.items():
            try:
                distance = district_polygon.distance(block_centroid)
                if distance < min_distance:
                    min_distance = distance
                    closest_district_type = district_type
            except Exception:
                continue
        
        return closest_district_type
    
    def _validate_zoning(self, map_data: MapData, config: MapConfig):
        """Validate and adjust zoning as needed."""
        # Count blocks by type
        block_counts = {}
        for block in map_data.city_blocks.values():
            district_type = block.district_type
            block_counts[district_type] = block_counts.get(district_type, 0) + 1
        
        print(f"Block distribution: {block_counts}")
        
        # Could add logic here to:
        # - Ensure minimum residential zones
        # - Balance commercial/industrial zones
        # - Validate transportation access
        # etc.
    
    def _create_basic_city_blocks(self, map_data: MapData) -> List[Polygon]:
        """Create basic rectangular city blocks when roads don't form polygons."""
        blocks = []
        block_size = 150  # Size of each block
        margin = 50       # Margin from edges
        
        # Create a grid of blocks
        for x in range(margin, map_data.width - margin, block_size + 20):
            for y in range(margin, map_data.height - margin, block_size + 20):
                # Check if this area is mostly on land
                center_x = x + block_size / 2
                center_y = y + block_size / 2
                
                grid_x = int(center_x / map_data.grid_size)
                grid_y = int(center_y / map_data.grid_size)
                
                if (0 <= grid_x < map_data.grid_width and 
                    0 <= grid_y < map_data.grid_height and 
                    map_data.land_mask[grid_y, grid_x]):
                    
                    # Create rectangular block
                    block = Polygon([
                        (x, y),
                        (x + block_size, y),
                        (x + block_size, y + block_size),
                        (x, y + block_size),
                        (x, y)
                    ])
                    
                    if block.is_valid and block.area > 1000:
                        blocks.append(block)
        
        return blocks 