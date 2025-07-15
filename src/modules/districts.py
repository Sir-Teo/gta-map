"""
Advanced district generation module.
Creates realistic urban hierarchies with multiple cities, towns, and villages following natural settlement patterns.
"""
import random
import math
from typing import List, Tuple, Optional, Dict
import numpy as np
from noise import pnoise2
from shapely.geometry import Polygon, Point
from shapely.ops import voronoi_diagram

from ..core.map_data import MapData, District
from ..config.settings import DistrictConfig


class AdvancedDistrictGenerator:
    """
    Advanced district generator that creates realistic urban hierarchies with multiple settlements.
    """
    
    def __init__(self):
        self.major_cities = []
        self.towns = []
        self.villages = []
        self.settlement_hierarchy = {}
    
    def place_districts(self, map_data: MapData, config: DistrictConfig):
        """
        Place districts with realistic urban hierarchy and settlement patterns.
        """
        print("  → Planning settlement hierarchy...")
        
        # First, identify suitable locations for settlements
        suitable_locations = self._find_suitable_settlement_locations(map_data)
        
        # Create settlement hierarchy
        self._create_settlement_hierarchy(map_data, suitable_locations, config)
        
        # Generate districts for each settlement
        self._generate_settlement_districts(map_data, config)
        
        print(f"  → Created {len(self.major_cities)} major cities, {len(self.towns)} towns, {len(self.villages)} villages")
    
    def _find_suitable_settlement_locations(self, map_data: MapData) -> List[Tuple[float, float, str, float]]:
        """Find locations suitable for different types of settlements."""
        locations = []
        
        # Analyze terrain for settlement suitability
        for y in range(10, map_data.grid_height - 10, 5):
            for x in range(10, map_data.grid_width - 10, 5):
                
                if not map_data.land_mask[y, x]:
                    continue
                
                elevation = map_data.heightmap[y, x]
                
                # Calculate suitability factors
                suitability_score = self._calculate_settlement_suitability(
                    map_data, x, y, elevation
                )
                
                if suitability_score > 0.3:  # Minimum threshold
                    # Convert to map coordinates
                    map_x = x * map_data.grid_size
                    map_y = y * map_data.grid_size
                    
                    # Determine settlement type based on suitability
                    if suitability_score > 0.8:
                        settlement_type = 'major_city'
                    elif suitability_score > 0.6:
                        settlement_type = 'city'
                    elif suitability_score > 0.4:
                        settlement_type = 'town'
                    else:
                        settlement_type = 'village'
                    
                    locations.append((map_x, map_y, settlement_type, suitability_score))
        
        # Sort by suitability score
        locations.sort(key=lambda x: x[3], reverse=True)
        
        return locations
    
    def _calculate_settlement_suitability(self, map_data: MapData, x: int, y: int, elevation: float) -> float:
        """Calculate how suitable a location is for settlement."""
        score = 0.0
        
        # Elevation preferences (moderate elevation is best)
        if 0.35 < elevation < 0.6:
            score += 0.4  # Good elevation
        elif 0.6 <= elevation < 0.75:
            score += 0.2  # Hills - okay but less ideal
        else:
            score -= 0.2  # Too high or too low
        
        # Water proximity (good for trade and resources)
        water_distance = self._calculate_water_distance(map_data, x, y)
        if water_distance < 5:
            score += 0.3  # Close to water
        elif water_distance < 15:
            score += 0.1  # Reasonable distance
        
        # River proximity (extra bonus for rivers)
        river_nearby = self._check_river_proximity(map_data, x, y)
        if river_nearby:
            score += 0.2
        
        # Flatness of surrounding area (good for building)
        flatness = self._calculate_area_flatness(map_data, x, y, radius=5)
        score += flatness * 0.3
        
        # Avoid too close to existing settlements
        existing_penalty = self._calculate_existing_settlement_penalty(x, y)
        score -= existing_penalty
        
        # Natural harbor bonus (coastal areas with good access)
        if self._is_natural_harbor(map_data, x, y):
            score += 0.4
        
        # Transportation potential (valleys and passes)
        transport_score = self._calculate_transport_potential(map_data, x, y)
        score += transport_score * 0.2
        
        return max(0.0, min(1.0, score))
    
    def _calculate_water_distance(self, map_data: MapData, x: int, y: int) -> float:
        """Calculate distance to nearest water."""
        min_dist = float('inf')
        
        for dy in range(-20, 21):
            for dx in range(-20, 21):
                nx, ny = x + dx, y + dy
                if 0 <= nx < map_data.grid_width and 0 <= ny < map_data.grid_height:
                    if not map_data.land_mask[ny, nx]:
                        dist = math.sqrt(dx**2 + dy**2)
                        min_dist = min(min_dist, dist)
                        
                        if min_dist < 5:  # Early exit for nearby water
                            return min_dist
        
        return min_dist if min_dist != float('inf') else 25
    
    def _check_river_proximity(self, map_data: MapData, x: int, y: int) -> bool:
        """Check if location is near a river."""
        map_x = x * map_data.grid_size
        map_y = y * map_data.grid_size
        
        for water_body in map_data.water_bodies.values():
            if water_body.water_type == 'river':
                if water_body.geometry.distance(Point(map_x, map_y)) < 100:
                    return True
        
        return False
    
    def _calculate_area_flatness(self, map_data: MapData, x: int, y: int, radius: int) -> float:
        """Calculate how flat the area around a point is."""
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
        
        # Convert to flatness score (inverted and normalized)
        flatness = max(0.0, 1.0 - std_dev * 10)
        return flatness
    
    def _calculate_existing_settlement_penalty(self, x: int, y: int) -> float:
        """Calculate penalty for being too close to existing settlements."""
        penalty = 0.0
        
        # Check distance to all existing settlements
        all_settlements = self.major_cities + self.towns + self.villages
        
        for settlement in all_settlements:
            sx, sy = settlement['location']
            distance = math.sqrt((x - sx)**2 + (y - sy)**2)
            
            # Minimum distances based on settlement types
            min_distance = {
                'major_city': 400,
                'city': 300,
                'town': 200,
                'village': 100
            }.get(settlement['type'], 150)
            
            if distance < min_distance:
                penalty += (min_distance - distance) / min_distance * 0.5
        
        return penalty
    
    def _is_natural_harbor(self, map_data: MapData, x: int, y: int) -> bool:
        """Check if location could be a natural harbor."""
        # Must be near coast
        if self._calculate_water_distance(map_data, x, y) > 3:
            return False
        
        # Check if there's a protected bay-like formation
        water_directions = []
        for angle in range(0, 360, 30):
            dx = int(5 * math.cos(math.radians(angle)))
            dy = int(5 * math.sin(math.radians(angle)))
            
            nx, ny = x + dx, y + dy
            if 0 <= nx < map_data.grid_width and 0 <= ny < map_data.grid_height:
                if not map_data.land_mask[ny, nx]:
                    water_directions.append(angle)
        
        # Good harbor has water in a limited arc (protected bay)
        if len(water_directions) > 3 and len(water_directions) < 8:
            return True
        
        return False
    
    def _calculate_transport_potential(self, map_data: MapData, x: int, y: int) -> float:
        """Calculate potential for transportation routes through this location."""
        score = 0.0
        
        # Check if location is in a valley (good for roads/railways)
        surrounding_elevations = []
        current_elevation = map_data.heightmap[y, x]
        
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                    
                nx, ny = x + dx, y + dy
                if 0 <= nx < map_data.grid_width and 0 <= ny < map_data.grid_height:
                    surrounding_elevations.append(map_data.heightmap[ny, nx])
        
        if surrounding_elevations:
            avg_surrounding = sum(surrounding_elevations) / len(surrounding_elevations)
            if current_elevation < avg_surrounding - 0.05:  # In a valley
                score += 0.5
        
        # Check for mountain passes (lower elevation between high areas)
        pass_score = self._check_mountain_pass(map_data, x, y)
        score += pass_score
        
        return score
    
    def _check_mountain_pass(self, map_data: MapData, x: int, y: int) -> float:
        """Check if location is a mountain pass."""
        current_elevation = map_data.heightmap[y, x]
        
        if current_elevation < 0.5:  # Too low to be a pass
            return 0.0
        
        # Look for high elevations on opposite sides
        directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]  # N, E, NE, NW
        
        for dx, dy in directions:
            # Check for high ground on both sides
            high_count_pos = 0
            high_count_neg = 0
            
            for dist in range(3, 8):
                # Positive direction
                nx, ny = x + dx * dist, y + dy * dist
                if 0 <= nx < map_data.grid_width and 0 <= ny < map_data.grid_height:
                    if map_data.heightmap[ny, nx] > current_elevation + 0.1:
                        high_count_pos += 1
                
                # Negative direction
                nx, ny = x - dx * dist, y - dy * dist
                if 0 <= nx < map_data.grid_width and 0 <= ny < map_data.grid_height:
                    if map_data.heightmap[ny, nx] > current_elevation + 0.1:
                        high_count_neg += 1
            
            if high_count_pos >= 2 and high_count_neg >= 2:
                return 0.3  # Found a pass
        
        return 0.0
    
    def _create_settlement_hierarchy(self, map_data: MapData, locations: List[Tuple[float, float, str, float]], config: DistrictConfig):
        """Create a realistic hierarchy of settlements."""
        
        # Determine number of settlements based on map size
        map_area = map_data.width * map_data.height
        area_factor = map_area / (2000 * 2000)  # Normalize to default size
        
        target_major_cities = max(1, int(1 * area_factor))
        target_cities = max(1, int(2 * area_factor))
        target_towns = max(2, int(4 * area_factor))
        target_villages = max(3, int(8 * area_factor))
        
        settlements_created = 0
        
        # Place major cities first (highest priority)
        for location in locations:
            if settlements_created >= target_major_cities + target_cities + target_towns + target_villages:
                break
                
            x, y, suggested_type, score = location
            
            # Check minimum distance constraints
            if self._check_minimum_distance_constraints(x, y, suggested_type):
                continue
            
            # Create settlement based on targets
            if len(self.major_cities) < target_major_cities and score > 0.7:
                settlement = self._create_settlement(x, y, 'major_city', score)
                self.major_cities.append(settlement)
                settlements_created += 1
                
            elif len(self.major_cities) + len(self.towns) < target_major_cities + target_cities and score > 0.6:
                settlement = self._create_settlement(x, y, 'city', score)
                self.towns.append(settlement)  # Cities go in towns list
                settlements_created += 1
                
            elif len(self.towns) < target_towns and score > 0.4:
                settlement = self._create_settlement(x, y, 'town', score)
                self.towns.append(settlement)
                settlements_created += 1
                
            elif len(self.villages) < target_villages and score > 0.3:
                settlement = self._create_settlement(x, y, 'village', score)
                self.villages.append(settlement)
                settlements_created += 1
    
    def _check_minimum_distance_constraints(self, x: float, y: float, settlement_type: str) -> bool:
        """Check if location violates minimum distance constraints."""
        min_distances = {
            'major_city': 600,
            'city': 400,
            'town': 250,
            'village': 150
        }
        
        min_dist = min_distances.get(settlement_type, 150)
        
        # Check against all existing settlements
        all_settlements = self.major_cities + self.towns + self.villages
        
        for settlement in all_settlements:
            sx, sy = settlement['location']
            distance = math.sqrt((x - sx)**2 + (y - sy)**2)
            
            if distance < min_dist:
                return True  # Violates constraint
        
        return False
    
    def _create_settlement(self, x: float, y: float, settlement_type: str, score: float) -> Dict:
        """Create a settlement data structure."""
        
        # Calculate settlement size based on type and score
        size_multipliers = {
            'major_city': 1.5,
            'city': 1.2,
            'town': 1.0,
            'village': 0.7
        }
        
        base_radius = 200 * size_multipliers[settlement_type] * (0.5 + score * 0.5)
        
        return {
            'location': (x, y),
            'type': settlement_type,
            'radius': base_radius,
            'suitability_score': score,
            'districts': []
        }
    
    def _generate_settlement_districts(self, map_data: MapData, config: DistrictConfig):
        """Generate districts for each settlement in the hierarchy."""
        
        # Generate districts for major cities
        for settlement in self.major_cities:
            self._generate_major_city_districts(map_data, settlement, config)
        
        # Generate districts for towns/cities
        for settlement in self.towns:
            self._generate_town_districts(map_data, settlement, config)
        
        # Generate districts for villages
        for settlement in self.villages:
            self._generate_village_districts(map_data, settlement, config)
    
    def _generate_major_city_districts(self, map_data: MapData, settlement: Dict, config: DistrictConfig):
        """Generate comprehensive districts for a major city."""
        
        x, y = settlement['location']
        radius = settlement['radius']
        
        # Major cities have full range of districts
        district_types = [
            ('downtown', 0.3, 0.8),
            ('commercial', 0.5, 1.2),
            ('industrial', 0.8, 1.5),
            ('residential', 0.6, 1.8),
            ('suburban', 1.2, 2.5),
            ('port', 0.4, 0.9) if self._is_coastal_settlement(map_data, x, y) else None,
            ('airport', 1.5, 2.0),
            ('park', 0.7, 1.5)
        ]
        
        # Remove None entries
        district_types = [dt for dt in district_types if dt is not None]
        
        self._place_districts_around_center(map_data, settlement, district_types, config)
    
    def _generate_town_districts(self, map_data: MapData, settlement: Dict, config: DistrictConfig):
        """Generate districts for a town or small city."""
        
        x, y = settlement['location']
        
        # Towns have fewer, simpler districts
        district_types = [
            ('commercial', 0.3, 0.6),
            ('residential', 0.5, 1.2),
            ('industrial', 0.8, 1.0) if random.random() > 0.3 else None,
            ('suburban', 0.9, 1.5),
            ('port', 0.4, 0.7) if self._is_coastal_settlement(map_data, x, y) and random.random() > 0.5 else None
        ]
        
        # Remove None entries
        district_types = [dt for dt in district_types if dt is not None]
        
        self._place_districts_around_center(map_data, settlement, district_types, config)
    
    def _generate_village_districts(self, map_data: MapData, settlement: Dict, config: DistrictConfig):
        """Generate simple districts for a village."""
        
        # Villages typically have just residential areas
        district_types = [
            ('residential', 0.3, 0.8),
            ('suburban', 0.6, 1.2)
        ]
        
        self._place_districts_around_center(map_data, settlement, district_types, config)
    
    def _place_districts_around_center(self, map_data: MapData, settlement: Dict, district_types: List[Tuple], config: DistrictConfig):
        """Place districts in a realistic pattern around settlement center."""
        
        center_x, center_y = settlement['location']
        base_radius = settlement['radius']
        
        for i, (district_type, min_radius_mult, max_radius_mult) in enumerate(district_types):
            
            # Calculate district placement
            angle = (i / len(district_types)) * 2 * math.pi + random.uniform(-0.5, 0.5)
            
            # Distance from center
            min_distance = base_radius * min_radius_mult
            max_distance = base_radius * max_radius_mult
            distance = random.uniform(min_distance, max_distance)
            
            # Special placement rules
            if district_type == 'downtown':
                # Downtown goes in the center
                district_x, district_y = center_x, center_y
                district_radius = base_radius * 0.3
                
            elif district_type == 'port':
                # Ports go toward water
                water_direction = self._find_water_direction(map_data, center_x, center_y)
                if water_direction is not None:
                    angle = water_direction
                district_x = center_x + math.cos(angle) * distance
                district_y = center_y + math.sin(angle) * distance
                district_radius = base_radius * 0.4
                
            elif district_type == 'airport':
                # Airports go away from city center, prefer flat areas
                angle = self._find_flat_direction(map_data, center_x, center_y, base_radius)
                district_x = center_x + math.cos(angle) * distance
                district_y = center_y + math.sin(angle) * distance
                district_radius = base_radius * 0.6
                
            elif district_type == 'industrial':
                # Industrial areas prefer river/rail access, downwind
                if random.random() > 0.5:
                    river_direction = self._find_river_direction(map_data, center_x, center_y)
                    if river_direction is not None:
                        angle = river_direction
                
                district_x = center_x + math.cos(angle) * distance
                district_y = center_y + math.sin(angle) * distance
                district_radius = base_radius * 0.5
                
            else:
                # Default placement
                district_x = center_x + math.cos(angle) * distance
                district_y = center_y + math.sin(angle) * distance
                district_radius = base_radius * random.uniform(0.3, 0.6)
            
            # Ensure district is on land and within map bounds
            if self._is_valid_district_location(map_data, district_x, district_y, district_radius):
                
                # Create district
                district = self._create_district(
                    district_x, district_y, district_radius, district_type, 
                    settlement, config
                )
                
                if district:
                    map_data.add_district(district)
                    settlement['districts'].append(district.id)
    
    def _is_coastal_settlement(self, map_data: MapData, x: float, y: float) -> bool:
        """Check if settlement is near coast."""
        grid_x = int(x / map_data.grid_size)
        grid_y = int(y / map_data.grid_size)
        
        return self._calculate_water_distance(map_data, grid_x, grid_y) < 10
    
    def _find_water_direction(self, map_data: MapData, x: float, y: float) -> Optional[float]:
        """Find direction to nearest water."""
        grid_x = int(x / map_data.grid_size)
        grid_y = int(y / map_data.grid_size)
        
        nearest_water = None
        min_distance = float('inf')
        
        for dy in range(-20, 21):
            for dx in range(-20, 21):
                nx, ny = grid_x + dx, grid_y + dy
                if 0 <= nx < map_data.grid_width and 0 <= ny < map_data.grid_height:
                    if not map_data.land_mask[ny, nx]:
                        distance = math.sqrt(dx**2 + dy**2)
                        if distance < min_distance:
                            min_distance = distance
                            nearest_water = (dx, dy)
        
        if nearest_water:
            dx, dy = nearest_water
            return math.atan2(dy, dx)
        
        return None
    
    def _find_flat_direction(self, map_data: MapData, x: float, y: float, radius: float) -> float:
        """Find direction toward flattest terrain."""
        
        best_direction = random.uniform(0, 2 * math.pi)
        best_flatness = -1
        
        # Try 8 directions
        for i in range(8):
            angle = (i / 8) * 2 * math.pi
            
            # Sample terrain in this direction
            sample_x = x + math.cos(angle) * radius
            sample_y = y + math.sin(angle) * radius
            
            flatness = self._sample_area_flatness(map_data, sample_x, sample_y, radius * 0.3)
            
            if flatness > best_flatness:
                best_flatness = flatness
                best_direction = angle
        
        return best_direction
    
    def _find_river_direction(self, map_data: MapData, x: float, y: float) -> Optional[float]:
        """Find direction to nearest river."""
        
        nearest_river_point = None
        min_distance = float('inf')
        
        for water_body in map_data.water_bodies.values():
            if water_body.water_type == 'river':
                # Find closest point on river
                point = Point(x, y)
                distance = water_body.geometry.distance(point)
                
                if distance < min_distance:
                    min_distance = distance
                    # Get nearest point on river geometry
                    if hasattr(water_body.geometry, 'coords'):
                        # LineString
                        coords = list(water_body.geometry.coords)
                        nearest_river_point = min(coords, key=lambda p: math.sqrt((p[0]-x)**2 + (p[1]-y)**2))
        
        if nearest_river_point:
            dx = nearest_river_point[0] - x
            dy = nearest_river_point[1] - y
            return math.atan2(dy, dx)
        
        return None
    
    def _sample_area_flatness(self, map_data: MapData, x: float, y: float, radius: float) -> float:
        """Sample flatness in an area around a point."""
        grid_x = int(x / map_data.grid_size)
        grid_y = int(y / map_data.grid_size)
        grid_radius = int(radius / map_data.grid_size)
        
        return self._calculate_area_flatness(map_data, grid_x, grid_y, grid_radius)
    
    def _is_valid_district_location(self, map_data: MapData, x: float, y: float, radius: float) -> bool:
        """Check if district location is valid."""
        
        # Check bounds
        if (x - radius < 0 or x + radius >= map_data.width or
            y - radius < 0 or y + radius >= map_data.height):
            return False
        
        # Check if center is on land
        grid_x = int(x / map_data.grid_size)
        grid_y = int(y / map_data.grid_size)
        
        if (0 <= grid_x < map_data.grid_width and 
            0 <= grid_y < map_data.grid_height and
            map_data.land_mask[grid_y, grid_x]):
            return True
        
        return False
    
    def _create_district(self, x: float, y: float, radius: float, district_type: str, settlement: Dict, config: DistrictConfig) -> Optional[District]:
        """Create a district with proper geometry."""
        
        # Generate district shape
        vertices = self._generate_district_shape(x, y, radius, district_type)
        
        if len(vertices) < 3:
            return None
        
        # Create district ID
        settlement_name = settlement['type']
        district_id = f"{district_type}_{settlement_name}_{len(settlement['districts'])}"
        
        # Get color
        color = config.colors.get(district_type, '#cccccc')
        
        district = District(
            id=district_id,
            center=(x, y),
            district_type=district_type,
            vertices=vertices,
            radius=radius,
            color=color
        )
        
        return district
    
    def _generate_district_shape(self, center_x: float, center_y: float, radius: float, district_type: str) -> List[Tuple[float, float]]:
        """Generate a realistic shape for the district."""
        
        vertices = []
        
        # Number of vertices based on district type
        if district_type in ['downtown', 'commercial']:
            num_vertices = random.randint(6, 10)  # More regular shapes
        elif district_type in ['industrial', 'airport']:
            num_vertices = random.randint(4, 8)   # More angular
        else:
            num_vertices = random.randint(8, 12)  # More organic
        
        for i in range(num_vertices):
            angle = (i / num_vertices) * 2 * math.pi
            
            # Add some randomness to radius
            radius_variation = random.uniform(0.7, 1.3)
            
            # Add noise for organic shape
            noise_factor = pnoise2(
                math.cos(angle) * 2, math.sin(angle) * 2, octaves=2
            ) * 0.3 + 1.0
            
            final_radius = radius * radius_variation * noise_factor
            
            x = center_x + math.cos(angle) * final_radius
            y = center_y + math.sin(angle) * final_radius
            
            vertices.append((x, y))
        
        return vertices


# Backward compatibility with existing system
class DistrictGenerator:
    """Legacy district generator - now uses advanced system internally."""
    
    def __init__(self):
        self.advanced_generator = AdvancedDistrictGenerator()
    
    def place_districts(self, map_data: MapData, config: DistrictConfig):
        """Place districts using advanced system."""
        self.advanced_generator.place_districts(map_data, config) 