"""
Advanced transportation generation module.
Handles road networks, highways, railways, bridges, and transportation infrastructure with terrain awareness.
"""
import random
import math
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
from noise import pnoise2
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points
from dataclasses import dataclass

from ..core.map_data import MapData, Road
from ..config.settings import TransportationConfig


@dataclass
class Bridge:
    """Represents a bridge structure."""
    id: str
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    bridge_type: str  # 'highway', 'road', 'railway'
    water_body_id: str
    length: float
    width: float
    color: str = "#8B4513"
    
    @property
    def linestring(self) -> LineString:
        """Get the bridge as a LineString."""
        return LineString([self.start_point, self.end_point])


@dataclass  
class Tunnel:
    """Represents a tunnel through mountains."""
    id: str
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    tunnel_type: str  # 'highway', 'road', 'railway'
    length: float
    width: float
    color: str = "#404040"
    
    @property
    def linestring(self) -> LineString:
        """Get the tunnel as a LineString."""
        return LineString([self.start_point, self.end_point])


class AdvancedTransportationGenerator:
    """
    Advanced transportation generator with terrain awareness, bridges, tunnels, and realistic routing.
    """
    
    def __init__(self):
        self.bridges = []
        self.tunnels = []
        self.railway_network = []
        self.major_highways = []
        self.elevation_cost_cache = {}
    
    def generate_highway_network(self, map_data: MapData, config: TransportationConfig):
        """
        Generate a network of highways connecting major areas with elevation awareness.
        """
        print("  → Generating highway network...")
        
        # Find major cities/districts for highway connections
        connection_points = self._find_highway_connection_points(map_data)
        
        # Generate primary interstate-style highways
        self._generate_primary_highways(map_data, config, connection_points)
        
        # Generate ring roads around major urban areas
        self._generate_ring_roads(map_data, config)
        
        print(f"  → Generated {len(self.major_highways)} major highways")
    
    def generate_arterial_grid(self, map_data: MapData, config: TransportationConfig):
        """
        Generate arterial roads following terrain contours and connecting districts.
        """
        print("  → Generating arterial road grid...")
        
        # Generate elevation-aware arterial roads
        self._generate_terrain_following_arterials(map_data, config)
        
        # Connect districts with direct arterials where possible
        self._generate_district_connectors(map_data, config)
        
        print("  → Arterial road network complete")
    
    def generate_local_roads(self, map_data: MapData, config: TransportationConfig):
        """
        Generate local roads and collectors with realistic patterns.
        """
        print("  → Generating local road network...")
        
        # Generate collector roads between arterials
        self._generate_collector_roads(map_data, config)
        
        # Generate local roads in districts with organic patterns
        self._generate_organic_local_roads(map_data, config)
        
        print("  → Local road network complete")
    
    def generate_bridges(self, map_data: MapData, config: TransportationConfig):
        """
        Automatically detect road-water intersections and place bridges.
        """
        print("  → Detecting river crossings and placing bridges...")
        
        bridge_count = 0
        
        # Check all roads for water crossings
        for road_id, road in map_data.roads.items():
            if not road.linestring:
                continue
                
            # Check intersections with water bodies
            for water_id, water_body in map_data.water_bodies.items():
                intersections = self._find_road_water_intersections(road, water_body)
                
                for intersection in intersections:
                    bridge = self._create_bridge_at_crossing(
                        road, water_body, intersection, f"bridge_{bridge_count}", config
                    )
                    if bridge:
                        self.bridges.append(bridge)
                        bridge_count += 1
        
        print(f"  → Placed {bridge_count} bridges")
    
    def generate_tunnels(self, map_data: MapData, config: TransportationConfig):
        """
        Generate tunnels through mountains for major highways.
        """
        print("  → Checking for tunnel opportunities...")
        
        tunnel_count = 0
        
        # Check highways for mountain crossings
        for road_id, road in map_data.roads.items():
            if road.road_type == 'highway' and road.linestring:
                tunnel_segments = self._find_tunnel_opportunities(map_data, road)
                
                for start_point, end_point in tunnel_segments:
                    tunnel = Tunnel(
                        id=f"tunnel_{tunnel_count}",
                        start_point=start_point,
                        end_point=end_point,
                        tunnel_type='highway',
                        length=math.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2),
                        width=road.width
                    )
                    self.tunnels.append(tunnel)
                    tunnel_count += 1
        
        print(f"  → Generated {tunnel_count} tunnels")
    
    def generate_railway_network(self, map_data: MapData, config: TransportationConfig):
        """
        Generate railway network connecting major cities and industrial areas.
        """
        print("  → Generating railway network...")
        
        # Find major cities for railway connections
        railway_cities = self._find_railway_cities(map_data)
        
        # Generate main railway lines
        self._generate_main_railway_lines(map_data, config, railway_cities)
        
        # Generate branch lines to industrial areas
        self._generate_branch_railway_lines(map_data, config)
        
        # Generate railway bridges
        self._generate_railway_bridges(map_data, config)
        
        print(f"  → Generated {len(self.railway_network)} railway segments")
    
    def _find_highway_connection_points(self, map_data: MapData) -> List[Tuple[float, float]]:
        """Find major points that should be connected by highways."""
        connection_points = []
        
        # Add district centers as connection points
        for district in map_data.districts.values():
            if district.district_type in ['downtown', 'commercial', 'industrial', 'airport']:
                connection_points.append(district.center)
        
        # Add map edge points for interstate connections
        width, height = map_data.width, map_data.height
        edge_points = [
            (width * 0.2, 0), (width * 0.8, 0),  # Top edge
            (width, height * 0.3), (width, height * 0.7),  # Right edge
            (width * 0.2, height), (width * 0.8, height),  # Bottom edge
            (0, height * 0.3), (0, height * 0.7)  # Left edge
        ]
        
        # Filter edge points to land areas
        for point in edge_points:
            grid_x = int(point[0] / map_data.grid_size)
            grid_y = int(point[1] / map_data.grid_size)
            
            if (0 <= grid_x < map_data.grid_width and 
                0 <= grid_y < map_data.grid_height and
                map_data.land_mask[grid_y, grid_x]):
                connection_points.append(point)
        
        return connection_points
    
    def _generate_primary_highways(self, map_data: MapData, config: TransportationConfig, connection_points: List[Tuple[float, float]]):
        """Generate primary interstate-style highways."""
        
        # Create highway network using minimum spanning tree approach
        if len(connection_points) < 2:
            return
        
        connected = {0}  # Start with first point
        highways_created = 0
        
        while len(connected) < len(connection_points) and highways_created < 8:
            best_connection = None
            best_cost = float('inf')
            
            # Find the best unconnected point to connect
            for i in connected:
                for j in range(len(connection_points)):
                    if j not in connected:
                        cost = self._calculate_highway_cost(
                            map_data, connection_points[i], connection_points[j]
                        )
                        if cost < best_cost:
                            best_cost = cost
                            best_connection = (i, j)
            
            if best_connection:
                start_point = connection_points[best_connection[0]]
                end_point = connection_points[best_connection[1]]
                
                highway = self._generate_elevation_aware_highway(
                    map_data, start_point, end_point, f"highway_{highways_created}", config
                )
                
                if highway:
                    map_data.add_road(highway)
                    self.major_highways.append(highway)
                    connected.add(best_connection[1])
                    highways_created += 1
    
    def _generate_ring_roads(self, map_data: MapData, config: TransportationConfig):
        """Generate ring roads around major urban areas."""
        
        # Find downtown areas for ring roads
        downtown_districts = [d for d in map_data.districts.values() if d.district_type == 'downtown']
        
        for i, district in enumerate(downtown_districts[:2]):  # Limit to 2 ring roads
            ring_road = self._create_ring_road(map_data, district, f"ring_{i}", config)
            if ring_road:
                map_data.add_road(ring_road)
    
    def _generate_terrain_following_arterials(self, map_data: MapData, config: TransportationConfig):
        """Generate arterials that follow natural terrain contours."""
        
        # Generate north-south arterials
        for x in range(config.arterial_spacing, map_data.width, config.arterial_spacing):
            arterial = self._generate_contour_following_road(
                map_data, (x, 0), (x, map_data.height), 'arterial', f"arterial_ns_{x}", config
            )
            if arterial:
                map_data.add_road(arterial)
        
        # Generate east-west arterials  
        for y in range(config.arterial_spacing, map_data.height, config.arterial_spacing):
            arterial = self._generate_contour_following_road(
                map_data, (0, y), (map_data.width, y), 'arterial', f"arterial_ew_{y}", config
            )
            if arterial:
                map_data.add_road(arterial)
    
    def _generate_district_connectors(self, map_data: MapData, config: TransportationConfig):
        """Generate direct arterial connections between nearby districts."""
        
        districts = list(map_data.districts.values())
        connector_id = 0
        
        for i, district1 in enumerate(districts):
            for district2 in districts[i+1:]:
                distance = math.sqrt(
                    (district1.center[0] - district2.center[0])**2 +
                    (district1.center[1] - district2.center[1])**2
                )
                
                # Connect nearby districts
                if 200 < distance < 800:
                    connector = self._generate_elevation_aware_road(
                        map_data, district1.center, district2.center, 
                        'arterial', f"connector_{connector_id}", config
                    )
                    if connector:
                        map_data.add_road(connector)
                        connector_id += 1
    
    def _generate_collector_roads(self, map_data: MapData, config: TransportationConfig):
        """Generate collector roads between arterials."""
        
        collector_spacing = config.arterial_spacing // 2
        collector_id = 0
        
        # Generate intermediate collectors
        for x in range(collector_spacing, map_data.width, config.arterial_spacing):
            collector = self._generate_contour_following_road(
                map_data, (x, 0), (x, map_data.height), 'collector', f"collector_ns_{collector_id}", config
            )
            if collector:
                map_data.add_road(collector)
                collector_id += 1
        
        for y in range(collector_spacing, map_data.height, config.arterial_spacing):
            collector = self._generate_contour_following_road(
                map_data, (0, y), (map_data.width, y), 'collector', f"collector_ew_{collector_id}", config
            )
            if collector:
                map_data.add_road(collector)
                collector_id += 1
    
    def _generate_organic_local_roads(self, map_data: MapData, config: TransportationConfig):
        """Generate local roads with organic, neighborhood-like patterns."""
        
        # Generate local roads in each district
        for district in map_data.districts.values():
            if district.polygon and district.district_type in ['residential', 'suburban', 'commercial']:
                self._generate_district_local_roads(map_data, district, config)
    
    def _generate_district_local_roads(self, map_data: MapData, district, config: TransportationConfig):
        """Generate local roads within a specific district."""
        
        if not district.polygon:
            return
        
        # Get district bounds
        minx, miny, maxx, maxy = district.polygon.bounds
        
        # Generate a small grid of local roads
        local_spacing = 150  # Smaller spacing for local roads
        road_id = 0
        
        # Vertical local roads
        for x in range(int(minx), int(maxx), local_spacing):
            if district.polygon.contains(Point(x, (miny + maxy) / 2)):
                local_road = self._generate_simple_road(
                    map_data, (x, miny), (x, maxy), 'local', f"local_{district.id}_{road_id}", config
                )
                if local_road:
                    map_data.add_road(local_road)
                    road_id += 1
        
        # Horizontal local roads
        for y in range(int(miny), int(maxy), local_spacing):
            if district.polygon.contains(Point((minx + maxx) / 2, y)):
                local_road = self._generate_simple_road(
                    map_data, (minx, y), (maxx, y), 'local', f"local_{district.id}_{road_id}", config
                )
                if local_road:
                    map_data.add_road(local_road)
                    road_id += 1
    
    def _calculate_highway_cost(self, map_data: MapData, start: Tuple[float, float], end: Tuple[float, float]) -> float:
        """Calculate the cost of building a highway between two points."""
        
        # Base distance cost
        distance = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        cost = distance
        
        # Sample elevation along the path
        steps = 20
        elevation_cost = 0
        
        for i in range(steps):
            t = i / (steps - 1)
            x = start[0] * (1 - t) + end[0] * t
            y = start[1] * (1 - t) + end[1] * t
            
            grid_x = int(x / map_data.grid_size)
            grid_y = int(y / map_data.grid_size)
            
            if 0 <= grid_x < map_data.grid_width and 0 <= grid_y < map_data.grid_height:
                elevation = map_data.heightmap[grid_y, grid_x]
                
                # High elevation increases cost (tunnels needed)
                if elevation > 0.7:
                    elevation_cost += (elevation - 0.7) * 1000
                
                # Water crossings increase cost (bridges needed)
                if not map_data.land_mask[grid_y, grid_x]:
                    elevation_cost += 500
        
        return cost + elevation_cost
    
    def _generate_elevation_aware_highway(
        self, map_data: MapData, start: Tuple[float, float], end: Tuple[float, float], 
        highway_id: str, config: TransportationConfig
    ) -> Optional[Road]:
        """Generate a highway that avoids steep terrain when possible."""
        
        points = self._pathfind_with_elevation(map_data, start, end, 'highway')
        
        if points and len(points) >= 2:
            return Road(
                id=highway_id,
                points=points,
                road_type='highway',
                width=config.road_styles['highway']['width'],
                color=config.road_styles['highway']['color']
            )
        
        return None
    
    def _generate_elevation_aware_road(
        self, map_data: MapData, start: Tuple[float, float], end: Tuple[float, float],
        road_type: str, road_id: str, config: TransportationConfig
    ) -> Optional[Road]:
        """Generate a road that follows reasonable elevation changes."""
        
        points = self._pathfind_with_elevation(map_data, start, end, road_type)
        
        if points and len(points) >= 2:
            style = config.road_styles.get(road_type, config.road_styles['local'])
            return Road(
                id=road_id,
                points=points,
                road_type=road_type,
                width=style['width'],
                color=style['color']
            )
        
        return None
    
    def _generate_contour_following_road(
        self, map_data: MapData, start: Tuple[float, float], end: Tuple[float, float],
        road_type: str, road_id: str, config: TransportationConfig
    ) -> Optional[Road]:
        """Generate a road that follows elevation contours when possible."""
        
        # Use simplified pathfinding that considers elevation
        points = self._simple_terrain_aware_path(map_data, start, end)
        
        if points and len(points) >= 2:
            style = config.road_styles.get(road_type, config.road_styles['local'])
            return Road(
                id=road_id,
                points=points,
                road_type=road_type,
                width=style['width'],
                color=style['color']
            )
        
        return None
    
    def _generate_simple_road(
        self, map_data: MapData, start: Tuple[float, float], end: Tuple[float, float],
        road_type: str, road_id: str, config: TransportationConfig
    ) -> Optional[Road]:
        """Generate a simple straight road with minor elevation adjustments."""
        
        # Simple straight line with small adjustments
        points = [start]
        
        # Add intermediate points for elevation following
        steps = max(3, int(math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2) / 100))
        
        for i in range(1, steps):
            t = i / steps
            x = start[0] * (1 - t) + end[0] * t
            y = start[1] * (1 - t) + end[1] * t
            
            # Add small adjustments based on local terrain
            grid_x = int(x / map_data.grid_size)
            grid_y = int(y / map_data.grid_size)
            
            if 0 <= grid_x < map_data.grid_width - 1 and 0 <= grid_y < map_data.grid_height - 1:
                # Small terrain-following adjustment
                dx = map_data.heightmap[grid_y, grid_x + 1] - map_data.heightmap[grid_y, grid_x]
                dy = map_data.heightmap[grid_y + 1, grid_x] - map_data.heightmap[grid_y, grid_x]
                
                # Adjust perpendicular to steep slopes
                adjustment = 20
                x += dy * adjustment
                y -= dx * adjustment
            
            points.append((x, y))
        
        points.append(end)
        
        style = config.road_styles.get(road_type, config.road_styles['local'])
        return Road(
            id=road_id,
            points=points,
            road_type=road_type,
            width=style['width'],
            color=style['color']
        )
    
    def _pathfind_with_elevation(self, map_data: MapData, start: Tuple[float, float], end: Tuple[float, float], road_type: str) -> List[Tuple[float, float]]:
        """Use A* pathfinding with elevation costs."""
        
        # Simplified pathfinding - for full implementation would use proper A*
        # For now, use straight line with elevation-aware waypoints
        
        points = [start]
        current = start
        
        steps = 20
        for i in range(1, steps):
            t = i / steps
            next_point = (
                start[0] * (1 - t) + end[0] * t,
                start[1] * (1 - t) + end[1] * t
            )
            
            # Check if we need to deviate for elevation
            detour = self._find_elevation_detour(map_data, current, next_point, road_type)
            if detour:
                points.extend(detour)
                current = detour[-1]
            else:
                points.append(next_point)
                current = next_point
        
        if points[-1] != end:
            points.append(end)
        
        return points
    
    def _simple_terrain_aware_path(self, map_data: MapData, start: Tuple[float, float], end: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Generate a simple path that avoids major elevation changes."""
        
        points = [start]
        
        # Calculate number of steps based on distance
        distance = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        steps = max(5, int(distance / 50))
        
        for i in range(1, steps):
            t = i / steps
            base_x = start[0] * (1 - t) + end[0] * t
            base_y = start[1] * (1 - t) + end[1] * t
            
            # Add terrain-following offset
            grid_x = int(base_x / map_data.grid_size)
            grid_y = int(base_y / map_data.grid_size)
            
            if 0 <= grid_x < map_data.grid_width and 0 <= grid_y < map_data.grid_height:
                # Small noise-based variation for more natural roads
                offset_x = pnoise2(base_x * 0.002, base_y * 0.002, octaves=2) * 30
                offset_y = pnoise2(base_x * 0.002 + 100, base_y * 0.002 + 100, octaves=2) * 30
                
                final_x = base_x + offset_x
                final_y = base_y + offset_y
                
                points.append((final_x, final_y))
        
        points.append(end)
        return points
    
    def _find_elevation_detour(self, map_data: MapData, start: Tuple[float, float], end: Tuple[float, float], road_type: str) -> Optional[List[Tuple[float, float]]]:
        """Find a detour around steep terrain if needed."""
        
        # Check if direct path has problematic elevation
        grid_start_x = int(start[0] / map_data.grid_size)
        grid_start_y = int(start[1] / map_data.grid_size)
        grid_end_x = int(end[0] / map_data.grid_size)
        grid_end_y = int(end[1] / map_data.grid_size)
        
        if (0 <= grid_start_x < map_data.grid_width and 0 <= grid_start_y < map_data.grid_height and
            0 <= grid_end_x < map_data.grid_width and 0 <= grid_end_y < map_data.grid_height):
            
            start_elevation = map_data.heightmap[grid_start_y, grid_start_x]
            end_elevation = map_data.heightmap[grid_end_y, grid_end_x]
            
            elevation_diff = abs(end_elevation - start_elevation)
            
            # If elevation change is too steep, create a detour
            if road_type != 'highway' and elevation_diff > 0.15:
                # Create a simple detour via intermediate point
                mid_x = (start[0] + end[0]) / 2
                mid_y = (start[1] + end[1]) / 2
                
                # Offset the midpoint perpendicular to the line
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                length = math.sqrt(dx**2 + dy**2)
                
                if length > 0:
                    # Perpendicular offset
                    offset_distance = 100
                    perp_x = -dy / length * offset_distance
                    perp_y = dx / length * offset_distance
                    
                    detour_point = (mid_x + perp_x, mid_y + perp_y)
                    return [detour_point]
        
        return None
    
    def _create_ring_road(self, map_data: MapData, district, ring_id: str, config: TransportationConfig) -> Optional[Road]:
        """Create a ring road around a district."""
        
        center_x, center_y = district.center
        
        # Calculate ring radius based on district size
        if district.polygon:
            bounds = district.polygon.bounds
            radius = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.8
        else:
            radius = district.radius * 1.5
        
        # Generate ring points
        ring_points = []
        num_points = 16
        
        for i in range(num_points + 1):  # +1 to close the ring
            angle = (i / num_points) * 2 * math.pi
            
            # Add variation for more natural shape
            radius_variation = pnoise2(math.cos(angle), math.sin(angle), octaves=2) * radius * 0.2
            current_radius = radius + radius_variation
            
            x = center_x + math.cos(angle) * current_radius
            y = center_y + math.sin(angle) * current_radius
            
            # Clamp to map bounds
            x = max(50, min(x, map_data.width - 50))
            y = max(50, min(y, map_data.height - 50))
            
            ring_points.append((x, y))
        
        return Road(
            id=ring_id,
            points=ring_points,
            road_type='arterial',
            width=config.road_styles['arterial']['width'],
            color=config.road_styles['arterial']['color']
        )
    
    def _find_road_water_intersections(self, road: Road, water_body) -> List[Point]:
        """Find where a road intersects with a water body."""
        
        if not road.linestring or not water_body.geometry:
            return []
        
        intersections = []
        
        try:
            intersection = road.linestring.intersection(water_body.geometry)
            
            if intersection.is_empty:
                return []
            
            # Handle different types of intersections
            if hasattr(intersection, 'geoms'):  # Multiple intersections
                for geom in intersection.geoms:
                    if geom.geom_type == 'Point':
                        intersections.append(geom)
            elif intersection.geom_type == 'Point':
                intersections.append(intersection)
                
        except Exception:
            pass  # Skip invalid intersections
        
        return intersections
    
    def _create_bridge_at_crossing(self, road: Road, water_body, intersection_point: Point, bridge_id: str, config: TransportationConfig) -> Optional[Bridge]:
        """Create a bridge at a road-water intersection."""
        
        if not road.linestring:
            return None
        
        # Find the road segment that crosses the water
        road_coords = list(road.linestring.coords)
        crossing_x, crossing_y = intersection_point.x, intersection_point.y
        
        # Find nearest road points before and after the crossing
        min_dist = float('inf')
        segment_idx = 0
        
        for i in range(len(road_coords) - 1):
            segment = LineString([road_coords[i], road_coords[i + 1]])
            dist = segment.distance(intersection_point)
            if dist < min_dist:
                min_dist = dist
                segment_idx = i
        
        if segment_idx < len(road_coords) - 1:
            start_point = road_coords[segment_idx]
            end_point = road_coords[segment_idx + 1]
            
            # Extend bridge slightly beyond water edges for safety
            bridge_length = math.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
            
            # Determine bridge type based on road type
            if road.road_type == 'highway':
                bridge_type = 'highway'
                bridge_width = road.width + 4  # Wider bridge for highway
            else:
                bridge_type = 'road'
                bridge_width = road.width + 2
            
            return Bridge(
                id=bridge_id,
                start_point=start_point,
                end_point=end_point,
                bridge_type=bridge_type,
                water_body_id=water_body.id,
                length=bridge_length,
                width=bridge_width
            )
        
        return None
    
    def _find_tunnel_opportunities(self, map_data: MapData, road: Road) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Find segments of the road that should be tunneled through mountains."""
        
        if not road.linestring:
            return []
        
        tunnel_segments = []
        road_coords = list(road.linestring.coords)
        
        tunnel_start = None
        
        for i, (x, y) in enumerate(road_coords):
            grid_x = int(x / map_data.grid_size)
            grid_y = int(y / map_data.grid_size)
            
            if 0 <= grid_x < map_data.grid_width and 0 <= grid_y < map_data.grid_height:
                elevation = map_data.heightmap[grid_y, grid_x]
                
                # Start tunnel if elevation is very high
                if elevation > 0.8 and tunnel_start is None:
                    tunnel_start = (x, y)
                
                # End tunnel if elevation drops or we reach the end
                elif elevation <= 0.8 and tunnel_start is not None:
                    tunnel_segments.append((tunnel_start, (x, y)))
                    tunnel_start = None
        
        # Close any remaining tunnel at the end
        if tunnel_start is not None and road_coords:
            tunnel_segments.append((tunnel_start, road_coords[-1]))
        
        return tunnel_segments
    
    def _find_railway_cities(self, map_data: MapData) -> List[Tuple[float, float]]:
        """Find major cities that should be connected by railway."""
        railway_cities = []
        
        # Major districts that warrant railway connections
        priority_types = ['downtown', 'industrial', 'commercial', 'airport', 'port']
        
        for district in map_data.districts.values():
            if district.district_type in priority_types:
                railway_cities.append(district.center)
        
        return railway_cities
    
    def _generate_main_railway_lines(self, map_data: MapData, config: TransportationConfig, cities: List[Tuple[float, float]]):
        """Generate main railway lines connecting major cities."""
        
        if len(cities) < 2:
            return
        
        # Connect cities with railway using minimum spanning tree approach
        connected = {0}
        railway_id = 0
        
        while len(connected) < len(cities) and railway_id < 6:
            best_connection = None
            best_distance = float('inf')
            
            for i in connected:
                for j in range(len(cities)):
                    if j not in connected:
                        distance = math.sqrt(
                            (cities[i][0] - cities[j][0])**2 + 
                            (cities[i][1] - cities[j][1])**2
                        )
                        if distance < best_distance:
                            best_distance = distance
                            best_connection = (i, j)
            
            if best_connection:
                start_city = cities[best_connection[0]]
                end_city = cities[best_connection[1]]
                
                railway = self._generate_railway_line(
                    map_data, start_city, end_city, f"railway_main_{railway_id}", config
                )
                
                if railway:
                    map_data.add_road(railway)  # Railways stored as roads with special type
                    self.railway_network.append(railway)
                    connected.add(best_connection[1])
                    railway_id += 1
    
    def _generate_branch_railway_lines(self, map_data: MapData, config: TransportationConfig):
        """Generate branch railway lines to industrial areas."""
        
        # Find industrial districts not yet connected
        industrial_districts = [d for d in map_data.districts.values() if d.district_type == 'industrial']
        
        for i, district in enumerate(industrial_districts[:3]):  # Limit to 3 branches
            # Find nearest main railway line
            nearest_point = self._find_nearest_railway_point(district.center)
            
            if nearest_point:
                branch = self._generate_railway_line(
                    map_data, nearest_point, district.center, f"railway_branch_{i}", config
                )
                
                if branch:
                    map_data.add_road(branch)
                    self.railway_network.append(branch)
    
    def _generate_railway_line(self, map_data: MapData, start: Tuple[float, float], end: Tuple[float, float], railway_id: str, config: TransportationConfig) -> Optional[Road]:
        """Generate a railway line between two points."""
        
        # Railways prefer gentle grades, so use elevation-aware pathfinding
        points = self._pathfind_with_elevation(map_data, start, end, 'railway')
        
        if points and len(points) >= 2:
            return Road(
                id=railway_id,
                points=points,
                road_type='railway',
                width=6,  # Standard railway width
                color='#8B4513'  # Brown color for railways
            )
        
        return None
    
    def _generate_railway_bridges(self, map_data: MapData, config: TransportationConfig):
        """Generate bridges for railway water crossings."""
        
        bridge_count = len(self.bridges)  # Continue numbering from road bridges
        
        for railway in self.railway_network:
            if not railway.linestring:
                continue
            
            # Check intersections with water bodies
            for water_id, water_body in map_data.water_bodies.items():
                intersections = self._find_road_water_intersections(railway, water_body)
                
                for intersection in intersections:
                    bridge = self._create_railway_bridge_at_crossing(
                        railway, water_body, intersection, f"railway_bridge_{bridge_count}", config
                    )
                    if bridge:
                        self.bridges.append(bridge)
                        bridge_count += 1
    
    def _create_railway_bridge_at_crossing(self, railway: Road, water_body, intersection_point: Point, bridge_id: str, config: TransportationConfig) -> Optional[Bridge]:
        """Create a railway bridge at a water crossing."""
        
        if not railway.linestring:
            return None
        
        # Similar to road bridge creation but with railway-specific parameters
        road_coords = list(railway.linestring.coords)
        crossing_x, crossing_y = intersection_point.x, intersection_point.y
        
        # Find the segment that crosses water
        min_dist = float('inf')
        segment_idx = 0
        
        for i in range(len(road_coords) - 1):
            segment = LineString([road_coords[i], road_coords[i + 1]])
            dist = segment.distance(intersection_point)
            if dist < min_dist:
                min_dist = dist
                segment_idx = i
        
        if segment_idx < len(road_coords) - 1:
            start_point = road_coords[segment_idx]
            end_point = road_coords[segment_idx + 1]
            
            bridge_length = math.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
            
            return Bridge(
                id=bridge_id,
                start_point=start_point,
                end_point=end_point,
                bridge_type='railway',
                water_body_id=water_body.id,
                length=bridge_length,
                width=8,  # Railway bridge width
                color='#654321'  # Darker brown for railway bridges
            )
        
        return None
    
    def _find_nearest_railway_point(self, target_point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Find the nearest point on an existing railway line."""
        
        if not self.railway_network:
            return None
        
        min_distance = float('inf')
        nearest_point = None
        
        for railway in self.railway_network:
            if railway.linestring:
                # Find nearest point on this railway line
                point = Point(target_point)
                nearest = nearest_points(railway.linestring, point)[0]
                distance = point.distance(nearest)
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_point = (nearest.x, nearest.y)
        
        return nearest_point


# Backward compatibility with existing system
class TransportationGenerator:
    """Legacy transportation generator - now uses advanced system internally."""
    
    def __init__(self):
        self.advanced_generator = AdvancedTransportationGenerator()
    
    def generate_highway_network(self, map_data: MapData, config: TransportationConfig):
        """Generate highway network using advanced system."""
        self.advanced_generator.generate_highway_network(map_data, config)
    
    def generate_arterial_grid(self, map_data: MapData, config: TransportationConfig):
        """Generate arterial grid using advanced system."""
        self.advanced_generator.generate_arterial_grid(map_data, config)
    
    def generate_local_roads(self, map_data: MapData, config: TransportationConfig):
        """Generate local roads using advanced system."""
        self.advanced_generator.generate_local_roads(map_data, config)
        
        # Also generate bridges, tunnels, and railways
        self.advanced_generator.generate_bridges(map_data, config)
        self.advanced_generator.generate_tunnels(map_data, config)
        self.advanced_generator.generate_railway_network(map_data, config) 