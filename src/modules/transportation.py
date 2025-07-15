"""
Advanced transportation generation module.
Handles road networks, highways, railways, bridges, and transportation infrastructure with terrain awareness.
"""
import random
import math
from typing import List, Tuple, Optional, Dict, Set, Any
import numpy as np
from noise import pnoise2
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points, unary_union
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
        self.arterial_spine = []
    
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
        
        print("  → Arterial road network complete")
    
    def generate_local_roads(self, map_data: MapData, config: TransportationConfig):
        """
        Generate local roads and collectors with realistic patterns.
        """
        print("  → Generating local road network...")
        
        # Generate collector roads between arterials (proper hierarchy)
        self._generate_collector_roads(map_data, config)
        
        # Generate local roads in districts with organic patterns (much less dense)
        self._generate_organic_local_roads(map_data, config)
        
        # Generate rural roads for less developed areas
        self._generate_rural_roads(map_data, config)
        
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
        """
        Generate organic arterial roads that follow terrain contours and natural features.
        """
        print("  → Generating organic arterial network...")
        
        # Find natural arterial starting points (major settlements, water access points)
        arterial_starts = self._find_natural_arterial_starts(map_data)
        
        # Create organic arterial network connecting major points
        arterial_count = 0
        max_arterials = min(4, len(arterial_starts))  # Reduced from 8 to 4
        
        for i in range(max_arterials):
            start_point = arterial_starts[i]
            
            # Find the best destination point
            best_end = self._find_natural_arterial_end(map_data, start_point)
            
            if best_end:
                # Create organic arterial road
                arterial = self._create_organic_arterial_road(
                    map_data, start_point, best_end, f"arterial_{arterial_count}", config
                )
                
                if arterial:
                    map_data.add_road(arterial)
                    self.arterial_spine.append(arterial)
                    arterial_count += 1
        
        # Create secondary arterials that branch off main ones (reduced)
        self._create_secondary_arterials(map_data, config)
        
        # --- NEW: Realistic, natural connections ---
        self._connect_settlements_realistically(map_data, config)
        
        print(f"  → Generated {arterial_count} organic arterial roads")
    
    def _connect_settlements_realistically(self, map_data: MapData, config: TransportationConfig):
        """Connect settlements in a realistic, natural way."""
        from shapely.geometry import Point
        # Gather settlements by type
        cities_towns = []
        villages = []
        for district in map_data.districts.values():
            if district.district_type in ['major_city', 'city', 'town', 'downtown', 'commercial', 'industrial', 'port', 'airport']:
                cities_towns.append((district.id, district.center))
            elif district.district_type in ['village', 'residential', 'suburban']:
                villages.append((district.id, district.center))
        # Connect each city/town to 1–2 nearest other cities/towns (if within 1500 units)
        for i, (id1, center1) in enumerate(cities_towns):
            dists = []
            for j, (id2, center2) in enumerate(cities_towns):
                if i != j:
                    dist = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
                    dists.append((dist, id2, center2))
            dists = [d for d in dists if d[0] < 1500]
            dists.sort()
            for _, id2, center2 in dists[:2]:
                if id1 < id2:
                    road_id = f"realistic_link_{id1}_{id2}"
                    road = self._create_organic_arterial_road(
                        map_data, center1, center2, road_id, config
                    )
                    if road:
                        map_data.add_road(road)
        # Connect each village to the nearest city/town by collector
        for id1, center1 in villages:
            best = None
            best_dist = float('inf')
            for id2, center2 in cities_towns:
                dist = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
                if dist < best_dist:
                    best = (id2, center2)
                    best_dist = dist
            if best and best_dist < 2000:
                id2, center2 = best
                road_id = f"realistic_collector_{id1}_{id2}"
                road = self._create_organic_arterial_road(
                    map_data, center1, center2, road_id, config
                )
                if road:
                    road.road_type = 'collector'
                    map_data.add_road(road)
    
    def _generate_collector_roads(self, map_data: MapData, config: TransportationConfig):
        """Generate collector roads that connect arterials with proper hierarchy."""
        print("    → Building collector road network...")
        
        # Get all arterial roads
        arterials = [road for road in map_data.roads.values() if road.road_type == 'arterial']
        
        if len(arterials) < 2:
            return
        
        # Create collector roads with much better spacing
        collector_count = 0
        max_collectors = 10  # Allow up to 10 collectors
        
        # Create collectors between arterials with proper spacing
        for i, arterial1 in enumerate(arterials):
            if collector_count >= max_collectors:
                break
            
            # Find best arterial to connect to (not adjacent ones)
            best_arterial = None
            best_distance = 0
            
            for j, arterial2 in enumerate(arterials[i+2:], i+2):  # Skip adjacent arterials
                # Check distance between arterials
                distance = self._calculate_arterial_distance(arterial1, arterial2)
                
                if 250 < distance < 1500 and distance > best_distance:  # Allow more connections, wider range
                    best_distance = distance
                    best_arterial = arterial2
            
            if best_arterial:
                # Create collector between these arterials
                connections = self._find_arterial_connection_points(arterial1, best_arterial)
                
                for start_point, end_point in connections[:3]:  # Allow up to 3 connections
                    collector_path = self._create_organic_path(map_data, start_point, end_point, 'collector')
                    
                    if collector_path and len(collector_path) > 2:
                        collector = Road(
                            id=f"collector_{collector_count}",
                            points=collector_path,
                            road_type='collector',
                            width=config.road_styles['collector']['width'],
                            color=config.road_styles['collector']['color']
                        )
                        map_data.add_road(collector)
                        collector_count += 1
        
        print(f"    → Created {collector_count} collector roads")
    
    def _generate_rural_roads(self, map_data: MapData, config: TransportationConfig):
        """Generate rural roads for less developed areas."""
        print("    → Generating rural road network...")
        
        rural_count = 0
        max_rural = 4
        
        # Find areas without major roads
        for y in range(200, map_data.height - 200, 300):  # Much larger spacing
            for x in range(200, map_data.width - 200, 300):
                
                # Check if this area is on land and not near major roads
                grid_x = int(x / map_data.grid_size)
                grid_y = int(y / map_data.grid_size)
                
                if (0 <= grid_x < map_data.grid_width and 
                    0 <= grid_y < map_data.grid_height and
                    map_data.land_mask[grid_y, grid_x]):
                    
                    # Check distance to nearest major road
                    nearest_road_distance = self._find_nearest_road_distance(map_data, (x, y))
                    
                    if nearest_road_distance > 200:  # Only if far from major roads
                        # Create rural road
                        rural_points = self._create_rural_road_path(map_data, (x, y))
                        
                        if len(rural_points) > 2:
                            rural_road = Road(
                                id=f"rural_{rural_count}",
                                points=rural_points,
                                road_type='rural',
                                width=config.road_styles['rural']['width'],
                                color=config.road_styles['rural']['color']
                            )
                            map_data.add_road(rural_road)
                            rural_count += 1
                            
                            if rural_count >= max_rural:
                                break
        
        print(f"    → Created {rural_count} rural roads")
    
    def _find_nearest_road_distance(self, map_data: MapData, point: Tuple[float, float]) -> float:
        """Find distance to nearest major road."""
        min_distance = float('inf')
        
        for road in map_data.roads.values():
            if road.road_type in ['highway', 'arterial', 'collector'] and road.points:
                for road_point in road.points:
                    distance = self._distance(point, road_point)
                    min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else 1000
    
    def _create_rural_road_path(self, map_data: MapData, start_point: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Create a rural road path that follows terrain naturally."""
        import math
        import random
        
        # Create a winding rural road
        length = random.uniform(150, 300)
        angle = random.uniform(0, 2 * math.pi)
        
        points = [start_point]
        num_segments = 6
        
        for i in range(1, num_segments + 1):
            t = i / num_segments
            
            # Winding path with gentle curves
            curve_angle = angle + math.sin(t * math.pi * 1.5) * 0.6
            segment_length = length / num_segments
            
            x = points[-1][0] + math.cos(curve_angle) * segment_length
            y = points[-1][1] + math.sin(curve_angle) * segment_length
            
            # Add terrain influence
            terrain_influence = self._get_terrain_influence(map_data, (x, y))
            terrain_x = math.cos(terrain_influence) * 20
            terrain_y = math.sin(terrain_influence) * 20
            
            final_x = x + terrain_x
            final_y = y + terrain_y
            
            # Ensure within map bounds
            final_x = max(50, min(map_data.width - 50, final_x))
            final_y = max(50, min(map_data.height - 50, final_y))
            
            points.append((final_x, final_y))
        
        return self._apply_organic_smoothing(points, 'rural')
    
    def _generate_organic_local_roads(self, map_data: MapData, config: TransportationConfig):
        """Generate organic local roads that follow natural patterns."""
        
        print("  → Creating organic local road networks...")
        
        # Generate organic local roads for districts with road access
        for district in map_data.districts.values():
            if district.polygon and district.district_type in ['residential', 'suburban', 'commercial']:
                # Check if district is connected to road network
                if self._district_has_road_access(map_data, district):
                    self._generate_organic_district_roads(map_data, district, config)
    
    def _district_has_road_access(self, map_data: MapData, district) -> bool:
        """Check if a district has reasonable access to the road network."""
        import math
        
        if not district.polygon:
            return False
        
        district_center = district.polygon.centroid
        
        # Check for nearby arterials or collectors
        main_roads = [road for road in map_data.roads.values() 
                     if road.road_type in ['arterial', 'collector']]
        
        for road in main_roads:
            if not road.points:
                continue
            
            for point in road.points:
                distance = math.sqrt(
                    (district_center.x - point[0])**2 + 
                    (district_center.y - point[1])**2
                )
                if distance < 500:  # Within 500 units of main road
                    return True
        
        return False
    
    def _generate_organic_district_roads(self, map_data: MapData, district, config: TransportationConfig):
        """Generate organic local roads within a district."""
        import math
        import random
        from shapely.geometry import Point
        
        if not district.polygon:
            return
        
        minx, miny, maxx, maxy = district.polygon.bounds
        district_size = max(maxx - minx, maxy - miny)
        
        # Only add local roads if district is large enough
        if district_size < 200:
            return
        
        road_id = 0
        
        # Create organic patterns based on district type
        if district.district_type == 'residential':
            self._create_organic_residential_pattern(map_data, district, config, road_id)
        elif district.district_type == 'suburban':
            self._create_organic_suburban_pattern(map_data, district, config, road_id)
        elif district.district_type == 'commercial':
            self._create_organic_commercial_pattern(map_data, district, config, road_id)
    
    def _create_residential_grid(self, map_data: MapData, district, config: TransportationConfig, start_id: int):
        """Create a simple residential grid pattern."""
        from shapely.geometry import Point
        
        minx, miny, maxx, maxy = district.polygon.bounds
        
        # Create 2-3 main residential streets
        num_streets = 2 if (maxx - minx) < 300 else 3
        
        for i in range(num_streets):
            # Create main street through district
            y_pos = miny + (i + 1) * (maxy - miny) / (num_streets + 1)
            
            main_street_points = []
            for x in range(int(minx + 20), int(maxx - 20), 50):
                if district.polygon.contains(Point(x, y_pos)):
                    main_street_points.append((x, y_pos))
            
            if len(main_street_points) > 2:
                # Add slight curves for natural feel
                curved_points = []
                for j, point in enumerate(main_street_points):
                    curve_offset = math.sin(j * 0.5) * 15
                    curved_points.append((point[0], point[1] + curve_offset))
                
                main_street = Road(
                    id=f"residential_main_{district.id}_{start_id + i}",
                    points=self._smooth_path(curved_points),
                    road_type='local',
                    width=config.road_styles['local']['width'],
                    color=config.road_styles['local']['color']
                )
                map_data.add_road(main_street)
                
                # Add 1-2 side streets
                for side_idx in range(2):
                    x_pos = int(minx + (side_idx + 1) * (maxx - minx) / 3)
                    
                    side_points = []
                    for y in range(int(y_pos - 80), int(y_pos + 80), 40):
                        if district.polygon.contains(Point(x_pos, y)):
                            side_points.append((x_pos, y))
                    
                    if len(side_points) > 1:
                        side_street = Road(
                            id=f"residential_side_{district.id}_{start_id + i}_{side_idx}",
                            points=side_points,
                            road_type='local',
                            width=config.road_styles['local']['width'],
                            color=config.road_styles['local']['color']
                        )
                        map_data.add_road(side_street)
    
    def _create_suburban_pattern(self, map_data: MapData, district, config: TransportationConfig, start_id: int):
        """Create curved suburban roads."""
        import math
        from shapely.geometry import Point
        
        minx, miny, maxx, maxy = district.polygon.bounds
        center_x, center_y = (minx + maxx) / 2, (miny + maxy) / 2
        
        # Create 1-2 curved suburban streets
        num_curves = 1 if (maxx - minx) < 300 else 2
        
        for i in range(num_curves):
            # Create curved path through district
            start_y = miny + (i + 1) * (maxy - miny) / (num_curves + 1)
            
            curved_points = []
            for x in range(int(minx + 30), int(maxx - 30), 40):
                # Create gentle S-curve
                curve_offset = math.sin((x - minx) * 0.01) * 40
                y_pos = start_y + curve_offset
                
                if district.polygon.contains(Point(x, y_pos)):
                    curved_points.append((x, y_pos))
            
            if len(curved_points) > 3:
                suburban_street = Road(
                    id=f"suburban_curve_{district.id}_{start_id + i}",
                    points=self._smooth_path(curved_points),
                    road_type='local',
                    width=config.road_styles['local']['width'],
                    color=config.road_styles['local']['color']
                )
                map_data.add_road(suburban_street)
    
    def _create_commercial_access(self, map_data: MapData, district, config: TransportationConfig, start_id: int):
        """Create commercial access roads."""
        from shapely.geometry import Point
        
        minx, miny, maxx, maxy = district.polygon.bounds
        
        # Create main commercial avenue
        center_y = (miny + maxy) / 2
        
        avenue_points = []
        for x in range(int(minx + 20), int(maxx - 20), 30):
            if district.polygon.contains(Point(x, center_y)):
                avenue_points.append((x, center_y))
        
        if len(avenue_points) > 2:
            commercial_avenue = Road(
                id=f"commercial_avenue_{district.id}_{start_id}",
                points=avenue_points,
                road_type='local',
                width=config.road_styles['local']['width'] * 1.2,  # Wider
                color=config.road_styles['local']['color']
            )
            map_data.add_road(commercial_avenue)
            
            # Add perpendicular access roads
            for i, point in enumerate(avenue_points[1:-1:2]):  # Every other point
                # Short perpendicular access
                access_points = [
                    point,
                    (point[0], point[1] - 60),
                    (point[0], point[1] + 60)
                ]
                
                # Filter points that are in district
                valid_access = [p for p in access_points if district.polygon.contains(Point(p[0], p[1]))]
                
                if len(valid_access) > 1:
                    access_road = Road(
                        id=f"commercial_access_{district.id}_{start_id}_{i}",
                        points=valid_access,
                        road_type='local',
                        width=config.road_styles['local']['width'],
                        color=config.road_styles['local']['color']
                    )
                    map_data.add_road(access_road)
    
    def _add_cul_de_sac(self, map_data: MapData, district, branch_point: Tuple[float, float], config: TransportationConfig, road_id: int):
        """Add a cul-de-sac branching from a main road."""
        import math
        import random
        from shapely.geometry import Point
        
        # Generate cul-de-sac direction
        angle = random.uniform(0, 2 * math.pi)
        length = random.uniform(60, 100)
        
        # Cul-de-sac center
        center_x = branch_point[0] + math.cos(angle) * length
        center_y = branch_point[1] + math.sin(angle) * length
        
        if not district.polygon.contains(Point(center_x, center_y)):
            return
        
        # Create circular end
        circle_points = [branch_point]
        radius = 25
        
        for i in range(8):
            circle_angle = i * 2 * math.pi / 8
            x = center_x + math.cos(circle_angle) * radius
            y = center_y + math.sin(circle_angle) * radius
            
            if district.polygon.contains(Point(x, y)):
                circle_points.append((x, y))
        
        circle_points.append(branch_point)  # Close the circle
        
        if len(circle_points) > 3:
            cul_de_sac = Road(
                id=f"cul_de_sac_{district.id}_{road_id}",
                points=circle_points,
                road_type='local',
                width=config.road_styles['local']['width'],
                color=config.road_styles['local']['color']
            )
            map_data.add_road(cul_de_sac)
    
    def _smooth_path(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Smooth a path using simple moving average."""
        if len(points) < 3:
            return points
        
        smoothed = [points[0]]  # Keep first point
        
        # Apply smoothing to intermediate points
        for i in range(1, len(points) - 1):
            # Simple 3-point average
            prev_x, prev_y = points[i-1]
            curr_x, curr_y = points[i]
            next_x, next_y = points[i+1]
            
            smooth_x = (prev_x + curr_x + next_x) / 3
            smooth_y = (prev_y + curr_y + next_y) / 3
            
            smoothed.append((smooth_x, smooth_y))
        
        smoothed.append(points[-1])  # Keep last point
        return smoothed
    
    def _offset_path(self, points: List[Tuple[float, float]], offset_distance: float) -> List[Tuple[float, float]]:
        """Create an offset path parallel to the original path."""
        import math
        
        if len(points) < 2:
            return points
        
        offset_points = []
        
        for i in range(len(points)):
            if i == 0:
                # First point: use direction to next point
                dx = points[1][0] - points[0][0]
                dy = points[1][1] - points[0][1]
            elif i == len(points) - 1:
                # Last point: use direction from previous point
                dx = points[i][0] - points[i-1][0]
                dy = points[i][1] - points[i-1][1]
            else:
                # Middle points: use average direction
                dx1 = points[i][0] - points[i-1][0]
                dy1 = points[i][1] - points[i-1][1]
                dx2 = points[i+1][0] - points[i][0]
                dy2 = points[i+1][1] - points[i][1]
                dx = (dx1 + dx2) / 2
                dy = (dy1 + dy2) / 2
            
            # Normalize direction
            length = math.sqrt(dx**2 + dy**2)
            if length > 0:
                dx /= length
                dy /= length
            
            # Calculate perpendicular offset
            offset_x = points[i][0] + dy * offset_distance
            offset_y = points[i][1] - dx * offset_distance
            
            offset_points.append((offset_x, offset_y))
        
        return offset_points
    
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
    
    def _apply_spline_smoothing(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Apply spline smoothing to create natural curves."""
        import numpy as np
        from scipy import interpolate
        
        if len(points) < 3:
            return points
        
        # Extract x and y coordinates
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        # Create parameter array
        t = np.linspace(0, 1, len(points))
        
        try:
            # Create spline interpolation
            cs_x = interpolate.CubicSpline(t, x_coords, bc_type='natural')
            cs_y = interpolate.CubicSpline(t, y_coords, bc_type='natural')
            
            # Generate more points for smooth curves
            t_new = np.linspace(0, 1, len(points) * 2)
            
            smooth_x = cs_x(t_new)
            smooth_y = cs_y(t_new)
            
            # Convert back to list of tuples
            smoothed_points = [(float(x), float(y)) for x, y in zip(smooth_x, smooth_y)]
            
            return smoothed_points
            
        except Exception:
            # Fallback to simple smoothing if spline fails
            return self._simple_smooth_path(points)
    
    def _simple_smooth_path(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Simple smoothing as fallback."""
        if len(points) < 3:
            return points
        
        smoothed = [points[0]]
        
        for i in range(1, len(points) - 1):
            # Weighted average of surrounding points
            prev_x, prev_y = points[i-1]
            curr_x, curr_y = points[i]
            next_x, next_y = points[i+1]
            
            smooth_x = (prev_x * 0.25 + curr_x * 0.5 + next_x * 0.25)
            smooth_y = (prev_y * 0.25 + curr_y * 0.5 + next_y * 0.25)
            
            smoothed.append((smooth_x, smooth_y))
        
        smoothed.append(points[-1])
        return smoothed
    
    def _get_road_direction_at_point(self, road_points: List[Tuple[float, float]], 
                                   point: Tuple[float, float]) -> Tuple[float, float]:
        """Get the direction of a road at a specific point."""
        import math
        
        # Find closest point on road
        min_distance = float('inf')
        closest_index = 0
        
        for i, road_point in enumerate(road_points):
            distance = math.sqrt((road_point[0] - point[0])**2 + (road_point[1] - point[1])**2)
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        
        # Calculate direction at this point
        if closest_index < len(road_points) - 1:
            next_point = road_points[closest_index + 1]
            current_point = road_points[closest_index]
        elif closest_index > 0:
            current_point = road_points[closest_index - 1]
            next_point = road_points[closest_index]
        else:
            return (1.0, 0.0)  # Default direction
        
        dx = next_point[0] - current_point[0]
        dy = next_point[1] - current_point[1]
        length = math.sqrt(dx**2 + dy**2)
        
        if length > 0:
            return (dx/length, dy/length)
        return (1.0, 0.0)
    
    def _find_nearest_road_point(self, point: Tuple[float, float], 
                                roads: List) -> Tuple[any, Tuple[float, float]]:
        """Find the nearest point on any of the given roads."""
        import math
        
        min_distance = float('inf')
        nearest_road = None
        nearest_point = None
        
        for road in roads:
            if not road.points:
                continue
            
            for road_point in road.points:
                distance = math.sqrt(
                    (road_point[0] - point[0])**2 + (road_point[1] - point[1])**2
                )
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_road = road
                    nearest_point = road_point
        
        return nearest_road, nearest_point
    
    def _get_terrain_influence(self, map_data: MapData, point: Tuple[float, float]) -> float:
        """Get simplified terrain influence on road direction."""
        import math
        
        if map_data.heightmap is None:
            return 0
        
        x, y = point
        grid_x = int(x / map_data.grid_size)
        grid_y = int(y / map_data.grid_size)
        
        if not (0 <= grid_x < map_data.grid_width-1 and 0 <= grid_y < map_data.grid_height-1):
            return 0
        
        # Calculate slope direction (roads prefer gentler slopes)
        dx = map_data.heightmap[grid_y, grid_x+1] - map_data.heightmap[grid_y, grid_x-1] if grid_x > 0 else 0
        dy = map_data.heightmap[grid_y+1, grid_x] - map_data.heightmap[grid_y-1, grid_x] if grid_y > 0 else 0
        
        # Return slight influence to avoid steep slopes
        if abs(dx) > 0.2 or abs(dy) > 0.2:
            slope_angle = math.atan2(dy, dx)
            return slope_angle * 0.1  # Very small influence
        
        return 0
    
    def _create_organic_residential_pattern(self, map_data: MapData, district, config: TransportationConfig, start_id: int):
        """Create an organic residential pattern with curved streets and cul-de-sacs."""
        import math
        import random
        from shapely.geometry import Point
        
        minx, miny, maxx, maxy = district.polygon.bounds
        district_center = district.polygon.centroid
        
        # Create organic main streets that follow terrain
        main_streets = []
        
        # Create only 1 main organic street (reduced from 1-2)
        num_streets = 1
        
        for i in range(num_streets):
            # Create organic main street
            start_point = (minx + 20, miny + (maxy - miny) / 2)
            end_point = (maxx - 20, miny + (maxy - miny) / 2)
            
            # Create organic path with curves
            street_points = self._create_organic_street_path(map_data, start_point, end_point, district)
            
            if len(street_points) > 2:
                street = Road(
                    id=f"organic_residential_main_{start_id + i}",
                    points=street_points,
                    road_type='local',
                    width=config.road_styles['local']['width'],
                    color=config.road_styles['local']['color']
                )
                map_data.add_road(street)
                main_streets.append(street)
        
        # Add organic side streets and cul-de-sacs (much reduced)
        self._add_organic_side_streets(map_data, district, main_streets, config, start_id + num_streets)
    
    def _add_organic_side_streets(self, map_data: MapData, district, main_streets, config: TransportationConfig, start_id: int):
        """Add organic side streets and cul-de-sacs to residential areas."""
        import math
        import random
        from shapely.geometry import Point
        
        road_id = start_id
        
        for main_street in main_streets:
            if not main_street.points or len(main_street.points) < 3:
                continue
            
            # Add side streets at intervals (much increased spacing)
            for i in range(3, len(main_street.points) - 3, 10):  # Increased from 6 to 10
                branch_point = main_street.points[i]
                
                # Create organic cul-de-sac
                cul_de_sac_points = self._create_organic_cul_de_sac(map_data, district, branch_point)
                
                if len(cul_de_sac_points) > 2:
                    cul_de_sac = Road(
                        id=f"organic_cul_de_sac_{road_id}",
                        points=cul_de_sac_points,
                        road_type='local',
                        width=config.road_styles['local']['width'] * 0.8,  # Narrower for cul-de-sacs
                        color=config.road_styles['local']['color']
                    )
                    map_data.add_road(cul_de_sac)
                    road_id += 1
                    
                    # Limit cul-de-sacs per main street
                    if road_id - start_id >= 2:
                        break
    
    def _add_organic_feeder_roads(self, map_data: MapData, district, loop_roads, config: TransportationConfig, start_id: int):
        """Add organic feeder roads to suburban areas."""
        import math
        import random
        from shapely.geometry import Point
        
        road_id = start_id
        
        for loop_road in loop_roads:
            if not loop_road.points or len(loop_road.points) < 4:
                continue
            
            # Add feeder roads at intervals (much increased spacing)
            for i in range(4, len(loop_road.points) - 4, 12):  # Increased from 8 to 12
                branch_point = loop_road.points[i]
                
                # Create organic feeder road
                feeder_points = self._create_organic_feeder_road(map_data, district, branch_point)
                
                if len(feeder_points) > 2:
                    feeder_road = Road(
                        id=f"organic_feeder_{road_id}",
                        points=feeder_points,
                        road_type='local',
                        width=config.road_styles['local']['width'],
                        color=config.road_styles['local']['color']
                    )
                    map_data.add_road(feeder_road)
                    road_id += 1
                    
                    # Limit feeder roads per loop
                    if road_id - start_id >= 2:
                        break
    
    def _create_organic_suburban_pattern(self, map_data: MapData, district, config: TransportationConfig, start_id: int):
        """Create an organic suburban pattern with winding roads and natural curves."""
        import math
        import random
        from shapely.geometry import Point
        
        minx, miny, maxx, maxy = district.polygon.bounds
        district_center = district.polygon.centroid
        
        # Create organic loop roads
        loop_roads = []
        
        # Create 1 main loop road (reduced from 1-2)
        num_loops = 1
        
        for i in range(num_loops):
            # Create organic loop road
            loop_points = self._create_organic_loop_road(map_data, district, i)
            
            if len(loop_points) > 3:
                loop_road = Road(
                    id=f"organic_suburban_loop_{start_id + i}",
                    points=loop_points,
                    road_type='local',
                    width=config.road_styles['local']['width'],
                    color=config.road_styles['local']['color']
                )
                map_data.add_road(loop_road)
                loop_roads.append(loop_road)
        
        # Add organic feeder roads (reduced)
        self._add_organic_feeder_roads(map_data, district, loop_roads, config, start_id + num_loops)
    
    def _create_commercial_access(self, map_data: MapData, district, config: TransportationConfig, start_id: int):
        """Create commercial access roads."""
        from shapely.geometry import Point
        
        minx, miny, maxx, maxy = district.polygon.bounds
        
        # Create main commercial avenue
        center_y = (miny + maxy) / 2
        
        avenue_points = []
        for x in range(int(minx + 20), int(maxx - 20), 30):
            if district.polygon.contains(Point(x, center_y)):
                avenue_points.append((x, center_y))
        
        if len(avenue_points) > 2:
            commercial_avenue = Road(
                id=f"commercial_avenue_{district.id}_{start_id}",
                points=avenue_points,
                road_type='local',
                width=config.road_styles['local']['width'] * 1.2,  # Wider
                color=config.road_styles['local']['color']
            )
            map_data.add_road(commercial_avenue)
            
            # Add perpendicular access roads
            for i, point in enumerate(avenue_points[1:-1:2]):  # Every other point
                # Short perpendicular access
                access_points = [
                    point,
                    (point[0], point[1] - 60),
                    (point[0], point[1] + 60)
                ]
                
                # Filter points that are in district
                valid_access = [p for p in access_points if district.polygon.contains(Point(p[0], p[1]))]
                
                if len(valid_access) > 1:
                    access_road = Road(
                        id=f"commercial_access_{district.id}_{start_id}_{i}",
                        points=valid_access,
                        road_type='local',
                        width=config.road_styles['local']['width'],
                        color=config.road_styles['local']['color']
                    )
                    map_data.add_road(access_road)
    
    def _add_organic_parking_roads(self, map_data: MapData, district, access_roads, config: TransportationConfig, start_id: int):
        """Add organic parking access roads to commercial areas."""
        import math
        import random
        from shapely.geometry import Point
        
        road_id = start_id
        
        for access_road in access_roads:
            if not access_road.points or len(access_road.points) < 3:
                continue
            
            # Add parking access roads at intervals
            for i in range(1, len(access_road.points) - 1, 2):
                branch_point = access_road.points[i]
                
                # Create organic parking access road
                parking_points = self._create_organic_parking_road(map_data, district, branch_point)
                
                if len(parking_points) > 2:
                    parking_road = Road(
                        id=f"organic_parking_{road_id}",
                        points=parking_points,
                        road_type='local',
                        width=config.road_styles['local']['width'] * 0.9,
                        color=config.road_styles['local']['color']
                    )
                    map_data.add_road(parking_road)
                    road_id += 1
    
    def _create_organic_cul_de_sac(self, map_data: MapData, district, start_point: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Create an organic cul-de-sac."""
        import math
        import random
        from shapely.geometry import Point
        
        # Create curved path ending in a loop
        length = random.uniform(60, 120)
        angle = random.uniform(0, 2 * math.pi)
        
        # Create curved path
        points = [start_point]
        num_segments = 6
        
        for i in range(1, num_segments + 1):
            t = i / num_segments
            
            # Curved path
            curve_angle = angle + math.sin(t * math.pi) * 0.5
            segment_length = length / num_segments
            
            x = points[-1][0] + math.cos(curve_angle) * segment_length
            y = points[-1][1] + math.sin(curve_angle) * segment_length
            
            # Add terrain influence
            terrain_influence = self._get_terrain_influence(map_data, (x, y))
            terrain_x = math.cos(terrain_influence) * 10
            terrain_y = math.sin(terrain_influence) * 10
            
            final_x = x + terrain_x
            final_y = y + terrain_y
            
            # Ensure point is within district
            if district.polygon.contains(Point(final_x, final_y)):
                points.append((final_x, final_y))
        
        # Add small loop at end
        if len(points) > 2:
            end_point = points[-1]
            loop_radius = 15
            
            for i in range(4):
                angle = (i / 4) * 2 * math.pi
                loop_x = end_point[0] + loop_radius * math.cos(angle)
                loop_y = end_point[1] + loop_radius * math.sin(angle)
                
                if district.polygon.contains(Point(loop_x, loop_y)):
                    points.append((loop_x, loop_y))
        
        return self._apply_organic_smoothing(points, 'local')
    
    def _create_organic_feeder_road(self, map_data: MapData, district, start_point: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Create an organic feeder road."""
        import math
        import random
        from shapely.geometry import Point
        
        # Create winding path
        length = random.uniform(80, 150)
        angle = random.uniform(0, 2 * math.pi)
        
        points = [start_point]
        num_segments = 8
        
        for i in range(1, num_segments + 1):
            t = i / num_segments
            
            # Winding path with multiple curves
            curve_angle = angle + math.sin(t * math.pi * 2) * 0.8
            segment_length = length / num_segments
            
            x = points[-1][0] + math.cos(curve_angle) * segment_length
            y = points[-1][1] + math.sin(curve_angle) * segment_length
            
            # Add terrain influence
            terrain_influence = self._get_terrain_influence(map_data, (x, y))
            terrain_x = math.cos(terrain_influence) * 15
            terrain_y = math.sin(terrain_influence) * 15
            
            final_x = x + terrain_x
            final_y = y + terrain_y
            
            # Ensure point is within district
            if district.polygon.contains(Point(final_x, final_y)):
                points.append((final_x, final_y))
        
        return self._apply_organic_smoothing(points, 'local')
    
    def _create_organic_parking_road(self, map_data: MapData, district, start_point: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Create an organic parking access road."""
        import math
        import random
        from shapely.geometry import Point
        
        # Create short access road
        length = random.uniform(40, 80)
        angle = random.uniform(0, 2 * math.pi)
        
        points = [start_point]
        num_segments = 4
        
        for i in range(1, num_segments + 1):
            t = i / num_segments
            
            # Simple curved path
            curve_angle = angle + math.sin(t * math.pi) * 0.3
            segment_length = length / num_segments
            
            x = points[-1][0] + math.cos(curve_angle) * segment_length
            y = points[-1][1] + math.sin(curve_angle) * segment_length
            
            # Add terrain influence
            terrain_influence = self._get_terrain_influence(map_data, (x, y))
            terrain_x = math.cos(terrain_influence) * 8
            terrain_y = math.sin(terrain_influence) * 8
            
            final_x = x + terrain_x
            final_y = y + terrain_y
            
            # Ensure point is within district
            if district.polygon.contains(Point(final_x, final_y)):
                points.append((final_x, final_y))
        
        return self._apply_organic_smoothing(points, 'local')
    
    def _find_arterial_connection_points(self, arterial1, arterial2) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Find good connection points between two arterials."""
        import math
        
        connections = []
        min_distance = 200  # Allow closer connections
        max_distance = 2000  # Allow farther connections
        
        # Sample points along arterials
        sample_points1 = self._sample_road_points(arterial1, 3)
        sample_points2 = self._sample_road_points(arterial2, 3)
        
        for point1 in sample_points1:
            for point2 in sample_points2:
                distance = math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
                
                if min_distance <= distance <= max_distance:
                    connections.append((point1, point2))
        
        # Sort by distance and return only the best connections
        connections.sort(key=lambda x: math.sqrt((x[1][0] - x[0][0])**2 + (x[1][1] - x[0][1])**2))
        return connections[:3]  # Return up to 3 connections
    
    def _sample_road_points(self, road, num_samples: int) -> List[Tuple[float, float]]:
        """Sample evenly spaced points along a road."""
        if not road.points or len(road.points) < 2:
            return []
        
        if len(road.points) <= num_samples:
            return road.points
        
        # Sample evenly spaced points
        indices = [int(i * (len(road.points) - 1) / (num_samples - 1)) for i in range(num_samples)]
        return [road.points[i] for i in indices]
    
    def _calculate_arterial_distance(self, arterial1, arterial2) -> float:
        """Calculate average distance between two arterials."""
        import math
        
        if not arterial1.points or not arterial2.points:
            return 0
        
        # Sample points from both arterials
        sample1 = self._sample_road_points(arterial1, 3)
        sample2 = self._sample_road_points(arterial2, 3)
        
        total_distance = 0
        count = 0
        
        for p1 in sample1:
            for p2 in sample2:
                distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0
    
    def _create_organic_commercial_pattern(self, map_data: MapData, district, config: TransportationConfig, start_id: int):
        """Create an organic commercial pattern with access roads and parking areas."""
        import math
        import random
        from shapely.geometry import Point
        
        minx, miny, maxx, maxy = district.polygon.bounds
        district_center = district.polygon.centroid
        
        # Create organic access roads
        access_roads = []
        
        # Create main access road
        access_start = (minx + 20, miny + (maxy - miny) / 2)
        access_end = (maxx - 20, miny + (maxy - miny) / 2)
        
        access_points = self._create_organic_street_path(map_data, access_start, access_end, district)
        
        if len(access_points) > 2:
            access_road = Road(
                id=f"organic_commercial_access_{start_id}",
                points=access_points,
                road_type='local',
                width=config.road_styles['local']['width'] * 1.2,  # Slightly wider for commercial
                color=config.road_styles['local']['color']
            )
            map_data.add_road(access_road)
            access_roads.append(access_road)
        
        # Add organic parking access roads (reduced)
        self._add_organic_parking_roads(map_data, district, access_roads, config, start_id + 1)
    
    def _create_organic_street_path(self, map_data: MapData, start: Tuple[float, float], 
                                   end: Tuple[float, float], district) -> List[Tuple[float, float]]:
        """Create an organic street path with natural curves."""
        import math
        import random
        # Create base path with natural curves
        distance = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        num_points = max(6, int(distance / 80))  # More points for smoother curves
        points = [start]
        for i in range(1, num_points):
            t = i / num_points
            base_x = start[0] + (end[0] - start[0]) * t
            base_y = start[1] + (end[1] - start[1]) * t
            # Stronger, more varied curves
            curve_amplitude = min(60, distance * 0.22)  # More pronounced
            curve_x = math.sin(t * math.pi * 2.2) * curve_amplitude * 0.8
            curve_y = math.sin(t * math.pi * 1.7 + math.pi/4) * curve_amplitude * 0.9
            # More terrain influence
            terrain_influence = self._get_terrain_influence(map_data, (base_x, base_y))
            terrain_x = math.cos(terrain_influence) * curve_amplitude * 0.7
            terrain_y = math.sin(terrain_influence) * curve_amplitude * 0.7
            # More randomness
            random_x = (random.random() - 0.5) * curve_amplitude * 0.5
            random_y = (random.random() - 0.5) * curve_amplitude * 0.5
            final_x = base_x + curve_x + terrain_x + random_x
            final_y = base_y + curve_y + terrain_y + random_y
            from shapely.geometry import Point
            if district.polygon.contains(Point(final_x, final_y)):
                points.append((final_x, final_y))
        points.append(end)
        # Stronger smoothing for city roads
        return self._apply_organic_smoothing(points, 'city')

    def _create_organic_loop_road(self, map_data: MapData, district, loop_index: int) -> List[Tuple[float, float]]:
        import math
        import random
        from shapely.geometry import Point
        minx, miny, maxx, maxy = district.polygon.bounds
        district_center = district.polygon.centroid
        center_x = district_center.x
        center_y = district_center.y
        radius_x = (maxx - minx) * 0.32
        radius_y = (maxy - miny) * 0.32
        if loop_index > 0:
            radius_x *= 0.7
            radius_y *= 0.7
        points = []
        num_points = 22  # More points for smoother loops
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            base_x = center_x + radius_x * math.cos(angle)
            base_y = center_y + radius_y * math.sin(angle)
            variation_amplitude = min(30, radius_x * 0.18)
            variation_x = math.sin(angle * 3.5) * variation_amplitude
            variation_y = math.cos(angle * 2.8) * variation_amplitude
            terrain_influence = self._get_terrain_influence(map_data, (base_x, base_y))
            terrain_x = math.cos(terrain_influence) * variation_amplitude * 0.7
            terrain_y = math.sin(terrain_influence) * variation_amplitude * 0.7
            random_x = (random.random() - 0.5) * variation_amplitude * 0.7
            random_y = (random.random() - 0.5) * variation_amplitude * 0.7
            final_x = base_x + variation_x + terrain_x + random_x
            final_y = base_y + variation_y + terrain_y + random_y
            if district.polygon.contains(Point(final_x, final_y)):
                points.append((final_x, final_y))
        if points and len(points) > 2:
            points.append(points[0])
        return self._apply_organic_smoothing(points, 'city')

    def _apply_organic_smoothing(self, points: list, road_type: str) -> list:
        """Apply organic smoothing to road points."""
        if len(points) < 3:
            return points
        smoothed = [points[0]]
        # Stronger smoothing for city roads
        if road_type == 'city':
            weight = 0.75
        elif road_type == 'arterial':
            weight = 0.4
        else:
            weight = 0.6
        for i in range(1, len(points) - 1):
            prev = points[i - 1]
            curr = points[i]
            next_point = points[i + 1]
            smoothed_x = curr[0] * (1 - weight) + (prev[0] + next_point[0]) / 2 * weight
            smoothed_y = curr[1] * (1 - weight) + (prev[1] + next_point[1]) / 2 * weight
            smoothed.append((smoothed_x, smoothed_y))
        smoothed.append(points[-1])
        return smoothed
    
    def _find_natural_arterial_starts(self, map_data: MapData) -> list:
        """Find natural starting points for arterial roads."""
        starts = []
        # Add major district centers
        major_districts = [d for d in map_data.districts.values() if d.district_type in ['downtown', 'commercial'] and d.center]
        for district in major_districts:
            starts.append(district.center)
        # Add water access points (ports, beaches)
        for district in map_data.districts.values():
            if district.district_type in ['port', 'beach'] and district.center:
                starts.append(district.center)
        # Add map edge access points
        edge_points = [
            (map_data.width * 0.2, map_data.height * 0.1),
            (map_data.width * 0.8, map_data.height * 0.1),
            (map_data.width * 0.2, map_data.height * 0.9),
            (map_data.width * 0.8, map_data.height * 0.9),
        ]
        # Filter to land areas
        for point in edge_points:
            grid_x = int(point[0] / map_data.grid_size)
            grid_y = int(point[1] / map_data.grid_size)
            if (0 <= grid_x < map_data.grid_width and 0 <= grid_y < map_data.grid_height and map_data.land_mask[grid_y, grid_x]):
                starts.append(point)
        # Remove duplicates and limit
        unique_starts = []
        for start in starts:
            if not any(self._distance(start, existing) < 200 for existing in unique_starts):
                unique_starts.append(start)
        return unique_starts[:8]  # Limit to 8 starting points
    
    def _distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        import math
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    
    def _find_natural_arterial_end(self, map_data: MapData, start: tuple) -> tuple:
        """Find a good destination for an arterial road."""
        best_end = None
        best_score = 0
        # Look for districts that need connection
        for district in map_data.districts.values():
            if district.center:
                distance = self._distance(start, district.center)
                if 200 < distance < 1000:  # Reasonable distance
                    score = 1
                    if district.district_type in ['commercial', 'downtown', 'port', 'airport']:
                        score += 1
                    if score > best_score:
                        best_score = score
                        best_end = district.center
        # If no good district found, create a natural endpoint
        if not best_end:
            # Create endpoint on opposite side of map
            if start[0] < map_data.width / 2:
                end_x = map_data.width * 0.8
            else:
                end_x = map_data.width * 0.2
            if start[1] < map_data.height / 2:
                end_y = map_data.height * 0.8
            else:
                end_y = map_data.height * 0.2
            best_end = (end_x, end_y)
        return best_end
    
    def _create_organic_arterial_road(self, map_data: MapData, start: tuple, end: tuple, road_id: str, config) -> object:
        """Create an organic arterial road that follows terrain and natural features."""
        # Generate path using organic routing
        path_points = self._create_organic_path(map_data, start, end, 'arterial')
        if len(path_points) < 2:
            return None
        # Apply natural curvature and smoothing
        smoothed_points = self._apply_organic_smoothing(path_points, 'arterial')
        # Create the road
        road = Road(
            id=road_id,
            road_type='arterial',
            points=smoothed_points,
            width=config.road_styles['arterial']['width'],
            color=config.road_styles['arterial']['color']
        )
        return road

    def _create_organic_path(self, map_data: MapData, start: tuple, end: tuple, road_type: str) -> list:
        """Create an organic path that follows terrain and natural features."""
        # Use A* pathfinding with terrain awareness
        path = self._organic_astar_pathfinding(map_data, start, end, road_type)
        if not path:
            # Fallback to direct path with terrain avoidance
            path = self._create_direct_terrain_aware_path(map_data, start, end, road_type)
        return path

    def _organic_astar_pathfinding(self, map_data: MapData, start: tuple, end: tuple, road_type: str) -> list:
        """A* pathfinding that considers terrain, water, and natural features."""
        # Convert to grid coordinates
        start_grid = (int(start[0] / map_data.grid_size), int(start[1] / map_data.grid_size))
        end_grid = (int(end[0] / map_data.grid_size), int(end[1] / map_data.grid_size))
        # A* implementation with terrain cost
        open_set = {start_grid}
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self._heuristic(start_grid, end_grid)}
        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            if current == end_grid:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_grid)
                path.reverse()
                # Convert back to map coordinates and add intermediate points
                return self._convert_grid_path_to_map_path(path, map_data)
            open_set.remove(current)
            # Check neighbors
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if (0 <= neighbor[0] < map_data.grid_width and 0 <= neighbor[1] < map_data.grid_height):
                    # Calculate movement cost based on terrain
                    terrain_cost = self._calculate_terrain_cost(map_data, neighbor, road_type)
                    if terrain_cost < float('inf'):  # Valid move
                        tentative_g_score = g_score[current] + terrain_cost
                        if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                            came_from[neighbor] = current
                            g_score[neighbor] = tentative_g_score
                            f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, end_grid)
                            open_set.add(neighbor)
        return []  # No path found

    def _heuristic(self, pos1: tuple, pos2: tuple) -> float:
        """Heuristic function for A* (Manhattan distance)."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _calculate_terrain_cost(self, map_data: MapData, grid_pos: tuple, road_type: str) -> float:
        """Calculate movement cost based on terrain type and elevation."""
        x, y = grid_pos
        if not map_data.land_mask[y, x]:
            return float('inf')  # Can't build on water
        elevation = map_data.heightmap[y, x]
        base_cost = 1.0
        # Elevation penalties
        if elevation > 0.7:  # Mountains
            base_cost *= 5.0
        elif elevation > 0.6:  # Hills
            base_cost *= 2.0
        elif elevation < 0.4:  # Low areas (flood risk)
            base_cost *= 1.5
        # Road type adjustments
        if road_type == 'highway':
            base_cost *= 0.8  # Highways can handle more difficult terrain
        elif road_type == 'local':
            base_cost *= 1.2  # Local roads prefer easier terrain
        return base_cost

    def _convert_grid_path_to_map_path(self, grid_path: list, map_data: MapData) -> list:
        """Convert grid path to map coordinates with intermediate points."""
        map_path = []
        for i, grid_pos in enumerate(grid_path):
            map_x = grid_pos[0] * map_data.grid_size + map_data.grid_size / 2
            map_y = grid_pos[1] * map_data.grid_size + map_data.grid_size / 2
            map_path.append((map_x, map_y))
            # Add intermediate points for smoother curves
            if i < len(grid_path) - 1:
                next_grid = grid_path[i + 1]
                # Add a point between current and next for smoother curves
                mid_x = (grid_pos[0] + next_grid[0]) / 2 * map_data.grid_size + map_data.grid_size / 2
                mid_y = (grid_pos[1] + next_grid[1]) / 2 * map_data.grid_size + map_data.grid_size / 2
                map_path.append((mid_x, mid_y))
        return map_path

    def _create_secondary_arterials(self, map_data: MapData, config: TransportationConfig):
        """Create secondary arterial roads that branch off main arterials."""
        if not self.arterial_spine:
            return
        secondary_count = 0
        max_secondary = min(10, len(self.arterial_spine) * 2)
        for arterial in self.arterial_spine:
            if secondary_count >= max_secondary:
                break
            branch_points = self._find_branching_points(arterial, map_data)
            for branch_point in branch_points:
                if secondary_count >= max_secondary:
                    break
                destination = self._find_secondary_destination(map_data, branch_point)
                if destination:
                    secondary_road = self._create_organic_arterial_road(
                        map_data, branch_point, destination, f"secondary_arterial_{secondary_count}", config
                    )
                    if secondary_road:
                        map_data.add_road(secondary_road)
                        secondary_count += 1

    def _find_branching_points(self, arterial: Road, map_data: MapData) -> list:
        """Find good points along an arterial for branching secondary roads."""
        if not arterial.points or len(arterial.points) < 3:
            return []
        branch_points = []
        # Sample points along the arterial with more spacing
        for i in range(2, len(arterial.points) - 2, 5):
            point = arterial.points[i]
            # Check if this is a good branching location
            if self._is_good_branching_location(map_data, point):
                branch_points.append(point)
        return branch_points[:2]  # Limit to 2 branches per arterial

    def _is_good_branching_location(self, map_data: MapData, point: tuple) -> bool:
        """Check if a point is suitable for branching a secondary road."""
        grid_x = int(point[0] / map_data.grid_size)
        grid_y = int(point[1] / map_data.grid_size)
        if not (0 <= grid_x < map_data.grid_width and 0 <= grid_y < map_data.grid_height):
            return False
        # Check if area is suitable (not too close to water, good elevation)
        elevation = map_data.heightmap[grid_y, grid_x]
        if elevation < 0.4 or elevation > 0.7:
            return False
        # Check if not too close to existing roads
        for road in map_data.roads.values():
            if road.points:
                min_dist = min(self._distance(point, road_point) for road_point in road.points)
                if min_dist < 100:
                    return False
        return True

    def _create_direct_terrain_aware_path(self, map_data: MapData, start: tuple, end: tuple, road_type: str) -> list:
        """Create a direct path that avoids major obstacles."""
        path = [start]
        num_points = max(3, int(self._distance(start, end) / 100))
        for i in range(1, num_points):
            t = i / num_points
            point = self._lerp(start, end, t)
            adjusted_point = self._avoid_obstacles(map_data, point, road_type)
            path.append(adjusted_point)
        path.append(end)
        return path

    def _lerp(self, start: tuple, end: tuple, t: float) -> tuple:
        """Linear interpolation between two points."""
        return (start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t)

    def _avoid_obstacles(self, map_data: MapData, point: tuple, road_type: str) -> tuple:
        """Adjust point to avoid water bodies and other obstacles."""
        x, y = point
        avoidance_radius = 80 if road_type in ['highway', 'arterial'] else 50
        # Check for water bodies
        for water_body in map_data.water_bodies.values():
            if hasattr(water_body, 'geometry') and water_body.geometry:
                distance = water_body.geometry.distance(Point(x, y))
                if distance < avoidance_radius:
                    # Calculate avoidance vector
                    if water_body.geometry.contains(Point(x, y)):
                        # Point is inside water - find nearest edge and move out
                        boundary = water_body.geometry.boundary
                        nearest_point = boundary.interpolate(boundary.project(Point(x, y)))
                        # Move away from water
                        avoid_x = x + (x - nearest_point.x) * 2
                        avoid_y = y + (y - nearest_point.y) * 2
                    else:
                        # Point is near water - adjust to maintain distance
                        centroid = water_body.geometry.centroid
                        dx = x - centroid.x
                        dy = y - centroid.y
                        length = math.sqrt(dx**2 + dy**2)
                        if length > 0:
                            # Move away from water center
                            avoid_x = x + (dx / length) * (avoidance_radius - distance)
                            avoid_y = y + (dy / length) * (avoidance_radius - distance)
                        else:
                            avoid_x, avoid_y = x, y
                    # Ensure within map bounds
                    x = max(50, min(map_data.width - 50, avoid_x))
                    y = max(50, min(map_data.height - 50, avoid_y))
        return (x, y)

    def connect_features_to_roads(self, map_data: MapData, config: TransportationConfig):
        """
        For each building, park, and POI in each city block, ensure it is connected to the nearest road.
        If not within 30 units of a road, generate a connector road.
        Group nearby features to share connectors, avoid obstacles, and connect to building edges.
        """
        from shapely.geometry import Point, LineString
        import math
        connector_id = 0
        all_roads = list(map_data.roads.values())
        # Helper to check if a feature is close to any road
        def is_connected(feature_center):
            for road in all_roads:
                for pt in road.points:
                    if ((pt[0] - feature_center[0]) ** 2 + (pt[1] - feature_center[1]) ** 2) ** 0.5 < 30:
                        return True
            return False
        # Helper to find nearest road point
        def nearest_road_point(feature_center):
            min_dist = float('inf')
            nearest = None
            for road in all_roads:
                for pt in road.points:
                    dist = ((pt[0] - feature_center[0]) ** 2 + (pt[1] - feature_center[1]) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        nearest = pt
            return nearest
        # Helper to find a building's edge closest to a point
        def building_edge_point(building, target):
            # Rectangle edges
            corners = [
                (building.x, building.y),
                (building.x + building.width, building.y),
                (building.x + building.width, building.y + building.length),
                (building.x, building.y + building.length)
            ]
            min_dist = float('inf')
            best = None
            for i in range(4):
                p1 = corners[i]
                p2 = corners[(i+1)%4]
                line = LineString([p1, p2])
                proj = line.interpolate(line.project(Point(target)))
                dist = Point(target).distance(proj)
                if dist < min_dist:
                    min_dist = dist
                    best = (proj.x, proj.y)
            return best
        # Group features by proximity (within 40 units)
        def group_features(centers):
            groups = []
            used = set()
            for i, c1 in enumerate(centers):
                if i in used:
                    continue
                group = [i]
                for j, c2 in enumerate(centers):
                    if i != j and j not in used:
                        if ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2) ** 0.5 < 40:
                            group.append(j)
                            used.add(j)
                used.add(i)
                groups.append(group)
            return groups
        # Collect all feature centers
        building_centers = []
        building_refs = []
        for building in map_data.buildings.values():
            # Use edge closest to nearest road, not center
            road_pt = nearest_road_point((building.x + building.width/2, building.y + building.length/2))
            if road_pt:
                edge_pt = building_edge_point(building, road_pt)
            else:
                edge_pt = (building.x + building.width/2, building.y + building.length/2)
            building_centers.append(edge_pt)
            building_refs.append(building)
        park_centers = []
        park_refs = []
        for park in map_data.parks.values():
            if hasattr(park, 'center'):
                center = park.center
            elif hasattr(park, 'polygon') and park.polygon:
                center = (park.polygon.centroid.x, park.polygon.centroid.y)
            else:
                continue
            park_centers.append(center)
            park_refs.append(park)
        poi_centers = []
        poi_refs = []
        for poi in map_data.pois.values():
            center = (poi.x + poi.width/2, poi.y + poi.height/2)
            poi_centers.append(center)
            poi_refs.append(poi)
        # Group and connect buildings
        for group in group_features(building_centers):
            group_pts = [building_centers[i] for i in group]
            # Use group centroid as connector start
            group_center = (sum(p[0] for p in group_pts)/len(group_pts), sum(p[1] for p in group_pts)/len(group_pts))
            if not is_connected(group_center):
                road_pt = nearest_road_point(group_center)
                if road_pt:
                    # Path from group center to road
                    connector_points = self._create_organic_path(map_data, group_center, road_pt, 'local')
                    if connector_points and len(connector_points) > 1:
                        connector_road = Road(
                            id=f"connector_building_{connector_id}",
                            points=connector_points,
                            road_type='connector_building',
                            width=config.road_styles['local']['width'] * 0.7,
                            color='#bbbbbb'
                        )
                        map_data.add_road(connector_road)
                        connector_id += 1
        # Group and connect parks
        for group in group_features(park_centers):
            group_pts = [park_centers[i] for i in group]
            group_center = (sum(p[0] for p in group_pts)/len(group_pts), sum(p[1] for p in group_pts)/len(group_pts))
            if not is_connected(group_center):
                road_pt = nearest_road_point(group_center)
                if road_pt:
                    connector_points = self._create_organic_path(map_data, group_center, road_pt, 'local')
                    if connector_points and len(connector_points) > 1:
                        connector_road = Road(
                            id=f"connector_park_{connector_id}",
                            points=connector_points,
                            road_type='connector_park',
                            width=config.road_styles['local']['width'] * 0.7,
                            color='#aaddaa'
                        )
                        map_data.add_road(connector_road)
                        connector_id += 1
        # Group and connect POIs
        for group in group_features(poi_centers):
            group_pts = [poi_centers[i] for i in group]
            group_center = (sum(p[0] for p in group_pts)/len(group_pts), sum(p[1] for p in group_pts)/len(group_pts))
            if not is_connected(group_center):
                road_pt = nearest_road_point(group_center)
                if road_pt:
                    connector_points = self._create_organic_path(map_data, group_center, road_pt, 'local')
                    if connector_points and len(connector_points) > 1:
                        connector_road = Road(
                            id=f"connector_poi_{connector_id}",
                            points=connector_points,
                            road_type='connector_poi',
                            width=config.road_styles['local']['width'] * 0.7,
                            color='#aaaaff'
                        )
                        map_data.add_road(connector_road)
                        connector_id += 1


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