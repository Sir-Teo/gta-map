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
        """Generate an organized arterial network with proper connectivity."""
        
        print("    → Building organized arterial backbone...")
        
        # Step 1: Create main arterial spine through the map
        self._create_arterial_spine(map_data, config)
        
        # Step 2: Create perpendicular arterials for grid structure
        self._create_arterial_cross_streets(map_data, config)
        
        # Step 3: Connect all districts to the arterial network
        self._connect_districts_to_arterials(map_data, config)
        
        print(f"    → Arterial backbone complete with proper connectivity")
    
    def _create_arterial_spine(self, map_data: MapData, config: TransportationConfig):
        """Create the main arterial spine running naturally through the map."""
        import math
        
        # Find the optimal route for main arterial - prefer flowing across diverse terrain
        best_start = self._find_natural_arterial_start(map_data)
        best_end = self._find_natural_arterial_end(map_data, best_start)
        
        # Create main arterial with natural flow that follows landscape
        main_arterial_path = self._create_flowing_spine_path(map_data, best_start, best_end)
        
        if main_arterial_path and len(main_arterial_path) > 2:
            main_arterial = Road(
                id="main_arterial_spine",
                points=main_arterial_path,
                road_type='arterial',
                width=config.road_styles['arterial']['width'] * 1.3,  # Make arterials wider
                color=config.road_styles['arterial']['color']
            )
            map_data.add_road(main_arterial)
            
        # Store spine for reference
        self.arterial_spine = main_arterial_path
    
    def _find_natural_arterial_start(self, map_data: MapData) -> Tuple[float, float]:
        """Find a natural starting point that considers terrain and districts."""
        
        # Look for districts but also consider terrain
        major_districts = [d for d in map_data.districts.values() 
                          if d.district_type in ['downtown', 'commercial'] and d.polygon]
        
        if major_districts:
            # Start from a major district but offset towards map edge
            district = major_districts[0]
            centroid = district.polygon.centroid
            
            # Move start towards nearest map edge for more natural flow
            edge_offset = min(map_data.width, map_data.height) * 0.15
            
            if centroid.x < map_data.width / 2:
                start_x = max(edge_offset, centroid.x - edge_offset)
            else:
                start_x = min(map_data.width - edge_offset, centroid.x + edge_offset)
                
            start_y = centroid.y
            
            return (start_x, start_y)
        else:
            # Use natural terrain-based starting point
            return (map_data.width * 0.15, map_data.height * 0.5)
    
    def _find_natural_arterial_end(self, map_data: MapData, start: Tuple[float, float]) -> Tuple[float, float]:
        """Find an ending point that creates natural flow across the map."""
        
        # Look for districts that create good cross-map flow
        districts = [d for d in map_data.districts.values() 
                    if d.district_type in ['downtown', 'commercial', 'residential'] and d.polygon]
        
        if districts:
            # Find district that creates good diagonal or flowing path
            best_end = None
            best_flow_score = 0
            
            for district in districts:
                centroid = district.polygon.centroid
                
                # Calculate flow score - prefer diagonal routes across map
                dx = abs(centroid.x - start[0])
                dy = abs(centroid.y - start[1])
                distance = math.sqrt(dx**2 + dy**2)
                
                # Prefer routes that go across the map diagonally
                diagonal_score = min(dx, dy) / max(dx, dy) if max(dx, dy) > 0 else 0
                distance_score = distance / math.sqrt(map_data.width**2 + map_data.height**2)
                
                flow_score = diagonal_score * 0.6 + distance_score * 0.4
                
                if flow_score > best_flow_score:
                    best_flow_score = flow_score
                    best_end = (centroid.x, centroid.y)
            
            if best_end:
                return best_end
        
        # Default: create diagonal flow across map
        if start[0] < map_data.width / 2:
            end_x = map_data.width * 0.85
        else:
            end_x = map_data.width * 0.15
            
        if start[1] < map_data.height / 2:
            end_y = map_data.height * 0.75
        else:
            end_y = map_data.height * 0.25
            
        return (end_x, end_y)
    
    def _create_flowing_spine_path(self, map_data: MapData, start: Tuple[float, float], 
                                  end: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Create a naturally flowing spine path that follows terrain features."""
        import math
        import random
        
        total_distance = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
        
        # More control points for major arterial
        num_control_points = max(6, int(total_distance / 180))
        
        control_points = [start]
        
        for i in range(1, num_control_points):
            t = i / num_control_points
            
            # Base interpolation
            base_x = start[0] * (1 - t) + end[0] * t
            base_y = start[1] * (1 - t) + end[1] * t
            
            # Major arterial should have gentle, sweeping curves
            curve_amplitude = min(120, total_distance * 0.1)
            
            # Create flowing curves that follow natural patterns
            primary_curve = math.sin(t * math.pi * 1.2) * curve_amplitude
            secondary_curve = math.sin(t * math.pi * 2.4 + math.pi/4) * curve_amplitude * 0.3
            
            # Add terrain awareness
            terrain_influence = self._get_enhanced_terrain_influence(map_data, (base_x, base_y))
            terrain_offset_x = math.cos(terrain_influence) * curve_amplitude * 0.4
            terrain_offset_y = math.sin(terrain_influence) * curve_amplitude * 0.4
            
            # Combine for natural flow
            flow_x = base_x + primary_curve * 0.7 + terrain_offset_x
            flow_y = base_y + secondary_curve + terrain_offset_y
            
            # Ensure within bounds with margin
            flow_x = max(100, min(map_data.width - 100, flow_x))
            flow_y = max(100, min(map_data.height - 100, flow_y))
            
            control_points.append((flow_x, flow_y))
        
        control_points.append(end)
        
        # Apply enhanced spline smoothing for arterial
        return self._apply_enhanced_spline_smoothing(control_points)
    
    def _get_enhanced_terrain_influence(self, map_data: MapData, point: Tuple[float, float]) -> float:
        """Enhanced terrain influence for major roads."""
        import math
        
        if map_data.heightmap is None:
            return 0
        
        x, y = point
        grid_x = int(x / map_data.grid_size)
        grid_y = int(y / map_data.grid_size)
        
        if not (0 <= grid_x < map_data.grid_width-2 and 0 <= grid_y < map_data.grid_height-2):
            return 0
        
        # Sample a larger area for major roads
        height_samples = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if 0 <= grid_x + dx < map_data.grid_width and 0 <= grid_y + dy < map_data.grid_height:
                    height_samples.append(map_data.heightmap[grid_y + dy, grid_x + dx])
        
        if len(height_samples) < 4:
            return 0
        
        # Find the direction of gentlest slope
        avg_height = sum(height_samples) / len(height_samples)
        gradient_x = (height_samples[2] - height_samples[0]) / 2 if len(height_samples) >= 3 else 0
        gradient_y = (height_samples[6] - height_samples[0]) / 2 if len(height_samples) >= 7 else 0
        
        # Major roads prefer following contours (perpendicular to slope)
        if abs(gradient_x) > 0.1 or abs(gradient_y) > 0.1:
            slope_angle = math.atan2(gradient_y, gradient_x)
            return slope_angle + math.pi/2  # Perpendicular to slope
        
        return 0
    
    def _apply_enhanced_spline_smoothing(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Enhanced spline smoothing for major arterials."""
        import numpy as np
        from scipy import interpolate
        
        if len(points) < 3:
            return points
        
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        # Use parameter based on distance for better curves
        distances = [0]
        for i in range(1, len(points)):
            dist = math.sqrt((x_coords[i] - x_coords[i-1])**2 + (y_coords[i] - y_coords[i-1])**2)
            distances.append(distances[-1] + dist)
        
        # Normalize distances
        total_dist = distances[-1]
        if total_dist > 0:
            t = np.array([d / total_dist for d in distances])
        else:
            t = np.linspace(0, 1, len(points))
        
        try:
            # Create smoother splines with more points
            cs_x = interpolate.CubicSpline(t, x_coords, bc_type='natural')
            cs_y = interpolate.CubicSpline(t, y_coords, bc_type='natural')
            
            # Generate many more points for very smooth curves
            t_new = np.linspace(0, 1, len(points) * 3)
            
            smooth_x = cs_x(t_new)
            smooth_y = cs_y(t_new)
            
            smoothed_points = [(float(x), float(y)) for x, y in zip(smooth_x, smooth_y)]
            return smoothed_points
            
        except Exception:
            return self._simple_smooth_path(points)
    
    def _create_arterial_cross_streets(self, map_data: MapData, config: TransportationConfig):
        """Create organized perpendicular arterials with natural intersections."""
        import math
        
        if not hasattr(self, 'arterial_spine') or not self.arterial_spine:
            return
        
        spine_length = len(self.arterial_spine)
        
        # Create only 1-2 major cross arterials for cleaner layout
        if spine_length > 10:
            intersection_indices = [spine_length // 3, 2 * spine_length // 3]
        else:
            intersection_indices = [spine_length // 2]  # Just one intersection for shorter spines
        
        for i, idx in enumerate(intersection_indices):
            if idx >= len(self.arterial_spine):
                continue
                
            intersection_point = self.arterial_spine[idx]
            
            # Calculate smooth perpendicular direction
            spine_direction = self._get_smooth_road_direction(self.arterial_spine, idx)
            perp_angle = math.atan2(spine_direction[1], spine_direction[0]) + math.pi/2
            
            # Create cross arterial with natural endpoints
            cross_length = min(500, map_data.width // 5)  # Shorter for better proportions
            
            # Find natural starting and ending points
            start_point = self._find_natural_cross_start(
                map_data, intersection_point, perp_angle + math.pi, cross_length
            )
            end_point = self._find_natural_cross_end(
                map_data, intersection_point, perp_angle, cross_length
            )
            
            # Create flowing cross arterial
            cross_path = self._create_flowing_cross_path(
                map_data, start_point, intersection_point, end_point
            )
            
            if cross_path and len(cross_path) > 2:
                cross_arterial = Road(
                    id=f"flowing_cross_arterial_{i}",
                    points=cross_path,
                    road_type='arterial',
                    width=config.road_styles['arterial']['width'] * 1.1,  # Slightly wider
                    color=config.road_styles['arterial']['color']
                )
                map_data.add_road(cross_arterial)
    
    def _get_smooth_road_direction(self, road_points: List[Tuple[float, float]], index: int) -> Tuple[float, float]:
        """Get smoothed direction at a road point by averaging nearby segments."""
        import math
        
        if len(road_points) < 3:
            return (1.0, 0.0)
        
        # Sample multiple segments around the point for smoother direction
        sample_range = min(3, len(road_points) // 4)
        
        directions = []
        
        for offset in range(-sample_range, sample_range + 1):
            idx = max(0, min(len(road_points) - 2, index + offset))
            
            if idx < len(road_points) - 1:
                p1 = road_points[idx]
                p2 = road_points[idx + 1]
                
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                length = math.sqrt(dx**2 + dy**2)
                
                if length > 0:
                    directions.append((dx / length, dy / length))
        
        if not directions:
            return (1.0, 0.0)
        
        # Average all directions
        avg_dx = sum(d[0] for d in directions) / len(directions)
        avg_dy = sum(d[1] for d in directions) / len(directions)
        
        # Normalize
        length = math.sqrt(avg_dx**2 + avg_dy**2)
        if length > 0:
            return (avg_dx / length, avg_dy / length)
        
        return (1.0, 0.0)
    
    def _find_natural_cross_start(self, map_data: MapData, intersection_point: Tuple[float, float], 
                                 angle: float, max_length: float) -> Tuple[float, float]:
        """Find natural starting point for cross arterial."""
        import math
        
        # Start with basic geometric calculation
        base_x = intersection_point[0] + math.cos(angle) * max_length
        base_y = intersection_point[1] + math.sin(angle) * max_length
        
        # Ensure within bounds
        base_x = max(80, min(map_data.width - 80, base_x))
        base_y = max(80, min(map_data.height - 80, base_y))
        
        # Look for nearby districts to connect to
        for district in map_data.districts.values():
            if district.polygon:
                centroid = district.polygon.centroid
                distance = math.sqrt(
                    (centroid.x - base_x)**2 + (centroid.y - base_y)**2
                )
                
                # If there's a district nearby, use it as endpoint
                if distance < 200:
                    return (centroid.x, centroid.y)
        
        return (base_x, base_y)
    
    def _find_natural_cross_end(self, map_data: MapData, intersection_point: Tuple[float, float], 
                               angle: float, max_length: float) -> Tuple[float, float]:
        """Find natural ending point for cross arterial."""
        import math
        
        # Start with basic geometric calculation  
        base_x = intersection_point[0] + math.cos(angle) * max_length
        base_y = intersection_point[1] + math.sin(angle) * max_length
        
        # Ensure within bounds
        base_x = max(80, min(map_data.width - 80, base_x))
        base_y = max(80, min(map_data.height - 80, base_y))
        
        # Look for nearby districts to connect to
        for district in map_data.districts.values():
            if district.polygon:
                centroid = district.polygon.centroid
                distance = math.sqrt(
                    (centroid.x - base_x)**2 + (centroid.y - base_y)**2
                )
                
                # If there's a district nearby, use it as endpoint
                if distance < 200:
                    return (centroid.x, centroid.y)
        
        return (base_x, base_y)
    
    def _create_flowing_cross_path(self, map_data: MapData, start: Tuple[float, float], 
                                  intersection: Tuple[float, float], end: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Create a flowing cross arterial path through intersection."""
        
        # Create two segments that flow smoothly through intersection
        segment1 = self._create_organized_path(map_data, start, intersection, 'arterial')
        segment2 = self._create_organized_path(map_data, intersection, end, 'arterial')
        
        if not segment1 or not segment2:
            return []
        
        # Combine segments, removing duplicate intersection point
        combined_path = segment1[:-1] + segment2
        
        # Apply additional smoothing at intersection area
        return self._smooth_intersection_area(combined_path, len(segment1) - 1)
    
    def _smooth_intersection_area(self, path: List[Tuple[float, float]], intersection_idx: int) -> List[Tuple[float, float]]:
        """Apply additional smoothing around intersection point."""
        
        if len(path) < 5 or intersection_idx < 2 or intersection_idx >= len(path) - 2:
            return path
        
        # Apply stronger smoothing in intersection area
        smoothed_path = path.copy()
        
        smooth_range = 2  # Smooth 2 points on each side of intersection
        
        for i in range(max(1, intersection_idx - smooth_range), 
                      min(len(path) - 1, intersection_idx + smooth_range + 1)):
            
            if i == 0 or i == len(path) - 1:
                continue
            
            # Apply stronger averaging for intersection smoothness
            prev_point = path[i - 1]
            curr_point = path[i]
            next_point = path[i + 1]
            
            # Weighted average with higher center weight for stability
            smooth_x = prev_point[0] * 0.2 + curr_point[0] * 0.6 + next_point[0] * 0.2
            smooth_y = prev_point[1] * 0.2 + curr_point[1] * 0.6 + next_point[1] * 0.2
            
            smoothed_path[i] = (smooth_x, smooth_y)
        
        return smoothed_path
    
    def _connect_districts_to_arterials(self, map_data: MapData, config: TransportationConfig):
        """Ensure all districts are connected to the arterial network."""
        import math
        
        # Get all existing arterials
        arterials = [road for road in map_data.roads.values() if road.road_type == 'arterial']
        
        if not arterials:
            return
        
        connector_id = 0
        max_arterial_connectors = 3  # Limit arterial connectors
        
        # Only connect major districts
        major_districts = [d for d in map_data.districts.values() 
                          if d.district_type in ['downtown', 'commercial'] and d.polygon]
        
        for district in major_districts[:max_arterial_connectors]:  # Limit to first 3 major districts
            district_center = district.polygon.centroid
            
            # Find nearest arterial
            nearest_arterial, nearest_point = self._find_nearest_road_point(
                (district_center.x, district_center.y), arterials
            )
            
            if nearest_arterial and nearest_point:
                # Create connector only if district is far from arterial
                distance = math.sqrt(
                    (district_center.x - nearest_point[0])**2 + 
                    (district_center.y - nearest_point[1])**2
                )
                
                if distance > 200:  # Only connect if reasonably far
                    connector_path = self._create_organized_path(
                        map_data, 
                        (district_center.x, district_center.y), 
                        nearest_point, 
                        'arterial'
                    )
                    
                    if connector_path and len(connector_path) > 2:
                        connector = Road(
                            id=f"major_district_connector_{connector_id}",
                            points=connector_path,
                            road_type='arterial',
                            width=config.road_styles['arterial']['width'],
                            color=config.road_styles['arterial']['color']
                        )
                        map_data.add_road(connector)
                        connector_id += 1
    
    def _find_optimal_arterial_start(self, map_data: MapData) -> Tuple[float, float]:
        """Find the best starting point for the main arterial."""
        
        # Look for major districts or use map edge
        major_districts = [d for d in map_data.districts.values() 
                          if d.district_type in ['downtown', 'commercial'] and d.polygon]
        
        if major_districts:
            # Start from major district
            district = major_districts[0]
            centroid = district.polygon.centroid
            return (centroid.x, centroid.y)
        else:
            # Start from map edge
            return (map_data.width * 0.1, map_data.height * 0.5)
    
    def _find_optimal_arterial_end(self, map_data: MapData, start: Tuple[float, float]) -> Tuple[float, float]:
        """Find the best ending point for the main arterial."""
        
        # Look for districts far from start
        major_districts = [d for d in map_data.districts.values() 
                          if d.district_type in ['downtown', 'commercial', 'residential'] and d.polygon]
        
        if major_districts:
            # Find farthest major district
            max_distance = 0
            best_end = None
            
            for district in major_districts:
                centroid = district.polygon.centroid
                distance = ((centroid.x - start[0])**2 + (centroid.y - start[1])**2)**0.5
                
                if distance > max_distance:
                    max_distance = distance
                    best_end = (centroid.x, centroid.y)
            
            if best_end:
                return best_end
        
        # Default: opposite side of map
        return (map_data.width * 0.9, map_data.height * 0.5)
    
    def _create_organized_path(self, map_data: MapData, start: Tuple[float, float], 
                             end: Tuple[float, float], road_type: str) -> List[Tuple[float, float]]:
        """Create a naturally flowing path with smooth curves and terrain awareness."""
        import math
        import random
        import numpy as np
        
        # Calculate base path parameters
        total_distance = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
        
        # More points for smoother curves, especially for longer roads
        if road_type == 'highway':
            num_points = max(8, int(total_distance / 150))
        elif road_type == 'arterial':
            num_points = max(6, int(total_distance / 200))
        else:
            num_points = max(4, int(total_distance / 250))
        
        # Check for obstacles and create natural routing
        raw_points = self._create_obstacle_aware_path(map_data, start, end, num_points, road_type)
        
        # Apply natural curvature
        curved_points = self._apply_natural_curvature(raw_points, road_type, total_distance)
        
        # Apply spline smoothing for natural curves
        smoothed_path = self._apply_spline_smoothing(curved_points)
        
        return smoothed_path
    
    def _create_obstacle_aware_path(self, map_data: MapData, start: Tuple[float, float], 
                                   end: Tuple[float, float], num_points: int, road_type: str) -> List[Tuple[float, float]]:
        """Create a path that naturally avoids obstacles like water and parks."""
        
        points = [start]
        
        for i in range(1, num_points):
            t = i / num_points
            
            # Basic interpolation
            base_x = start[0] * (1 - t) + end[0] * t
            base_y = start[1] * (1 - t) + end[1] * t
            
            # Check for nearby obstacles and adjust path
            adjusted_point = self._avoid_obstacles(map_data, (base_x, base_y), road_type)
            points.append(adjusted_point)
        
        points.append(end)
        return points
    
    def _avoid_obstacles(self, map_data: MapData, point: Tuple[float, float], road_type: str) -> Tuple[float, float]:
        """Adjust point to avoid water bodies and respect natural features."""
        x, y = point
        avoidance_radius = 100 if road_type in ['highway', 'arterial'] else 60
        
        # Check for water bodies
        for water_body in map_data.water_bodies.values():
            if hasattr(water_body, 'polygon') and water_body.polygon:
                distance = water_body.polygon.distance(Point(x, y))
                
                if distance < avoidance_radius:
                    # Calculate avoidance vector
                    if water_body.polygon.contains(Point(x, y)):
                        # Point is inside water - find nearest edge and move out
                        boundary = water_body.polygon.boundary
                        nearest_point = boundary.interpolate(boundary.project(Point(x, y)))
                        
                        # Move away from water
                        avoid_x = x + (x - nearest_point.x) * 2
                        avoid_y = y + (y - nearest_point.y) * 2
                    else:
                        # Point is near water - adjust to maintain distance
                        centroid = water_body.polygon.centroid
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
        
        # Check for parks and create scenic routing
        for park in map_data.parks.values():
            if hasattr(park, 'polygon') and park.polygon:
                distance = park.polygon.distance(Point(x, y))
                
                # Create scenic routes around parks (closer approach for beauty)
                scenic_distance = 40 if road_type in ['arterial', 'collector'] else 60
                
                if distance < scenic_distance:
                    # Route around park perimeter for scenic effect
                    centroid = park.polygon.centroid
                    dx = x - centroid.x
                    dy = y - centroid.y
                    length = math.sqrt(dx**2 + dy**2)
                    
                    if length > 0:
                        # Maintain scenic distance
                        scenic_x = centroid.x + (dx / length) * scenic_distance
                        scenic_y = centroid.y + (dy / length) * scenic_distance
                        
                        x = max(50, min(map_data.width - 50, scenic_x))
                        y = max(50, min(map_data.height - 50, scenic_y))
        
        return (x, y)
    
    def _apply_natural_curvature(self, points: List[Tuple[float, float]], road_type: str, total_distance: float) -> List[Tuple[float, float]]:
        """Apply natural curvature based on road type."""
        import math
        import random
        
        curved_points = []
        
        for i, (x, y) in enumerate(points):
            if i == 0 or i == len(points) - 1:
                # Keep start and end points fixed
                curved_points.append((x, y))
                continue
            
            t = i / (len(points) - 1)
            
            # Add natural curvature based on road type
            if road_type == 'highway':
                # Highways: gentle, sweeping curves
                curve_amplitude = min(80, total_distance * 0.08)
                curve_frequency = 0.7
            elif road_type == 'arterial':
                # Arterials: moderate curves
                curve_amplitude = min(60, total_distance * 0.12)
                curve_frequency = 1.0
            else:
                # Collectors/Local: more curves
                curve_amplitude = min(40, total_distance * 0.15)
                curve_frequency = 1.5
            
            # Create natural S-curves
            curve_offset_x = math.sin(t * math.pi * curve_frequency) * curve_amplitude * 0.6
            curve_offset_y = math.sin(t * math.pi * curve_frequency + math.pi/3) * curve_amplitude
            
            # Add some randomness for natural variation
            random_offset_x = (random.random() - 0.5) * curve_amplitude * 0.3
            random_offset_y = (random.random() - 0.5) * curve_amplitude * 0.3
            
            # Combine all influences
            final_x = x + curve_offset_x + random_offset_x
            final_y = y + curve_offset_y + random_offset_y
            
            curved_points.append((final_x, final_y))
        
        return curved_points
    
    def _create_systematic_collectors(self, map_data: MapData, config: TransportationConfig):
        """Create collectors that systematically connect to arterials with better distribution."""
        import math
        
        # Get all arterials
        arterials = [road for road in map_data.roads.values() if road.road_type == 'arterial']
        
        if len(arterials) < 2:
            return
        
        collector_id = 0
        max_collectors = 6  # Reduced further to prevent clustering
        
        # Create collectors with better spacing to avoid spider web effect
        connected_arterials = set()
        
        for i, arterial1 in enumerate(arterials):
            if collector_id >= max_collectors:
                break
                
            if i in connected_arterials:
                continue
                
            # Find best arterial to connect to (not adjacent ones)
            best_arterial = None
            best_distance = 0
            
            for j, arterial2 in enumerate(arterials[i+2:], i+2):  # Skip adjacent arterials
                if j in connected_arterials:
                    continue
                    
                # Check distance between arterials
                distance = self._calculate_arterial_distance(arterial1, arterial2)
                
                if 300 < distance < 700 and distance > best_distance:  # Good spacing
                    best_distance = distance
                    best_arterial = arterial2
            
            if best_arterial:
                # Create single collector between these arterials
                connections = self._find_arterial_connection_points(arterial1, best_arterial)
                
                for start_point, end_point in connections[:1]:  # Only one connection
                    collector_path = self._create_organized_path(
                        map_data, start_point, end_point, 'collector'
                    )
                    
                    if collector_path and len(collector_path) > 2:
                        collector = Road(
                            id=f"systematic_collector_{collector_id}",
                            points=collector_path,
                            road_type='collector',
                            width=config.road_styles['collector']['width'],
                            color=config.road_styles['collector']['color']
                        )
                        map_data.add_road(collector)
                        connected_arterials.add(i)
                        connected_arterials.add(arterials.index(best_arterial))
                        collector_id += 1
                        break
    
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
    
    def _generate_collector_roads(self, map_data: MapData, config: TransportationConfig):
        """Generate organized collector roads that connect properly to arterials."""
        
        print("    → Building organized collector network...")
        
        # Step 1: Create systematic collectors between arterials
        self._create_systematic_collectors(map_data, config)
        
        # Step 2: Connect districts to collectors
        self._connect_districts_to_collectors(map_data, config)
        
        print("    → Collector network complete with proper hierarchy")
    
    def _find_arterial_connection_points(self, arterial1, arterial2) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Find good connection points between two arterials."""
        import math
        
        connections = []
        min_distance = 250  # Increased minimum distance
        max_distance = 500  # Decreased maximum distance
        
        # Sample fewer points along arterials
        sample_points1 = self._sample_road_points(arterial1, 3)  # Reduced from 5 to 3
        sample_points2 = self._sample_road_points(arterial2, 3)
        
        for point1 in sample_points1:
            for point2 in sample_points2:
                distance = math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
                
                if min_distance <= distance <= max_distance:
                    connections.append((point1, point2))
        
        # Sort by distance and return only the best connection
        connections.sort(key=lambda x: math.sqrt((x[1][0] - x[0][0])**2 + (x[1][1] - x[0][1])**2))
        return connections[:1]  # Return only 1 connection
    
    def _sample_road_points(self, road, num_samples: int) -> List[Tuple[float, float]]:
        """Sample evenly spaced points along a road."""
        if not road.points or len(road.points) < 2:
            return []
        
        if len(road.points) <= num_samples:
            return road.points
        
        # Sample evenly spaced points
        indices = [int(i * (len(road.points) - 1) / (num_samples - 1)) for i in range(num_samples)]
        return [road.points[i] for i in indices]
    
    def _connect_districts_to_collectors(self, map_data: MapData, config: TransportationConfig):
        """Connect districts to the collector network if they're not already connected."""
        import math
        
        # Get all collectors and arterials
        collectors = [road for road in map_data.roads.values() if road.road_type == 'collector']
        arterials = [road for road in map_data.roads.values() if road.road_type == 'arterial']
        main_roads = collectors + arterials
        
        if not main_roads:
            return
        
        connector_id = 0
        max_connectors = 6  # Limit district connectors
        
        for district in map_data.districts.values():
            if not district.polygon or connector_id >= max_connectors:
                continue
            
            district_center = district.polygon.centroid
            
            # Find nearest main road
            nearest_road, nearest_point = self._find_nearest_road_point(
                (district_center.x, district_center.y), main_roads
            )
            
            if nearest_road and nearest_point:
                distance = math.sqrt(
                    (district_center.x - nearest_point[0])**2 + 
                    (district_center.y - nearest_point[1])**2
                )
                
                # Be more selective about connections
                if 200 < distance < 350 and district.district_type in ['downtown', 'commercial']:
                    connector_path = self._create_organized_path(
                        map_data, 
                        (district_center.x, district_center.y), 
                        nearest_point, 
                        'collector'
                    )
                    
                    if connector_path and len(connector_path) > 2:
                        connector = Road(
                            id=f"district_collector_{connector_id}",
                            points=connector_path,
                            road_type='collector',
                            width=config.road_styles['collector']['width'],
                            color=config.road_styles['collector']['color']
                        )
                        map_data.add_road(connector)
                        connector_id += 1

    def _generate_organic_local_roads(self, map_data: MapData, config: TransportationConfig):
        """Generate organized local roads within districts."""
        
        print("    → Creating organized local road networks...")
        
        # Only generate local roads in districts that have good access to main roads
        for district in map_data.districts.values():
            if district.polygon and district.district_type in ['residential', 'suburban', 'commercial']:
                # Check if district is connected to road network
                if self._district_has_road_access(map_data, district):
                    self._generate_organized_district_roads(map_data, district, config)
    
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
    
    def _generate_organized_district_roads(self, map_data: MapData, district, config: TransportationConfig):
        """Generate clean, organized local roads within a district."""
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
        
        # Create a simple organized pattern based on district type
        if district.district_type == 'residential':
            self._create_residential_grid(map_data, district, config, road_id)
        elif district.district_type == 'suburban':
            self._create_suburban_pattern(map_data, district, config, road_id)
        elif district.district_type == 'commercial':
            self._create_commercial_access(map_data, district, config, road_id)
    
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