import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle, Polygon as MatplotlibPolygon, Patch
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
import random
import math
from noise import pnoise2
from scipy.spatial import Voronoi, cKDTree
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import polygonize
from collections import defaultdict
import heapq
import json

# Custom JSON encoder to handle Numpy and Shapely types
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

class GTAMapGenerator:
    def __init__(self, width=2000, height=2000, seed=None):
        self.width = width
        self.height = height
        self.seed = seed if seed is not None else random.randint(0, 10000)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Generation parameters
        self.grid_size = 20
        self.grid_width = width // self.grid_size
        self.grid_height = height // self.grid_size
        self.water_level = 0.35
        self.beach_level = 0.4
        self.district_count = 10

        # Data structures
        self.heightmap = np.zeros((self.grid_height, self.grid_width))
        self.land_mask = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        self.districts = []
        self.buildings = []
        self.roads = []
        self.bridges = []
        self.blocks = []

        # Optional feature containers (populated in future enhancements)
        self.water_bodies = []  # rivers, lakes, coastline shapes
        self.landmarks = []     # airports, stadiums, etc.
        self.parks = []         # green public spaces

        # Road hierarchy parameters
        self.arterial_spacing = 400  # distance (in map units) between major boulevards
        self.collector_spacing = 200 # secondary roads
        self.local_road_density = 0.3 # chance of generating local roads
        self.road_curve_factor = 0.2  # how much roads curve (0-1)
        self.road_styles = {
            'arterial': {'color': '#333333', 'width': 8},
            'collector': {'color': '#555555', 'width': 5},
            'highway':  {'color': '#000000', 'width': 10},
            'local':    {'color': '#777777', 'width': 3},
            'rural':    {'color': '#999999', 'width': 2},
            'path':     {'color': '#bbbbbb', 'width': 1}
        }

        # District colors
        self.district_colors = {
            'downtown': '#ff9900',
            'commercial': '#33cc33',
            'industrial': '#666666',
            'residential': '#ff66cc',
            'suburban': '#66cccc',
            'hills': '#996633',
            'beach': '#33cccc',
            'port': '#6633cc',
            'airport': '#cc33cc',
            'park': '#33cc33',
        }
        
        # POI Types with their properties
        self.poi_types = {
            'airport': {
                'size': (150, 250),
                'color': '#cc33cc',
                'buffer': 100,  # minimum distance from other POIs
                'requires_flat': True
            },
            'stadium': {
                'size': (80, 120), 
                'color': '#3399cc',
                'buffer': 50,
                'requires_flat': True
            },
            'hospital': {
                'size': (40, 60),
                'color': '#ff3333',
                'buffer': 20,
                'requires_flat': False
            },
            'university': {
                'size': (100, 150),
                'color': '#9966cc',
                'buffer': 40,
                'requires_flat': False
            },
            'mall': {
                'size': (60, 80),
                'color': '#cc6699',
                'buffer': 30,
                'requires_flat': False
            },
            'port': {
                'size': (120, 180),
                'color': '#3366cc',
                'buffer': 80,
                'requires_flat': False,
                'requires_water': True
            }
        }

    def generate_complete_map(self):
        print(f"Generating new map with seed: {self.seed}")
        self._generate_heightmap()
        print("✓ Heightmap generated")
        self._define_land_and_water()
        print("✓ Land and water defined")
        self._generate_water_bodies()
        print("✓ Water bodies generated")
        self._place_districts()
        print("✓ Districts placed")
        self._generate_highway_network()
        print("✓ Highway network generated")
        self._generate_arterial_grid()
        print("✓ Arterial grid generated")
        self._generate_road_network()
        print("✓ Road network generated")
        self._define_city_blocks()
        print("✓ City blocks defined")
        self._generate_parks()
        print("✓ Parks generated")
        self._generate_pois()
        print("✓ Points of interest generated")
        self._populate_city_blocks()
        print("✓ Buildings populated")
        print("Map generation complete!")
        return self.get_map_data()

    def _generate_heightmap(self):
        """Generate a realistic heightmap with distinct mountain ranges and terrain features"""
        # Use multiple frequency Perlin noise for natural terrain
        scale = self.grid_width * 0.005
        mountain_scale = scale * 2
        
        # Create a base heightmap
        self.heightmap = np.zeros((self.grid_height, self.grid_width))
        
        # Generate 1-3 distinct mountain ranges
        mountain_count = random.randint(1, 3)
        mountain_centers = []
        
        for _ in range(mountain_count):
            # Place mountain ranges away from the edges
            mx = random.uniform(self.width * 0.2, self.width * 0.8)
            my = random.uniform(self.height * 0.2, self.height * 0.8)
            mountain_centers.append((mx, my))
        
        # Generate the base terrain
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                nx = x / self.grid_width - 0.5
                ny = y / self.grid_height - 0.5
                
                # Base continent shape (large features)
                e = (1.0 * pnoise2(1 * nx * scale, 1 * ny * scale, octaves=4, persistence=0.5, lacunarity=2.0, base=self.seed) +
                     0.5 * pnoise2(2 * nx * scale, 2 * ny * scale, octaves=4, persistence=0.5, lacunarity=2.0, base=self.seed+1) +
                     0.25 * pnoise2(4 * nx * scale, 4 * ny * scale, octaves=4, persistence=0.5, lacunarity=2.0, base=self.seed+2))
                e /= (1 + 0.5 + 0.25)
                
                # Create a coastal falloff
                d = min(1, (math.sqrt(nx**2 + ny**2) / math.sqrt(0.5**2 + 0.5**2))**0.5)
                e = (e + 1) / 2  # map to 0-1
                e = e * (1 - d * 0.5)  # lower edges to form island
                
                # Add mountain ranges
                mountain_influence = 0
                for mx, my in mountain_centers:
                    # Convert mountain center to grid coordinates
                    grid_mx = mx / self.grid_size
                    grid_my = my / self.grid_size
                    
                    # Distance to mountain center
                    dist = math.sqrt((x - grid_mx)**2 + (y - grid_my)**2)
                    max_dist = min(self.grid_width, self.grid_height) * 0.3
                    
                    if dist < max_dist:
                        # Create a distance-based falloff for mountain height
                        falloff = 1 - (dist / max_dist)**2
                        
                        # Add some noise to the mountain
                        mountain_noise = pnoise2(x * mountain_scale * 0.1, y * mountain_scale * 0.1, 
                                              octaves=6, persistence=0.6, lacunarity=2.2, base=self.seed+10) 
                        
                        # Higher persistence and octaves make more detailed mountains
                        mountain_height = falloff * (mountain_noise * 0.5 + 0.5) * 0.4
                        mountain_influence = max(mountain_influence, mountain_height)
                
                # Combine base terrain with mountains
                self.heightmap[y, x] = min(0.95, e + mountain_influence)  # Cap at 0.95 to avoid pure white

    def _define_land_and_water(self):
        self.land_mask = self.heightmap > self.water_level
        
    def _generate_water_bodies(self):
        """Generate natural water features like rivers and lakes"""
        # Generate up to 3 rivers across the map
        num_rivers = random.randint(0, 3)
        for i in range(num_rivers):
            self._generate_river()
            
        # Generate some lakes in lower-elevation areas
        num_lakes = random.randint(2, 5)
        for i in range(num_lakes):
            self._generate_lake()
    
    def _generate_river(self):
        """Generate a more visible and realistic winding river"""
        # Increase the number of rivers and their width for visibility
        # Choose a starting edge - prefer top (mountains) or sides for more natural flow
        edge = random.choice(['top', 'top', 'right', 'left', 'bottom'])
        
        # Generate start point and initial direction
        if edge == 'top':
            start_x = random.randint(int(self.width * 0.2), int(self.width * 0.8))
            start_y = 0
            direction = (random.uniform(-0.3, 0.3), 1)  # Flow downward with slight variance
        elif edge == 'right':
            start_x = self.width
            start_y = random.randint(int(self.height * 0.2), int(self.height * 0.8))
            direction = (-1, random.uniform(-0.3, 0.3))  # Flow leftward with variance
        elif edge == 'bottom':
            start_x = random.randint(int(self.width * 0.2), int(self.width * 0.8))
            start_y = self.height
            direction = (random.uniform(-0.3, 0.3), -1)  # Flow upward with variance
        else:  # left
            start_x = 0
            start_y = random.randint(int(self.height * 0.2), int(self.height * 0.8))
            direction = (1, random.uniform(-0.3, 0.3))  # Flow rightward with variance
            
        # Normalize direction vector
        magnitude = (direction[0]**2 + direction[1]**2)**0.5
        direction = (direction[0]/magnitude, direction[1]/magnitude)
        
        # Use heightmap to influence river flow - rivers tend to flow toward lower elevations
        # Increase river width for better visibility
        river_points = [(start_x, start_y)]
        x, y = start_x, start_y
        
        # Make rivers wider and more prominent
        river_width = random.uniform(30, 60)  # Increased width
        river_length = random.randint(int(max(self.width, self.height) * 0.5), 
                                   max(self.width, self.height)) # Longer rivers
        
        # Parameters for more natural river curves
        perlin_scale = 0.005  # Reduced scale for smoother curves
        perlin_strength = 70  # Increased strength for more pronounced curves
        terrain_influence = 0.5  # How much the terrain affects river direction
        
        for i in range(river_length):
            # Get current grid position
            grid_x = min(int(x / self.grid_size), self.grid_width-2)
            grid_y = min(int(y / self.grid_size), self.grid_height-2)
            
            # Terrain influence - rivers tend to flow downhill (along gradient)
            if 0 <= grid_x < self.grid_width-1 and 0 <= grid_y < self.grid_height-1:
                # Calculate terrain gradient (simplified)
                dx_terrain = self.heightmap[grid_y, grid_x+1] - self.heightmap[grid_y, grid_x]
                dy_terrain = self.heightmap[grid_y+1, grid_x] - self.heightmap[grid_y, grid_x]
                
                # Normalize terrain gradient if non-zero
                terrain_mag = math.sqrt(dx_terrain**2 + dy_terrain**2)
                if terrain_mag > 0.001:
                    dx_terrain /= terrain_mag
                    dy_terrain /= terrain_mag
            else:
                dx_terrain, dy_terrain = 0, 0
            
            # Add Perlin noise influence for natural meandering
            perlin_val = pnoise2(x * perlin_scale, y * perlin_scale, octaves=3, 
                               persistence=0.6, lacunarity=2.0, base=self.seed+100)
            noise_angle = perlin_val * 2 * math.pi  # Convert to angle
            
            # Blend original direction, terrain influence, and noise
            # Rivers should mostly follow the terrain gradient (downhill) with some meandering
            nx = direction[0] * (0.4) - dx_terrain * terrain_influence + math.cos(noise_angle) * (0.2)
            ny = direction[1] * (0.4) - dy_terrain * terrain_influence + math.sin(noise_angle) * (0.2)
            
            # Renormalize direction
            mag = math.sqrt(nx**2 + ny**2)
            if mag > 0.001:
                nx, ny = nx/mag, ny/mag
            
            # Update direction for next iteration (rivers tend to maintain their course)
            direction = (nx, ny)
            
            # Take a step
            step_size = 15  # Larger step size for more efficient river generation
            x += nx * step_size
            y += ny * step_size
            
            # Check if we've gone outside the map or reached water
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                break
                
            river_points.append((x, y))
            
        # Only add if we have enough points
        if len(river_points) > 10:
            # Create a more visibly blue river
            self.water_bodies.append({
                'type': 'river',
                'points': river_points,
                'width': river_width,
                'color': '#0077cc'  # More vibrant blue
            })
            
            # Update land mask to reflect the river with a wider area of influence
            for i in range(len(river_points)-1):
                x1, y1 = river_points[i]
                x2, y2 = river_points[i+1]
                for t in np.linspace(0, 1, 30):  # More sample points for smoother rivers
                    x = int(x1 * (1-t) + x2 * t)
                    y = int(y1 * (1-t) + y2 * t)
                    
                    # Convert to grid coordinates
                    grid_x = min(int(x / self.grid_size), self.grid_width-1)
                    grid_y = min(int(y / self.grid_size), self.grid_height-1)
                    
                    # Set a wider circle around each point as water
                    if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                        river_grid_width = max(2, int(river_width / self.grid_size / 2))  # Wider river
                        for dx in range(-river_grid_width, river_grid_width+1):
                            for dy in range(-river_grid_width, river_grid_width+1):
                                if dx*dx + dy*dy <= river_grid_width*river_grid_width:
                                    nx, ny = grid_x + dx, grid_y + dy
                                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                                        self.land_mask[ny, nx] = False
    
    def _generate_lake(self):
        """Generate a lake in a low area of the map"""
        # Find areas with low elevation but above water level
        potential_areas = np.argwhere((self.heightmap > self.water_level) & 
                                     (self.heightmap < self.water_level + 0.1))
        
        if len(potential_areas) == 0:
            return  # No suitable areas found
            
        # Select a random point in these areas
        idx = np.random.choice(len(potential_areas))
        center_y, center_x = potential_areas[idx]
        
        # Convert to map coordinates
        center_x = center_x * self.grid_size
        center_y = center_y * self.grid_size
        
        # Generate an irregular lake shape
        num_points = random.randint(8, 15)
        lake_radius = random.uniform(30, 100)
        lake_points = []
        
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            # Vary the radius with perlin noise
            radius_var = pnoise2(math.cos(angle), math.sin(angle), octaves=2, 
                               persistence=0.5, lacunarity=2.0, base=self.seed+200)
            radius = lake_radius * (1 + radius_var * 0.5)  # +/- 50% variation
            x = center_x + math.cos(angle) * radius
            y = center_y + math.sin(angle) * radius
            lake_points.append((x, y))
            
        # Close the loop
        lake_points.append(lake_points[0])
        
        # Add the lake to water bodies
        try:
            lake_polygon = Polygon(lake_points)
            if lake_polygon.is_valid and lake_polygon.area > 100:
                self.water_bodies.append({
                    'type': 'lake',
                    'polygon': lake_polygon,
                    'color': '#0066cc'
                })
                
                # Update land mask for the lake
                min_x, min_y, max_x, max_y = lake_polygon.bounds
                min_grid_x = max(0, int(min_x / self.grid_size))
                min_grid_y = max(0, int(min_y / self.grid_size))
                max_grid_x = min(self.grid_width-1, int(max_x / self.grid_size))
                max_grid_y = min(self.grid_height-1, int(max_y / self.grid_size))
                
                for grid_y in range(min_grid_y, max_grid_y+1):
                    for grid_x in range(min_grid_x, max_grid_x+1):
                        # Check if grid cell center is in lake polygon
                        point = Point(grid_x * self.grid_size + self.grid_size/2, 
                                     grid_y * self.grid_size + self.grid_size/2)
                        if lake_polygon.contains(point):
                            self.land_mask[grid_y, grid_x] = False
        except:
            # Skip invalid polygons
            pass

    def _generate_highway_network(self):
        """Generate a network of highways connecting major areas of the map"""
        # Create a few main highway paths across the map
        num_highways = random.randint(2, 4)
        
        # Define highway entry/exit points around the edges of the map
        edge_points = []
        
        # Add potential entry/exit points at the edges
        for i in range(4, self.width-1, int(self.width/6)):  # Top edge
            if np.any(self.land_mask[:20, i//self.grid_size]):
                edge_points.append((i, 0, 'top'))
        
        for i in range(4, self.height-1, int(self.height/6)):  # Right edge
            if np.any(self.land_mask[i//self.grid_size, -20:]):
                edge_points.append((self.width, i, 'right'))
        
        for i in range(4, self.width-1, int(self.width/6)):  # Bottom edge
            if np.any(self.land_mask[-20:, i//self.grid_size]):
                edge_points.append((i, self.height, 'bottom'))
        
        for i in range(4, self.height-1, int(self.height/6)):  # Left edge
            if np.any(self.land_mask[i//self.grid_size, :20]):
                edge_points.append((0, i, 'left'))
        
        # Shuffle the edge points to get random connections
        random.shuffle(edge_points)
        
        # Generate highways between different edge points
        for i in range(min(num_highways, len(edge_points)//2)):
            if len(edge_points) < 2:
                break
                
            start_point = edge_points.pop(0)
            # Try to find an endpoint on a different edge
            end_idx = 0
            for j, ep in enumerate(edge_points):
                if ep[2] != start_point[2]:  # Different edge type
                    end_idx = j
                    break
            end_point = edge_points.pop(end_idx)
            
            # Create a curved highway path between these points
            self._generate_curved_highway(start_point, end_point)
            
        # Also add a ring road/beltway if the map is large enough
        if self.width > 1000 and self.height > 1000 and random.random() > 0.3:
            self._generate_ring_highway()
    
    def _generate_curved_highway(self, start_point, end_point):
        """Generate a curved highway path between two points"""
        start_x, start_y, _ = start_point
        end_x, end_y, _ = end_point
        
        # Generate 1-3 control points for a natural curve
        num_controls = random.randint(1, 3)
        control_points = []
        
        # Calculate the midpoint and use it as basis for control point calculation
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        
        for i in range(num_controls):
            # Generate points that deviate from the straight line
            # More deviation for the middle control point if there are multiple
            if num_controls > 1:
                deviation = 0.3 if i == num_controls // 2 else 0.15
            else:
                deviation = 0.2
                
            # Calculate perpendicular direction to create offset
            dx = end_x - start_x
            dy = end_y - start_y
            # Perpendicular direction (rotate 90 degrees)
            perp_x = -dy
            perp_y = dx
            # Normalize
            mag = (perp_x**2 + perp_y**2)**0.5
            if mag > 0:
                perp_x /= mag
                perp_y /= mag
            
            # Control points distributed along the path
            t = (i + 1) / (num_controls + 1)  # Distribute control points
            point_x = start_x * (1-t) + end_x * t
            point_y = start_y * (1-t) + end_y * t
            
            # Add perpendicular offset
            offset = random.uniform(-1, 1) * deviation * min(self.width, self.height)
            point_x += perp_x * offset
            point_y += perp_y * offset
            
            # Ensure the control point is within bounds
            point_x = max(0, min(self.width, point_x))
            point_y = max(0, min(self.height, point_y))
            
            control_points.append((point_x, point_y))
        
        # Generate the Bezier curve for the highway
        highway_points = []
        steps = 50  # Number of points to generate along the curve
        
        all_points = [(start_x, start_y)] + control_points + [(end_x, end_y)]
        for t in np.linspace(0, 1, steps):
            point = self._bezier_point(all_points, t)
            highway_points.append(point)
        
        # Add the highway
        self._add_polyline_as_road(highway_points, 'highway')
    
    def _bezier_point(self, points, t):
        """Calculate a point along a Bezier curve defined by control points at parameter t"""
        n = len(points) - 1
        point = [0, 0]
        
        for i, p in enumerate(points):
            # Bernstein polynomial
            coeff = math.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
            point[0] += coeff * p[0]
            point[1] += coeff * p[1]
            
        return tuple(point)
    
    def _generate_ring_highway(self):
        """Generate a ring road around the central part of the map"""
        # Define the ring as an ellipse
        center_x = self.width / 2
        center_y = self.height / 2
        # Make the ring somewhat smaller than the full map
        a = self.width * 0.3  # Semi-major axis
        b = self.height * 0.3  # Semi-minor axis
        
        # Add some randomness to the center
        center_x += random.uniform(-self.width * 0.1, self.width * 0.1)
        center_y += random.uniform(-self.height * 0.1, self.height * 0.1)
        
        # Generate points along the ellipse
        ring_points = []
        steps = 50
        
        for i in range(steps):
            angle = 2 * math.pi * i / steps
            # Add some noise to the radius at each angle
            noise = pnoise2(math.cos(angle), math.sin(angle), octaves=2, 
                          persistence=0.5, lacunarity=2.0, base=self.seed+300)
            noise_factor = 1 + noise * 0.15  # +/- 15% variation
            
            x = center_x + a * noise_factor * math.cos(angle)
            y = center_y + b * noise_factor * math.sin(angle)
            
            # Ensure the point is within bounds
            x = max(0, min(self.width, x))
            y = max(0, min(self.height, y))
            
            ring_points.append((x, y))
            
        # Close the loop
        ring_points.append(ring_points[0])
        
        # Add the ring highway
        self._add_polyline_as_road(ring_points, 'highway')

    def _place_districts(self):
        # Find viable land points for district centers
        land_points = np.argwhere(self.land_mask)
        if len(land_points) == 0:
            print("Warning: No land to place districts on.")
            return

        # Prioritize flatter areas for certain districts
        gradient_x, gradient_y = np.gradient(self.heightmap)
        slope = np.sqrt(gradient_x**2 + gradient_y**2)
        flat_areas = np.argwhere((self.land_mask) & (slope < 0.01))
        
        # Ensure we have enough districts to include all important types
        self.district_count = max(self.district_count, 12)  # Ensure enough districts for all types
        
        # Define the key district types we want to ensure are included
        key_districts = ['downtown', 'commercial', 'industrial', 'residential', 'suburban', 
                         'hills', 'beach', 'port', 'airport', 'park']
        
        # Place important districts first, then fill the rest with random types
        centers = []
        districts_placed = []
        
        # Explicitly place key district types first
        for district_type in key_districts:
            point = None
            
            # Special placement logic for different district types
            if district_type in ['downtown', 'commercial']:
                # Downtown and commercial districts should be central
                center_area = np.array([self.grid_width/2, self.grid_height/2])
                distances = np.sum((land_points - center_area) ** 2, axis=1)
                closest_points = land_points[np.argsort(distances)[:20]]
                if len(closest_points) > 0:
                    point = closest_points[np.random.choice(len(closest_points))]
            
            elif district_type == 'airport':
                # Airports need large flat areas
                if len(flat_areas) > 0:
                    # Sort by distance from edge for easier access
                    edge_dist = np.minimum(
                        np.minimum(flat_areas[:, 0], self.grid_width - flat_areas[:, 0]),
                        np.minimum(flat_areas[:, 1], self.grid_height - flat_areas[:, 1])
                    )
                    airport_candidates = flat_areas[edge_dist < self.grid_width/4]
                    if len(airport_candidates) > 0:
                        point = airport_candidates[np.random.choice(len(airport_candidates))]
                    else:
                        point = flat_areas[np.random.choice(len(flat_areas))]
            
            elif district_type == 'beach':
                # Beaches should be near water at beach level
                beach_points = np.argwhere((self.heightmap > self.water_level) & 
                                          (self.heightmap < self.beach_level))
                if len(beach_points) > 0:
                    point = beach_points[np.random.choice(len(beach_points))]
            
            elif district_type == 'port':
                # Ports should be right next to water
                water_edges = []
                for i in range(1, self.grid_height-1):
                    for j in range(1, self.grid_width-1):
                        if self.land_mask[i, j]:
                            # Check if adjacent to water
                            if (not self.land_mask[i-1, j] or not self.land_mask[i+1, j] or
                                not self.land_mask[i, j-1] or not self.land_mask[i, j+1]):
                                water_edges.append([i, j])
                if water_edges:
                    point = np.array(water_edges[np.random.choice(len(water_edges))])
            
            # Default placement if special placement failed
            if point is None:
                if len(land_points) > 0:
                    point = land_points[np.random.choice(len(land_points))]
                else:
                    continue  # Skip if no valid points
            
            # Convert grid coordinates to world coordinates
            point_world = point * self.grid_size
            
            # Check if this point is far enough from existing centers
            too_close = False
            min_distance = 300  # Minimum distance between district centers
            for existing_center in centers:
                if np.linalg.norm(existing_center - point_world) < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                centers.append(point_world)
                districts_placed.append(district_type)
            
            # Stop if we've reached our district count
            if len(centers) >= self.district_count:
                break
        
        # Fill remaining district slots with random types if needed
        remaining_districts = self.district_count - len(centers)
        if remaining_districts > 0:
            random_types = key_districts.copy()
            random.shuffle(random_types)
            
            for i in range(remaining_districts):
                district_type = random_types[i % len(random_types)]
                
                # Find a point that's not too close to existing centers
                attempts = 0
                found_point = False
                while attempts < 20 and not found_point:
                    idx = np.random.choice(len(land_points))
                    point = land_points[idx]
                    point_world = point * self.grid_size
                    
                    # Check distance from existing centers
                    too_close = False
                    for existing_center in centers:
                        if np.linalg.norm(existing_center - point_world) < 250:
                            too_close = True
                            break
                    
                    if not too_close:
                        centers.append(point_world)
                        districts_placed.append(district_type)
                        found_point = True
                    
                    attempts += 1
        
        # Generate Voronoi diagram to define district boundaries
        if len(centers) >= 3:  # Voronoi requires at least 3 points
            vor = Voronoi([c.tolist() for c in centers])
            
            # Extract district vertices from Voronoi diagram
            for i, (center, district_type) in enumerate(zip(centers, districts_placed)):
                vertices = []
                for region_idx in vor.point_region:
                    if vor.point_region[i] == region_idx:
                        region = vor.regions[region_idx]
                        if not -1 in region and len(region) > 0:
                            for vertex_idx in region:
                                if vertex_idx >= 0:
                                    vertex = vor.vertices[vertex_idx].tolist()
                                    # Clip to map boundaries
                                    vertex[0] = max(0, min(vertex[0], self.width))
                                    vertex[1] = max(0, min(vertex[1], self.height))
                                    vertices.append(vertex)
                
                self.districts.append({
                    'center': center,
                    'type': district_type,
                    'vertices': vertices,
                    'radius': random.uniform(200, 400),
                    'polygon': None
                })
                
                # Create polygon for district if we have vertices
                if len(vertices) >= 3:
                    from shapely.geometry import Polygon
                    try:
                        self.districts[-1]['polygon'] = Polygon(vertices)
                    except Exception:
                        pass  # Skip invalid polygons

    def _generate_arterial_grid(self):
        """Generate an organic, realistic road network based on terrain and districts."""
        # We'll create several types of roads:
        # 1. Major highways connecting districts
        # 2. Radial roads emanating from downtown/central areas
        # 3. Curved roads following terrain contours
        # 4. Local roads within districts
        
        # Use heightmap and district locations to guide road placement
        major_nodes = []
        
        # Add district centers as major nodes
        for district in self.districts:
            if 'center' in district:
                major_nodes.append((district['center'][0], district['center'][1]))
        
        # If no districts, create some default nodes
        if not major_nodes:
            major_nodes = [(self.width/2, self.height/2)]
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                r = min(self.width, self.height) * 0.4
                x = self.width/2 + r * np.cos(angle)
                y = self.height/2 + r * np.sin(angle)
                major_nodes.append((x, y))
        
        # Connect districts with natural roads following terrain
        self._connect_major_nodes(major_nodes)
        
        # Generate radial roads from downtown/center
        downtown_centers = []
        for district in self.districts:
            if district['type'] in ['downtown', 'commercial']:
                downtown_centers.append((district['center'][0], district['center'][1]))
        
        # If no downtown, use map center
        if not downtown_centers:
            downtown_centers = [(self.width/2, self.height/2)]
        
        # Generate radial roads from each downtown center
        for center in downtown_centers:
            self._generate_radial_roads(center)
        
        # Generate contour-following roads based on heightmap
        self._generate_terrain_following_roads()
        
        # Add some random connector roads
        self._generate_random_connectors(major_nodes)
    
    def _connect_major_nodes(self, nodes):
        """Connect major nodes with a clear, organized road network using a hierarchical approach."""
        if len(nodes) < 2:
            return
            
        # First, identify important nodes that should be connected by highways
        # This will typically include downtown, commercial centers, and airports
        downtown_nodes = []
        important_nodes = []
        other_nodes = []
        
        # Classify nodes based on district types
        for i, node in enumerate(nodes):
            # Find the district this node belongs to
            node_district = None
            for district in self.districts:
                if 'center' in district and district['center'][0] == node[0] and district['center'][1] == node[1]:
                    node_district = district
                    break
            
            if node_district:
                if node_district['type'] in ['downtown']:
                    downtown_nodes.append((i, node))
                elif node_district['type'] in ['commercial', 'airport', 'port']:
                    important_nodes.append((i, node))
                else:
                    other_nodes.append((i, node))
            else:
                other_nodes.append((i, node))
        
        # If we don't have a downtown node, use the center of the map
        if not downtown_nodes:
            center_node = (self.width/2, self.height/2)
            closest_idx = min(range(len(nodes)), key=lambda i: ((nodes[i][0] - center_node[0])**2 + 
                                                           (nodes[i][1] - center_node[1])**2))
            downtown_nodes = [(closest_idx, nodes[closest_idx])]
        
        # Calculate distances between all nodes
        n = len(nodes)
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((nodes[i][0] - nodes[j][0])**2 + (nodes[i][1] - nodes[j][1])**2)
                
                # Classify edge type based on endpoints
                i_important = any(idx == i for idx, _ in downtown_nodes + important_nodes)
                j_important = any(idx == j for idx, _ in downtown_nodes + important_nodes)
                
                # Priority score: lower is higher priority
                # Connect downtown to downtown first, then downtown to important, then important to important
                priority = 1
                if any(idx == i for idx, _ in downtown_nodes) and any(idx == j for idx, _ in downtown_nodes):
                    priority = 0
                elif (any(idx == i for idx, _ in downtown_nodes) and any(idx == j for idx, _ in important_nodes)) or \
                     (any(idx == j for idx, _ in downtown_nodes) and any(idx == i for idx, _ in important_nodes)):
                    priority = 0.5
                elif i_important and j_important:
                    priority = 1
                else:
                    priority = 2 + dist * 0.0001  # Lower priority, influenced slightly by distance
                
                edges.append((i, j, dist, priority))
        
        # Sort edges by priority (first) and then by distance (second)
        edges.sort(key=lambda x: (x[3], x[2]))
        
        # Union-find data structure for Kruskal's algorithm
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            parent[find(x)] = find(y)
        
        # Kruskal's algorithm to create minimum spanning tree with priority
        mst_edges = []
        for i, j, dist, priority in edges:
            if find(i) != find(j):
                union(i, j)
                mst_edges.append((i, j, priority))
        
        # Add some additional connections to create cycles (more realistic)
        # Focus on adding more connections between important nodes
        num_extra_edges = min(n//2, len(edges) - len(mst_edges))  # More extra edges
        extra_count = 0
        
        for i, j, dist, priority in edges:
            if not any((i, j, p) in mst_edges or (j, i, p) in mst_edges for p in [0, 0.5, 1, 2]):
                # Prefer adding connections between important nodes
                i_important = any(idx == i for idx, _ in downtown_nodes + important_nodes)
                j_important = any(idx == j for idx, _ in downtown_nodes + important_nodes)
                
                if (i_important or j_important) and dist < min(self.width, self.height) * 0.3:
                    mst_edges.append((i, j, priority))
                    extra_count += 1
                elif extra_count < num_extra_edges // 2 and dist < min(self.width, self.height) * 0.2:
                    mst_edges.append((i, j, priority))
                    extra_count += 1
                
                if extra_count >= num_extra_edges:
                    break
        
        # Create roads between connected nodes with appropriate road types
        noise_scale = 0.005
        noise_strength = 300
        
        for i, j, priority in mst_edges:
            start = nodes[i]
            end = nodes[j]
            
            # Set road type based on priority
            if priority <= 0.5:  # Downtown connections get highways
                road_type = 'highway'
                road_width = 15  # Wider
                curve_factor = 0.7  # Less curvy (more direct)
            elif priority <= 1.0:  # Important node connections get arterials
                road_type = 'arterial'
                road_width = 12
                curve_factor = 0.8
            else:  # Everything else gets collector roads
                road_type = 'collector'
                road_width = 8
                curve_factor = 1.0
                
            # Check if path is over water, if so, create a bridge
            is_bridge = False
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            mid_grid_x = int(mid_x / self.grid_size)
            mid_grid_y = int(mid_y / self.grid_size)
            if 0 <= mid_grid_x < self.grid_width and 0 <= mid_grid_y < self.grid_height:
                if not self.land_mask[mid_grid_y, mid_grid_x]:
                    is_bridge = True
            
            # Generate curved path between points
            dist = np.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
            num_points = max(5, int(dist / 40))  # More points for smoother curves
            
            # Create base line with controlled curve
            path = []
            for t in np.linspace(0, 1, num_points):
                x = start[0] * (1-t) + end[0] * t
                y = start[1] * (1-t) + end[1] * t
                
                # Add perlin noise deviation
                angle = math.atan2(end[1] - start[1], end[0] - start[0])
                perp_x = math.sin(angle)
                perp_y = -math.cos(angle)
                
                # Less noise for straighter roads
                noise_val = pnoise2(x * noise_scale, y * noise_scale, octaves=2, 
                                   persistence=0.5, lacunarity=2.0, base=self.seed+5)
                
                # Scale noise by distance from endpoints (less curve near start/end)
                # This creates more natural looking road connections
                endpoint_factor = min(t, 1-t) * 2  # 0 at endpoints, 1 in middle
                offset = noise_val * noise_strength * endpoint_factor * curve_factor
                
                # Apply offset perpendicular to direction
                x += perp_x * offset
                y += perp_y * offset
                
                path.append((x, y))
                
            # Add as road or bridge
            if is_bridge:
                self.bridges.append({
                    'points': path,
                    'width': road_width,
                    'type': 'bridge'
                })
            else:
                self.roads.append({
                    'points': path,
                    'width': road_width,
                    'type': road_type
                })
    
    def _generate_radial_roads(self, center):
        """Generate roads radiating from a central point (like downtown)."""
        num_radials = random.randint(5, 8)  # Number of main radial roads
        
        for i in range(num_radials):
            angle = (i / num_radials) * 2 * np.pi
            # Add some randomness to the angle
            angle += random.uniform(-0.2, 0.2)
            
            # Calculate the end point at the map edge
            # Find intersection with map boundary
            dist = max(self.width, self.height) * 1.5  # Ensure it's long enough
            end_x = center[0] + dist * np.cos(angle)
            end_y = center[1] + dist * np.sin(angle)
            
            # Clip to map boundaries
            if end_x < 0:
                t = -center[0] / (end_x - center[0])
                end_y = center[1] + t * (end_y - center[1])
                end_x = 0
            elif end_x > self.width:
                t = (self.width - center[0]) / (end_x - center[0])
                end_y = center[1] + t * (end_y - center[1])
                end_x = self.width
            
            if end_y < 0:
                t = -center[1] / (end_y - center[1])
                end_x = center[0] + t * (end_x - center[0])
                end_y = 0
            elif end_y > self.height:
                t = (self.height - center[1]) / (end_y - center[1])
                end_x = center[0] + t * (end_x - center[0])
                end_y = self.height
            
            # Create a slightly curved radial road
            points = []
            num_points = 10
            
            for j in range(num_points + 1):
                t = j / num_points
                x = center[0] + t * (end_x - center[0])
                y = center[1] + t * (end_y - center[1])
                
                # Add slight curve using Perlin noise
                if 0 < j < num_points:
                    noise_val = pnoise2(x * 0.002, y * 0.002, octaves=2, persistence=0.5, 
                                      lacunarity=2.0, base=self.seed + i * 100)
                    perp_x = -(end_y - center[1])
                    perp_y = (end_x - center[0])
                    perp_len = np.sqrt(perp_x**2 + perp_y**2)
                    if perp_len > 0:
                        perp_x /= perp_len
                        perp_y /= perp_len
                        
                        # Scale curve based on distance from center
                        curve_scale = 100 * t * (1-t)
                        x += perp_x * noise_val * curve_scale
                        y += perp_y * noise_val * curve_scale
                
                points.append((x, y))
            
            # Main radials are arterial roads
            self._add_polyline_as_road(points, 'arterial')
    
    def _generate_terrain_following_roads(self):
        """Generate roads that follow terrain contours."""
        # Use heightmap gradients to find contour lines
        gradient_x, gradient_y = np.gradient(self.heightmap)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Find areas with moderate slope - good for contour roads
        moderate_slope = ((gradient_magnitude > 0.01) & (gradient_magnitude < 0.05) & self.land_mask)
        contour_candidates = np.argwhere(moderate_slope)
        
        # Select some random starting points for contour roads
        num_contour_roads = random.randint(3, 7)
        
        if len(contour_candidates) == 0:
            return
            
        for _ in range(num_contour_roads):
            if len(contour_candidates) == 0:
                break
                
            idx = np.random.choice(len(contour_candidates))
            start = contour_candidates[idx] * self.grid_size
            
            # Follow the contour by moving perpendicular to the gradient
            points = [(start[1], start[0])]  # Note: we swap x and y here because of numpy indexing
            current = start.copy()
            
            # Generate a contour road of reasonable length
            for _ in range(100):
                # Get gradient at current position
                grid_x = int(current[0])
                grid_y = int(current[1])
                
                # Check if we're out of bounds
                if (grid_y >= self.grid_height or grid_x >= self.grid_width or 
                    grid_y < 0 or grid_x < 0):
                    break
                    
                # Get gradient direction (perpendicular to contour)
                gx = gradient_x[grid_y, grid_x]
                gy = gradient_y[grid_y, grid_x]
                
                if abs(gx) + abs(gy) < 0.001:  # Too flat, stop
                    break
                    
                # Move perpendicular to gradient (along contour)
                # Normalize gradient vector
                mag = np.sqrt(gx*gx + gy*gy)
                if mag > 0:
                    gx /= mag
                    gy /= mag
                    
                # Perpendicular direction
                px, py = -gy, gx
                
                # Random direction along contour
                if random.random() < 0.5:
                    px, py = -px, -py
                    
                # Step size along contour
                step_size = 20
                
                # Update current position
                current[0] += px * step_size
                current[1] += py * step_size
                
                # Add some noise for natural roads
                noise_val = pnoise2(current[0] * 0.01, current[1] * 0.01, 
                                  octaves=2, persistence=0.5, lacunarity=2.0,
                                  base=self.seed)
                
                current[0] += noise_val * 5
                current[1] += noise_val * 5
                
                # Ensure we stay within map bounds
                current[0] = max(0, min(current[0], self.grid_width - 1))
                current[1] = max(0, min(current[1], self.grid_height - 1))
                
                # Only add point if still on land
                grid_x = int(current[0])
                grid_y = int(current[1])
                
                if grid_y < self.grid_height and grid_x < self.grid_width and grid_y >= 0 and grid_x >= 0:
                    if self.land_mask[grid_y, grid_x]:
                        points.append((current[1] * self.grid_size, current[0] * self.grid_size))
                    else:
                        break  # Stop at water
                else:
                    break  # Stop at map boundary
            
            # Add contour road if long enough
            if len(points) >= 10:
                self._add_polyline_as_road(points, 'collector')
    
    def _generate_random_connectors(self, major_nodes):
        """Generate some random connector roads between areas."""
        # Add some random roads connecting different areas
        num_connectors = random.randint(5, 10)
        
        for _ in range(num_connectors):
            # Pick a random point on an existing road
            if not self.roads:
                continue  # No roads to connect to
                
            road_idx = random.randint(0, len(self.roads) - 1)
            road = self.roads[road_idx]
            
            if len(road['points']) < 2:
                continue
                
            # Pick a random point on this road
            point_idx = random.randint(0, len(road['points']) - 1)
            start = road['points'][point_idx]
            
            # Generate a random direction
            angle = random.uniform(0, 2 * np.pi)
            
            # Road length
            length = random.uniform(300, 700)
            
            # Calculate end point
            end_x = start[0] + length * np.cos(angle)
            end_y = start[1] + length * np.sin(angle)
            
            # Clip to map boundaries
            end_x = max(0, min(end_x, self.width))
            end_y = max(0, min(end_y, self.height))
            
            # Generate curved road
            points = [start]
            
            # Create a curved path
            num_points = random.randint(4, 8)
            for i in range(1, num_points):
                t = i / num_points
                
                # Linear interpolation
                x = start[0] + t * (end_x - start[0])
                y = start[1] + t * (end_y - start[1])
                
                # Add some noise for curves
                noise_val = pnoise2(x * 0.01, y * 0.01, octaves=2, persistence=0.6, 
                                 lacunarity=2.0, base=self.seed + _ * 100)
                
                # Create perpendicular vector for offset
                dx = end_x - start[0]
                dy = end_y - start[1]
                
                # Normalize
                length = max(0.001, np.sqrt(dx*dx + dy*dy))
                perpx, perpy = -dy/length, dx/length
                
                # Apply curve
                curve_factor = 100 * noise_val * t * (1-t)
                x += perpx * curve_factor
                y += perpy * curve_factor
                
                points.append((x, y))
            
            points.append((end_x, end_y))
            
            # Add connector road (collector or local road)
            road_type = 'collector' if random.random() < 0.3 else 'local'
            self._add_polyline_as_road(points, road_type)

    def _add_polyline_as_road(self, pts, rtype):
        """Split a polyline into land and bridge segments and store them."""
        if len(pts) < 2:
            return
        current_segment = [pts[0]]
        is_bridge = not self.land_mask[int(pts[0][1]/self.grid_size)%self.grid_height, int(pts[0][0]/self.grid_size)%self.grid_width]
        for i in range(1, len(pts)):
            p = pts[i]
            on_land = self.land_mask[int(p[1]/self.grid_size)%self.grid_height, int(p[0]/self.grid_size)%self.grid_width]
            if on_land != (not is_bridge):
                # Segment type changed
                if len(current_segment) > 1:
                    self._store_segment(current_segment, rtype, is_bridge)
                current_segment = [pts[i-1], p]
                is_bridge = not on_land
            else:
                current_segment.append(p)
        # store last
        if len(current_segment) > 1:
            self._store_segment(current_segment, rtype, is_bridge)

    def _store_segment(self, seg_pts, rtype, is_bridge):
        if is_bridge:
            self.bridges.append({'points': seg_pts.copy(), 'type': 'bridge', 'width': self.road_styles[rtype]['width']})
        else:
            self.roads.append({'points': seg_pts.copy(), 'type': rtype, 'width': self.road_styles[rtype]['width']})

    def _generate_road_network(self):
        # Connect district centers to the main arterial road network
        all_road_points = []
        for r in self.roads: 
            all_road_points.extend(r['points'])
        for b in self.bridges:
            all_road_points.extend(b['points'])

        if not all_road_points:
            print("Warning: No arterial roads to connect to.")
            return

        road_tree = cKDTree(all_road_points)
        district_centers = [d['center'] for d in self.districts]

        for center in district_centers:
            # Find the nearest point on the arterial network
            dist, idx = road_tree.query(center)
            nearest_road_point = all_road_points[idx]

            # Define start and end for A* search (in grid coordinates)
            start_node = tuple((center / self.grid_size).astype(int))
            end_node = tuple((np.array(nearest_road_point) / self.grid_size).astype(int))
            
            if start_node == end_node: continue

            # Generate a collector road to the arterial grid
            path = self._a_star_search(start_node, end_node)
            if path:
                path_points = [(p[1]*self.grid_size, p[0]*self.grid_size) for p in path]
                self._add_polyline_as_road(path_points, 'collector')

    def _a_star_search(self, start, end):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, end)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                return self._reconstruct_path(came_from, current)

            for neighbor in self._get_neighbors(current):
                tentative_g_score = g_score[current] + self._cost_to_move(current, neighbor)
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, end)
                    if neighbor not in [i[1] for i in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None

    def _get_neighbors(self, node):
        y, x = node
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0: continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.grid_height and 0 <= nx < self.grid_width:
                    neighbors.append((ny, nx))
        return neighbors

    def _heuristic(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def _cost_to_move(self, a, b):
        cost = self._heuristic(a, b)
        # Higher cost for steep terrain
        cost += abs(self.heightmap[a] - self.heightmap[b]) * 25

        # High cost for water to represent bridge building
        if not self.land_mask[b]:
            cost += 40 # Bridge penalty

        # Slightly prefer straight lines
        if a[0] != b[0] and a[1] != b[1]:
            cost *= 1.01 # Penalize diagonal moves slightly

        return cost

    def _reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

    def _define_city_blocks(self):
        """Detect city blocks as polygons enclosed by roads and bridges."""
        all_lines = []
        for road in self.roads:
            if len(road['points']) > 1:
                all_lines.append(LineString(road['points']))
        for bridge in self.bridges:
            if len(bridge['points']) > 1:
                all_lines.append(LineString(bridge['points']))

        if not all_lines:
            print("No roads or bridges to form city blocks.")
            return

        # Use polygonize to find closed loops in the road network
        polygons = list(polygonize(all_lines))
        print(f"✓ {len(polygons)} potential polygons found.")

        district_centers = {d['type']: Point(d['center']) for d in self.districts}
        
        # Clear previous blocks and repopulate with valid ones
        self.blocks = []
        for poly in polygons:
            # CRITICAL: Ensure we only process valid Polygon objects
            if isinstance(poly, Polygon) and poly.is_valid and poly.area > 0:
                # Assign a district type based on which district center it contains
                found_type = 'unclassified'
                for dtype, center_point in district_centers.items():
                    if poly.contains(center_point):
                        found_type = dtype
                        break
                self.blocks.append({'polygon': poly, 'type': found_type})
        
        print(f"✓ {len(self.blocks)} valid city blocks defined.")
        
    def _generate_parks(self):
        """Generate parks and green spaces throughout the map"""
        # Find large blocks for potential parks (larger blocks are better for parks)
        park_candidates = [b for b in self.blocks if b['polygon'].area > 5000]
        
        # If not enough large blocks, just pick from all blocks
        if len(park_candidates) < 5:
            park_candidates = self.blocks
            
        # Convert some blocks to parks (randomly select 5-10% of blocks)
        num_parks = max(3, int(len(park_candidates) * random.uniform(0.05, 0.1)))
        
        # Sort by area to prefer larger blocks for parks
        park_candidates.sort(key=lambda b: b['polygon'].area, reverse=True)
        
        # Select some blocks to convert to parks (biased toward larger blocks)
        park_blocks = []
        
        # 50% chance of selecting from the top 20% of blocks (largest ones)
        top_tier = int(len(park_candidates) * 0.2) + 1
        for _ in range(num_parks // 2):
            if park_candidates:
                idx = random.randint(0, min(top_tier, len(park_candidates)) - 1)
                park_blocks.append(park_candidates.pop(idx))
        
        # Rest are selected from remaining blocks
        for _ in range(num_parks - len(park_blocks)):
            if park_candidates:
                idx = random.randint(0, len(park_candidates) - 1)
                park_blocks.append(park_candidates.pop(idx))
        
        # Convert selected blocks to parks
        for block in park_blocks:
            polygon = block['polygon']
            # Create a park with an irregular shape by slightly altering the polygon
            park_poly = self._create_irregular_shape(polygon)
            
            # Generate internal park features (paths, trees, lakes, etc)
            park_features = self._generate_park_features(park_poly)
            
            # Add to parks list
            self.parks.append({
                'polygon': park_poly,
                'features': park_features,
                'type': 'park',
                'name': self._generate_park_name()
            })
            
            # Remove any blocks that became parks from the regular blocks list
            self.blocks = [b for b in self.blocks if b != block]
    
    def _create_irregular_shape(self, polygon):
        """Create a more natural, irregular shape from a polygon"""
        # For now, just return the original polygon
        # In a more advanced version, we could use noise to make the edges more irregular
        return polygon
    
    def _generate_park_features(self, polygon):
        """Generate internal features for a park"""
        # Simple placeholder for now
        features = {
            'paths': [],
            'trees': [],
            'water_features': [],
            'amenities': []
        }
        
        # Extract the bounds for convenience
        min_x, min_y, max_x, max_y = polygon.bounds
        
        # Generate some random tree clusters
        num_trees = int(polygon.area / 5000)  # One tree per 5000 area units
        for _ in range(num_trees):
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            point = Point(x, y)
            if polygon.contains(point):
                features['trees'].append((x, y))
                
        # Maybe add a small lake if the park is large enough
        if polygon.area > 20000 and random.random() < 0.4:
            # Find a good spot for a lake
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            features['water_features'].append({
                'type': 'lake', 
                'center': (center_x, center_y),
                'radius': min(max_x - min_x, max_y - min_y) * 0.15
            })
            
        # Add some paths through the park
        # For now just a simple grid; could be more organic in future versions
        return features
        
    def _generate_park_name(self):
        """Generate a plausible name for a park"""
        prefixes = ["Central", "Memorial", "Riverside", "Lakeside", "Golden", "Green", 
                  "Sunset", "Pine", "Oak", "Maple", "Spring", "Summer", "Autumn", "Winter"]
        suffixes = ["Park", "Gardens", "Reserve", "Common", "Fields", "Meadows", "Woods"]
        
        return f"{random.choice(prefixes)} {random.choice(suffixes)}"
    
    def _generate_pois(self):
        """Generate Points of Interest across the map"""
        # Find suitable locations for different POI types
        for poi_type, props in self.poi_types.items():
            # How many of this type to generate
            num_to_generate = 1 if poi_type in ['airport', 'port'] else random.randint(1, 3)
            
            # Find locations that match the requirements
            candidates = self._find_poi_locations(poi_type, props, num_to_generate)
            
            # Generate the POIs at these locations
            for location in candidates:
                self._create_poi(poi_type, props, location)
    
    def _find_poi_locations(self, poi_type, props, count):
        """Find suitable locations for a specific POI type"""
        suitable_locations = []
        width, height = props['size']
        
        # Existing POI locations for distance checking
        existing_pois = [(l['x'], l['y']) for l in self.landmarks]
        
        # Different search strategies based on POI type
        if poi_type == 'airport':
            # Airports need large flat areas away from the center
            gradient_x, gradient_y = np.gradient(self.heightmap)
            slope = np.sqrt(gradient_x**2 + gradient_y**2)
            flat_areas = np.argwhere((self.land_mask) & (slope < 0.01))
            
            # Convert to map coordinates and filter by distance from center
            center_x, center_y = self.width/2, self.height/2
            for point in flat_areas:
                y, x = point
                map_x, map_y = x * self.grid_size, y * self.grid_size
                
                # Distance check - prefer areas away from center
                dist_from_center = ((map_x - center_x)**2 + (map_y - center_y)**2)**0.5
                if dist_from_center > min(self.width, self.height) * 0.3:
                    # Check if area around this point is large enough
                    has_space = True
                    for dx in range(-width//2, width//2):
                        for dy in range(-height//2, height//2):
                            check_x, check_y = x + dx//self.grid_size, y + dy//self.grid_size
                            if not (0 <= check_x < self.grid_width and 0 <= check_y < self.grid_height) or not self.land_mask[check_y, check_x]:
                                has_space = False
                                break
                                
                    if has_space:
                        suitable_locations.append((map_x, map_y))
                        
        elif poi_type == 'port':
            # Ports need to be near water
            # Find land areas adjacent to water
            coastal_points = []
            for y in range(1, self.grid_height-1):
                for x in range(1, self.grid_width-1):
                    if self.land_mask[y, x]:
                        # Check if any neighbor is water
                        has_water_neighbor = False
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                if not self.land_mask[y+dy, x+dx]:
                                    has_water_neighbor = True
                                    break
                                    
                        if has_water_neighbor:
                            coastal_points.append((x * self.grid_size, y * self.grid_size))
                            
            random.shuffle(coastal_points)
            suitable_locations = coastal_points[:min(10, len(coastal_points))]
            
        else:
            # Other POIs can be in urban areas
            # Find blocks that match the district type preference
            district_preference = {'stadium': 'downtown', 'hospital': 'residential',
                                  'university': 'residential', 'mall': 'commercial'}
                                  
            preferred_type = district_preference.get(poi_type, None)
            
            # Find matching blocks or just use any blocks if no preference
            candidate_blocks = []
            for block in self.blocks:
                if not preferred_type or block['type'] == preferred_type:
                    if block['polygon'].area > width * height * 1.2:  # Ensure block is big enough
                        candidate_blocks.append(block)
                        
            # If not enough preferred blocks, add others
            if len(candidate_blocks) < count:
                additional = [b for b in self.blocks if b['polygon'].area > width * height * 1.2 
                             and b not in candidate_blocks]
                candidate_blocks.extend(additional)
                
            # Extract centroids from blocks
            for block in candidate_blocks:
                centroid = block['polygon'].centroid
                suitable_locations.append((centroid.x, centroid.y))
                
        # Filter by minimum distance between POIs
        filtered_locations = []
        min_distance = props['buffer']
        
        for loc in suitable_locations:
            too_close = False
            for existing in existing_pois + filtered_locations:
                dist = ((loc[0] - existing[0])**2 + (loc[1] - existing[1])**2)**0.5
                if dist < min_distance:
                    too_close = True
                    break
                    
            if not too_close:
                filtered_locations.append(loc)
                if len(filtered_locations) >= count:
                    break
                    
        return filtered_locations
    
    def _create_poi(self, poi_type, props, location):
        """Create a point of interest at the specified location"""
        x, y = location
        width, height = props['size']
        
        # Add some randomness to dimensions
        width *= random.uniform(0.8, 1.2)
        height *= random.uniform(0.8, 1.2)
        
        # Create the POI
        poi = {
            'type': poi_type,
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'color': props['color'],
            'name': self._generate_poi_name(poi_type)
        }
        
        # Airport needs a runway
        if poi_type == 'airport':
            runway_length = width * 0.8
            runway_width = height * 0.2
            # Randomly orient the runway
            angle = random.uniform(0, math.pi)
            poi['runway'] = {
                'length': runway_length,
                'width': runway_width,
                'angle': angle
            }
            
        self.landmarks.append(poi)
        
    def _generate_poi_name(self, poi_type):
        """Generate a name for the POI based on its type"""
        if poi_type == 'airport':
            prefixes = ["International", "Municipal", "Regional", "Metropolitan"]
            return f"{random.choice(['Liberty', 'Franklin', 'Washington', 'Jefferson', 'Lincoln', 'Kennedy'])} {random.choice(prefixes)} Airport"
        
        elif poi_type == 'stadium':
            sponsors = ["Arena", "Stadium", "Coliseum", "Center", "Field"]
            return f"{random.choice(['Victory', 'Champions', 'Unity', 'Liberty', 'Patriot'])} {random.choice(sponsors)}"
        
        elif poi_type == 'hospital':
            types = ["General", "Memorial", "Community", "Regional"]
            return f"{random.choice(['Mercy', 'Saint Mary', 'Providence', 'Unity', 'Hope'])} {random.choice(types)} Hospital"
        
        elif poi_type == 'university':
            return f"University of {random.choice(['Liberty', 'Franklin', 'Washington', 'Jefferson', 'Springfield'])}"
        
        elif poi_type == 'mall':
            return f"{random.choice(['Grand', 'Central', 'Plaza', 'Metro', 'Royal'])} {random.choice(['Mall', 'Shopping Center', 'Galleria'])}"
        
        elif poi_type == 'port':
            return f"{random.choice(['Harbor', 'Port', 'Marina', 'Docks'])} of {random.choice(['Liberty', 'Franklin', 'Springfield', 'Bayview'])}"
        
        return f"{poi_type.capitalize()} #{random.randint(1, 99)}"

    def _populate_city_blocks(self):
        """Populate city blocks with buildings, aligned to the block's edges."""
        for block in self.blocks:
            polygon = block['polygon']
            min_x, min_y, max_x, max_y = polygon.bounds
            
            # Adjust density based on block type
            density_map = {'downtown': 0.001, 'commercial': 0.0008, 'residential': 0.0005, 'industrial': 0.0004}
            density = density_map.get(block['type'], 0.0003)
            num_buildings = int(polygon.area * density)

            for _ in range(num_buildings):
                # Generate random points until one is inside the polygon
                while True:
                    px = random.uniform(min_x, max_x)
                    py = random.uniform(min_y, max_y)
                    point = Point(px, py)
                    if polygon.contains(point):
                        # Simple placement for now, alignment will be next
                        grid_x, grid_y = int(px / self.grid_size), int(py / self.grid_size)
                        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height and self.land_mask[grid_y, grid_x]:
                            self.buildings.append({
                                'x': px, 'y': py,
                                'width': random.uniform(10, 30), 'length': random.uniform(10, 30),
                                'type': block['type']
                            })
                        break

    def get_map_data(self):
        return {
            'width': self.width,
            'height': self.height,
            'blocks': self.blocks,
            'roads': self.roads,
            'bridges': self.bridges,
            'buildings': self.buildings,
            'water_bodies': self.water_bodies,
            'parks': self.parks,
            'landmarks': self.landmarks
        }

    def _prepare_for_json(self, data):
        """Recursively convert Shapely objects to JSON-serializable formats."""
        if isinstance(data, dict):
            return {k: self._prepare_for_json(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._prepare_for_json(i) for i in data]
        if isinstance(data, tuple):
            return [self._prepare_for_json(i) for i in data]
        if isinstance(data, Polygon):
            return {'type': 'Polygon', 'exterior': list(data.exterior.coords)}
        if isinstance(data, LineString):
            return {'type': 'LineString', 'coords': list(data.coords)}
        if isinstance(data, Point):
            return {'type': 'Point', 'coords': list(data.coords)}
        # Convert any NumPy scalar types to their native Python equivalents
        if isinstance(data, np.generic):
            return data.item()
        # Convert NumPy arrays to lists
        if isinstance(data, np.ndarray):
            return data.tolist()
        return data

    def save_map(self, filename):
        map_data = self.get_map_data()
        json_compatible_data = self._prepare_for_json(map_data)
        with open(filename, 'w') as f:
            json.dump(json_compatible_data, f, cls=NumpyJSONEncoder, indent=2)
        print(f"Map saved to {filename}")

    def visualize_map(self, save_path=None):
        fig, ax = plt.subplots(1, 1, figsize=(16, 16))
        ax.set_facecolor('#4a90e2') # Water color

        # Draw terrain with heightmap-based coloring
        cmap = mcolors.LinearSegmentedColormap.from_list('terrain', ['#a67c52', '#348c31', '#2d5016', '#6b6b6b'])
        ax.imshow(self.heightmap, cmap=cmap, extent=(0, self.width, self.height, 0), vmin=self.water_level, vmax=1.0)
        ax.imshow(np.invert(self.land_mask), cmap='Blues_r', extent=(0, self.width, self.height, 0), alpha=0.5, vmin=0, vmax=1)
        
        # District color scheme with better visual distinction
        district_colors = {
            'residential': '#FFEDA0',  # Pale yellow
            'commercial': '#FA9FB5',  # Pink
            'industrial': '#BCBDDC',  # Light purple
            'downtown': '#FD8D3C',    # Orange
            'suburban': '#EDF8B1',    # Very light yellow
            'beach': '#FFFFCC',       # Pale sand
            'airport': '#C7E9B4',     # Light green-blue
            'hills': '#74C476',       # Medium green
            'port': '#6BAED6',        # Light blue
            'park': '#41AB5D'         # Green
        }
        
        # Draw district boundaries and labels
        district_patches = []
        district_labels = []
        
        for district in self.districts:
            if 'polygon' in district and district['polygon'] is not None and district['polygon'].is_valid:
                color = district_colors.get(district['type'], 'white')
                # Use matplotlib's Polygon for rendering
                coords = list(district['polygon'].exterior.coords)
                patch = MatplotlibPolygon(coords, alpha=0.5, fc=color, ec='#555555', lw=1.5, zorder=1)
                ax.add_patch(patch)
                
                # Add district label at center point
                if 'center' in district:
                    district_labels.append({
                        'pos': district['center'],
                        'text': district['type'].upper(),
                        'color': '#333333',
                        'size': 14 if district['type'] in ['downtown', 'airport', 'port'] else 12
                    })

        # Draw city blocks with subtle fill
        for block in self.blocks:
            if 'polygon' in block and block['polygon'] is not None and block['polygon'].is_valid:
                coords = list(block['polygon'].exterior.coords)
                ax.fill(coords[:, 0], coords[:, 1], color='#DDDDDD', alpha=0.2, zorder=3)

        # Draw roads with clear hierarchy based on type
        # First draw the outlines for all roads (black base)
        for road in self.roads:
            points = np.array(road['points'])
            width = road.get('width', 5)
            road_type = road.get('type', 'local')
            
            # Draw thicker black outline for all roads
            ax.plot(points[:, 0], points[:, 1], color='#000000', 
                   linewidth=width+2, solid_capstyle='round', 
                   alpha=0.7, zorder=4)
        
        # Then draw the colored road surfaces
        for road in self.roads:
            points = np.array(road['points'])
            width = road.get('width', 5)
            road_type = road.get('type', 'local')
            
            if road_type == 'highway':
                # Highways - yellow center line
                ax.plot(points[:, 0], points[:, 1], color='#F0F0F0', 
                       linewidth=width, solid_capstyle='round', 
                       alpha=0.9, zorder=5)
                ax.plot(points[:, 0], points[:, 1], color='#FFDD00', 
                       linewidth=width*0.3, solid_capstyle='round', 
                       alpha=0.9, zorder=6)
            elif road_type == 'arterial':
                # Arterial - wider white roads
                ax.plot(points[:, 0], points[:, 1], color='#FFFFFF', 
                       linewidth=width, solid_capstyle='round', 
                       alpha=0.9, zorder=5)
            elif road_type == 'collector':
                # Collector roads - light gray
                ax.plot(points[:, 0], points[:, 1], color='#E0E0E0', 
                       linewidth=width, solid_capstyle='round', 
                       alpha=0.8, zorder=5)
            else:  # local roads
                ax.plot(points[:, 0], points[:, 1], color='#D0D0D0', 
                       linewidth=width, solid_capstyle='round', 
                       alpha=0.7, zorder=5)

        # Draw bridges with elevated appearance
        for bridge in self.bridges:
            points = np.array(bridge['points'])
            width = bridge.get('width', 8)
            # Shadow/structure
            ax.plot(points[:, 0], points[:, 1], color='#404040', 
                   linewidth=width+4, solid_capstyle='round', 
                   alpha=0.7, zorder=6)
            # Road surface
            ax.plot(points[:, 0], points[:, 1], color='#909090', 
                   linewidth=width, solid_capstyle='round', 
                   alpha=0.9, zorder=7)
            # Center line
            ax.plot(points[:, 0], points[:, 1], color='#FFFF00', 
                   linewidth=width*0.2, solid_capstyle='round', 
                   alpha=0.7, zorder=8, linestyle=(0, (5, 10)))

        # Draw buildings with 3D-like appearance
        for building in self.buildings:
            # Buildings are stored as rectangles with x, y, width, length
            if 'x' in building and 'y' in building and 'width' in building and 'length' in building:
                x = building['x']
                y = building['y']
                width = building['width']
                length = building['length']
                
                # Create rectangle points
                rect_points = np.array([
                    [x, y],
                    [x + width, y],
                    [x + width, y + length],
                    [x, y + length]
                ])
                
                building_type = building.get('type', 'generic')
                district_type = building.get('district_type', 'residential')
                
                # Color buildings by their district type for visual coherence
                if district_type == 'downtown':
                    color = '#A0A0A0'  # Gray for downtown skyscrapers
                elif district_type == 'commercial':
                    color = '#C0C0C0'  # Light gray for commercial buildings
                elif district_type == 'industrial':
                    color = '#909090'  # Dark gray for industrial
                elif district_type == 'airport':
                    color = '#B0B0B0'  # Medium gray for airport buildings
                else:
                    color = '#D0D0D0'  # Light gray for other buildings
                
                # Add shadow effect for buildings
                shadow_offset = 2  # Pixels to offset shadow
                shadow_points = rect_points + np.array([shadow_offset, -shadow_offset])
                ax.fill(shadow_points[:, 0], shadow_points[:, 1], color='black', alpha=0.2, zorder=9)
                
                # Draw building
                ax.fill(rect_points[:, 0], rect_points[:, 1], color=color, alpha=0.9, zorder=10)

        # Draw district labels with better typography
        for label in district_labels:
            ax.text(label['pos'][0], label['pos'][1], label['text'], 
                    fontsize=label['size'], ha='center', va='center', 
                    color=label['color'], weight='bold', zorder=11,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

        # Set map limits
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        
        # Add title and remove axes
        ax.set_title("GTA-Style Procedural Map", fontsize=20, pad=20, fontweight='bold')
        ax.axis('off')
        
        # Add legend with clearer organization and grouping
        district_elements = [
            Patch(facecolor=district_colors['downtown'], alpha=0.5, edgecolor='#555555', label='Downtown'),
            Patch(facecolor=district_colors['commercial'], alpha=0.5, edgecolor='#555555', label='Commercial'),
            Patch(facecolor=district_colors['residential'], alpha=0.5, edgecolor='#555555', label='Residential'),
            Patch(facecolor=district_colors['industrial'], alpha=0.5, edgecolor='#555555', label='Industrial'),
            Patch(facecolor=district_colors['suburban'], alpha=0.5, edgecolor='#555555', label='Suburban')
        ]
        
        special_district_elements = [
            Patch(facecolor=district_colors['beach'], alpha=0.5, edgecolor='#555555', label='Beach'),
            Patch(facecolor=district_colors['airport'], alpha=0.5, edgecolor='#555555', label='Airport'),
            Patch(facecolor=district_colors['hills'], alpha=0.5, edgecolor='#555555', label='Hills'),
            Patch(facecolor=district_colors['port'], alpha=0.5, edgecolor='#555555', label='Port'),
            Patch(facecolor=district_colors['park'], alpha=0.5, edgecolor='#555555', label='Park')
        ]
        
        road_elements = [
            Line2D([0], [0], color='#000000', linewidth=4, label='Road Outline'),
            Line2D([0], [0], color='#FFDD00', linewidth=2, label='Highway'),
            Line2D([0], [0], color='#FFFFFF', linewidth=2, label='Arterial'),
            Line2D([0], [0], color='#E0E0E0', linewidth=2, label='Collector'),
            Line2D([0], [0], color='#909090', linewidth=2, label='Bridge')
        ]
        
        nature_elements = [
            Patch(facecolor='#4682B4', alpha=0.7, label='Water'),
            # Use a gradient patch for terrain
            Patch(facecolor='#74C476', alpha=0.6, label='Lower Elevation'),
            Patch(facecolor='#6B8E23', alpha=0.6, label='Higher Elevation')
        ]
        
        # Create four separate legends for different categories
        # Urban Districts
        ax.legend(handles=district_elements, loc='upper left', 
                 title='Urban Districts', framealpha=0.9, title_fontsize=12)
        
        # Special Districts - positioned below urban districts
        special_legend = ax.legend(handles=special_district_elements, loc='upper left', 
                                  title='Special Areas', framealpha=0.9, title_fontsize=12,
                                  bbox_to_anchor=(0.0, 0.75))
        ax.add_artist(special_legend)
        
        # Roads - positioned on right side
        road_legend = ax.legend(handles=road_elements, loc='upper right', 
                               title='Transportation', framealpha=0.9, title_fontsize=12)
        ax.add_artist(road_legend)
        
        # Natural features - positioned below roads
        nature_legend = ax.legend(handles=nature_elements, loc='upper right', 
                                 title='Natural Features', framealpha=0.9, title_fontsize=12,
                                 bbox_to_anchor=(1.0, 0.75))
        ax.add_artist(nature_legend)
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

if __name__ == "__main__":
    generator = GTAMapGenerator(width=2000, height=2000, seed=42)
    generator.generate_complete_map()
    generator.visualize_map('gta_map_improved.png')
    generator.save_map('gta_map_improved.json')

