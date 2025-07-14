import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import random
import math
from noise import pnoise2
from scipy.spatial import Voronoi
import json

class GTAMapGenerator:
    def __init__(self, width=2000, height=2000, seed=None):
        """
        Initialize the GTA-style map generator
        
        Args:
            width: Map width in units
            height: Map height in units
            seed: Random seed for reproducible generation
        """
        self.width = width
        self.height = height
        self.seed = seed or random.randint(0, 10000)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Map data structures
        self.roads = []
        self.buildings = []
        self.districts = []
        self.water_bodies = []
        self.parks = []
        self.landmarks = []
        
        # Generation parameters
        self.road_density = 0.15
        self.building_density = 0.6
        self.district_count = 8
        
    def generate_terrain(self):
        """Generate base terrain using Perlin noise"""
        terrain = np.zeros((self.height // 10, self.width // 10))
        
        for i in range(terrain.shape[0]):
            for j in range(terrain.shape[1]):
                # Multiple octaves of noise for realistic terrain
                elevation = (
                    pnoise2(i * 0.01, j * 0.01, octaves=4, persistence=0.5, lacunarity=2.0, base=self.seed) * 0.5 +
                    pnoise2(i * 0.05, j * 0.05, octaves=2, persistence=0.3, lacunarity=2.0, base=self.seed + 1) * 0.3 +
                    pnoise2(i * 0.1, j * 0.1, octaves=1, persistence=0.2, lacunarity=2.0, base=self.seed + 2) * 0.2
                )
                terrain[i, j] = elevation
        
        return terrain
    
    def generate_districts(self):
        """Generate city districts using Voronoi diagrams"""
        # Generate district centers
        centers = []
        for _ in range(self.district_count):
            x = random.uniform(self.width * 0.1, self.width * 0.9)
            y = random.uniform(self.height * 0.1, self.height * 0.9)
            centers.append([x, y])
        
        # Create Voronoi diagram
        vor = Voronoi(centers)
        
        district_types = ['downtown', 'residential', 'industrial', 'commercial', 
                         'suburban', 'beach', 'hills', 'airport']
        
        for i, center in enumerate(centers):
            district_type = district_types[i % len(district_types)]
            self.districts.append({
                'center': center,
                'type': district_type,
                'radius': random.uniform(200, 400)
            })
    
    def generate_road_network(self):
        """Generate a realistic road network"""
        # Main highways (ring roads and major arteries)
        self.generate_highways()
        
        # Secondary roads
        self.generate_secondary_roads()
        
        # Local streets within districts
        self.generate_local_streets()
    
    def generate_highways(self):
        """Generate major highways and ring roads"""
        # Outer ring road
        center_x, center_y = self.width // 2, self.height // 2
        radius = min(self.width, self.height) * 0.4
        
        ring_points = []
        for angle in np.linspace(0, 2 * np.pi, 32):
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            ring_points.append([x, y])
        
        self.roads.append({
            'points': ring_points,
            'type': 'highway',
            'width': 20
        })
        
        # Major arteries (cross-city highways)
        arteries = [
            [[0, center_y], [self.width, center_y]],  # Horizontal
            [[center_x, 0], [center_x, self.height]],  # Vertical
            [[0, 0], [self.width, self.height]],  # Diagonal
            [[0, self.height], [self.width, 0]]  # Diagonal
        ]
        
        for artery in arteries:
            self.roads.append({
                'points': artery,
                'type': 'highway',
                'width': 15
            })
    
    def generate_secondary_roads(self):
        """Generate secondary roads connecting districts"""
        for i, district1 in enumerate(self.districts):
            for j, district2 in enumerate(self.districts[i+1:], i+1):
                if random.random() < 0.7:  # 70% chance of connection
                    # Create curved road between districts
                    start = district1['center']
                    end = district2['center']
                    
                    # Add some curvature
                    mid_x = (start[0] + end[0]) / 2 + random.uniform(-100, 100)
                    mid_y = (start[1] + end[1]) / 2 + random.uniform(-100, 100)
                    
                    road_points = self.create_curved_road(start, [mid_x, mid_y], end)
                    
                    self.roads.append({
                        'points': road_points,
                        'type': 'secondary',
                        'width': 10
                    })
    
    def generate_local_streets(self):
        """Generate local street grids within districts"""
        for district in self.districts:
            center = district['center']
            radius = district['radius']
            district_type = district['type']
            
            if district_type in ['downtown', 'commercial']:
                self.generate_grid_streets(center, radius, spacing=50)
            elif district_type in ['residential', 'suburban']:
                self.generate_organic_streets(center, radius)
            elif district_type == 'industrial':
                self.generate_industrial_streets(center, radius)
    
    def generate_grid_streets(self, center, radius, spacing=50):
        """Generate grid-based street layout"""
        cx, cy = center
        
        # Vertical streets
        for x in range(int(cx - radius), int(cx + radius), spacing):
            if abs(x - cx) < radius:
                y_range = math.sqrt(radius**2 - (x - cx)**2)
                self.roads.append({
                    'points': [[x, cy - y_range], [x, cy + y_range]],
                    'type': 'local',
                    'width': 6
                })
        
        # Horizontal streets
        for y in range(int(cy - radius), int(cy + radius), spacing):
            if abs(y - cy) < radius:
                x_range = math.sqrt(radius**2 - (y - cy)**2)
                self.roads.append({
                    'points': [[cx - x_range, y], [cx + x_range, y]],
                    'type': 'local',
                    'width': 6
                })
    
    def generate_organic_streets(self, center, radius):
        """Generate organic, curved streets for residential areas"""
        cx, cy = center
        
        # Create several curved streets radiating from center
        for angle in np.linspace(0, 2 * np.pi, 8):
            start_x = cx + 20 * np.cos(angle)
            start_y = cy + 20 * np.sin(angle)
            end_x = cx + radius * 0.8 * np.cos(angle)
            end_y = cy + radius * 0.8 * np.sin(angle)
            
            # Add curves and branches
            points = [[start_x, start_y]]
            current_x, current_y = start_x, start_y
            
            steps = 10
            for i in range(steps):
                t = i / steps
                target_x = start_x + t * (end_x - start_x)
                target_y = start_y + t * (end_y - start_y)
                
                # Add some randomness
                noise_x = random.uniform(-15, 15)
                noise_y = random.uniform(-15, 15)
                
                current_x = target_x + noise_x
                current_y = target_y + noise_y
                points.append([current_x, current_y])
            
            self.roads.append({
                'points': points,
                'type': 'local',
                'width': 5
            })
    
    def generate_industrial_streets(self, center, radius):
        """Generate wide, straight streets for industrial areas"""
        cx, cy = center
        
        # Large grid with wider spacing
        spacing = 80
        for x in range(int(cx - radius), int(cx + radius), spacing):
            if abs(x - cx) < radius:
                y_range = math.sqrt(radius**2 - (x - cx)**2)
                self.roads.append({
                    'points': [[x, cy - y_range], [x, cy + y_range]],
                    'type': 'industrial',
                    'width': 8
                })
        
        for y in range(int(cy - radius), int(cy + radius), spacing):
            if abs(y - cy) < radius:
                x_range = math.sqrt(radius**2 - (y - cy)**2)
                self.roads.append({
                    'points': [[cx - x_range, y], [cx + x_range, y]],
                    'type': 'industrial',
                    'width': 8
                })
    
    def create_curved_road(self, start, mid, end, segments=20):
        """Create a curved road using quadratic Bezier curve"""
        points = []
        for t in np.linspace(0, 1, segments):
            x = (1-t)**2 * start[0] + 2*(1-t)*t * mid[0] + t**2 * end[0]
            y = (1-t)**2 * start[1] + 2*(1-t)*t * mid[1] + t**2 * end[1]
            points.append([x, y])
        return points
    
    def generate_buildings(self):
        """Generate buildings based on district types and road proximity"""
        for district in self.districts:
            center = district['center']
            radius = district['radius']
            district_type = district['type']
            
            # Building parameters based on district type
            if district_type == 'downtown':
                building_count = 200
                height_range = (50, 300)
                size_range = (20, 60)
            elif district_type == 'commercial':
                building_count = 150
                height_range = (20, 100)
                size_range = (30, 80)
            elif district_type == 'residential':
                building_count = 300
                height_range = (10, 40)
                size_range = (15, 35)
            elif district_type == 'industrial':
                building_count = 100
                height_range = (15, 50)
                size_range = (40, 120)
            else:
                building_count = 80
                height_range = (10, 30)
                size_range = (20, 50)
            
            # Generate buildings
            for _ in range(building_count):
                # Random position within district
                angle = random.uniform(0, 2 * np.pi)
                distance = random.uniform(0, radius * 0.9)
                
                x = center[0] + distance * np.cos(angle)
                y = center[1] + distance * np.sin(angle)
                
                # Check if position is valid (not on roads)
                if self.is_valid_building_position(x, y):
                    width = random.uniform(*size_range)
                    length = random.uniform(*size_range)
                    height = random.uniform(*height_range)
                    
                    self.buildings.append({
                        'x': x,
                        'y': y,
                        'width': width,
                        'length': length,
                        'height': height,
                        'type': district_type,
                        'rotation': random.uniform(0, 360)
                    })
    
    def is_valid_building_position(self, x, y, min_distance=15):
        """Check if a position is valid for building placement"""
        for road in self.roads:
            for i in range(len(road['points']) - 1):
                p1 = road['points'][i]
                p2 = road['points'][i + 1]
                
                # Calculate distance from point to line segment
                distance = self.point_to_line_distance([x, y], p1, p2)
                if distance < min_distance + road['width'] / 2:
                    return False
        return True
    
    def point_to_line_distance(self, point, line_start, line_end):
        """Calculate distance from point to line segment"""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        A = x0 - x1
        B = y0 - y1
        C = x2 - x1
        D = y2 - y1
        
        dot = A * C + B * D
        len_sq = C * C + D * D
        
        if len_sq == 0:
            return math.sqrt(A * A + B * B)
        
        param = dot / len_sq
        
        if param < 0:
            xx, yy = x1, y1
        elif param > 1:
            xx, yy = x2, y2
        else:
            xx = x1 + param * C
            yy = y1 + param * D
        
        dx = x0 - xx
        dy = y0 - yy
        return math.sqrt(dx * dx + dy * dy)
    
    def generate_landmarks(self):
        """Generate special landmarks and points of interest"""
        landmark_types = [
            {'name': 'Airport', 'size': 300, 'type': 'airport'},
            {'name': 'Stadium', 'size': 150, 'type': 'stadium'},
            {'name': 'Mall', 'size': 200, 'type': 'mall'},
            {'name': 'Hospital', 'size': 100, 'type': 'hospital'},
            {'name': 'University', 'size': 180, 'type': 'university'},
            {'name': 'Port', 'size': 250, 'type': 'port'}
        ]
        
        for landmark in landmark_types:
            # Find suitable location
            attempts = 0
            while attempts < 50:
                x = random.uniform(landmark['size'], self.width - landmark['size'])
                y = random.uniform(landmark['size'], self.height - landmark['size'])
                
                if self.is_valid_landmark_position(x, y, landmark['size']):
                    self.landmarks.append({
                        'name': landmark['name'],
                        'x': x,
                        'y': y,
                        'size': landmark['size'],
                        'type': landmark['type']
                    })
                    break
                attempts += 1
    
    def is_valid_landmark_position(self, x, y, size):
        """Check if position is valid for landmark placement"""
        # Check distance from other landmarks
        for landmark in self.landmarks:
            distance = math.sqrt((x - landmark['x'])**2 + (y - landmark['y'])**2)
            if distance < size + landmark['size']:
                return False
        return True
    
    def generate_water_bodies(self):
        """Generate rivers, lakes, and coastal areas"""
        # Generate a river
        river_points = []
        start_x = random.uniform(0, self.width)
        start_y = 0
        
        current_x, current_y = start_x, start_y
        
        while current_y < self.height:
            # River flows generally downward with some meandering
            next_x = current_x + random.uniform(-30, 30)
            next_y = current_y + random.uniform(20, 50)
            
            # Keep within bounds
            next_x = max(50, min(self.width - 50, next_x))
            
            river_points.append([next_x, next_y])
            current_x, current_y = next_x, next_y
        
        self.water_bodies.append({
            'type': 'river',
            'points': river_points,
            'width': 40
        })
        
        # Generate lakes
        for _ in range(3):
            lake_x = random.uniform(100, self.width - 100)
            lake_y = random.uniform(100, self.height - 100)
            lake_radius = random.uniform(50, 120)
            
            self.water_bodies.append({
                'type': 'lake',
                'center': [lake_x, lake_y],
                'radius': lake_radius
            })
    
    def generate_parks(self):
        """Generate parks and green spaces"""
        for district in self.districts:
            if district['type'] in ['residential', 'suburban', 'downtown']:
                # Add parks within districts
                for _ in range(random.randint(1, 3)):
                    angle = random.uniform(0, 2 * np.pi)
                    distance = random.uniform(0, district['radius'] * 0.7)
                    
                    park_x = district['center'][0] + distance * np.cos(angle)
                    park_y = district['center'][1] + distance * np.sin(angle)
                    park_size = random.uniform(30, 80)
                    
                    if self.is_valid_building_position(park_x, park_y, park_size):
                        self.parks.append({
                            'x': park_x,
                            'y': park_y,
                            'size': park_size,
                            'type': 'park'
                        })
    
    def generate_complete_map(self):
        """Generate the complete map with all features"""
        print(f"Generating GTA-style map (seed: {self.seed})...")
        
        # Generate in order of dependency
        self.generate_districts()
        print("✓ Districts generated")
        
        self.generate_road_network()
        print("✓ Road network generated")
        
        self.generate_water_bodies()
        print("✓ Water bodies generated")
        
        self.generate_buildings()
        print("✓ Buildings generated")
        
        self.generate_landmarks()
        print("✓ Landmarks generated")
        
        self.generate_parks()
        print("✓ Parks generated")
        
        print("Map generation complete!")
        return self.get_map_data()
    
    def get_map_data(self):
        """Return all map data as a dictionary"""
        return {
            'width': self.width,
            'height': self.height,
            'seed': self.seed,
            'roads': self.roads,
            'buildings': self.buildings,
            'districts': self.districts,
            'water_bodies': self.water_bodies,
            'parks': self.parks,
            'landmarks': self.landmarks
        }
    
    def save_map(self, filename):
        """Save map data to JSON file"""
        map_data = self.get_map_data()
        with open(filename, 'w') as f:
            json.dump(map_data, f, indent=2)
        print(f"Map saved to {filename}")
    
    def visualize_map(self, save_path=None, show_labels=True):
        """Create a visual representation of the generated map"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 16))
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.set_facecolor('#2d5016')  # Dark green background
        
        # Draw water bodies first (background)
        for water in self.water_bodies:
            if water['type'] == 'river':
                points = np.array(water['points'])
                for i in range(len(points) - 1):
                    x_vals = [points[i][0], points[i+1][0]]
                    y_vals = [points[i][1], points[i+1][1]]
                    ax.plot(x_vals, y_vals, color='#4a90e2', linewidth=water['width'], alpha=0.8)
            elif water['type'] == 'lake':
                circle = plt.Circle(water['center'], water['radius'], color='#4a90e2', alpha=0.8)
                ax.add_patch(circle)
        
        # Draw districts (background zones)
        district_colors = {
            'downtown': '#ffeb3b',
            'residential': '#4caf50',
            'industrial': '#9e9e9e',
            'commercial': '#ff9800',
            'suburban': '#8bc34a',
            'beach': '#ffc107',
            'hills': '#795548',
            'airport': '#607d8b'
        }
        
        for district in self.districts:
            color = district_colors.get(district['type'], '#cccccc')
            circle = plt.Circle(district['center'], district['radius'], 
                              color=color, alpha=0.2, linewidth=2, fill=True)
            ax.add_patch(circle)
            
            if show_labels:
                ax.text(district['center'][0], district['center'][1], 
                       district['type'].title(), ha='center', va='center',
                       fontsize=10, fontweight='bold')
        
        # Draw parks
        for park in self.parks:
            circle = plt.Circle([park['x'], park['y']], park['size'], 
                              color='#4caf50', alpha=0.6)
            ax.add_patch(circle)
        
        # Draw roads
        road_colors = {
            'highway': '#333333',
            'secondary': '#555555',
            'local': '#777777',
            'industrial': '#666666'
        }
        
        for road in self.roads:
            color = road_colors.get(road['type'], '#888888')
            points = np.array(road['points'])
            
            if len(points) > 1:
                for i in range(len(points) - 1):
                    x_vals = [points[i][0], points[i+1][0]]
                    y_vals = [points[i][1], points[i+1][1]]
                    ax.plot(x_vals, y_vals, color=color, linewidth=road['width'], alpha=0.8)
        
        # Draw buildings
        building_colors = {
            'downtown': '#1976d2',
            'commercial': '#f57c00',
            'residential': '#388e3c',
            'industrial': '#616161',
            'suburban': '#689f38'
        }
        
        for building in self.buildings:
            color = building_colors.get(building['type'], '#757575')
            # Simple rectangle representation
            rect = Rectangle((building['x'] - building['width']/2, 
                            building['y'] - building['length']/2),
                           building['width'], building['length'],
                           color=color, alpha=0.7)
            ax.add_patch(rect)
        
        # Draw landmarks
        landmark_colors = {
            'airport': '#ff5722',
            'stadium': '#e91e63',
            'mall': '#9c27b0',
            'hospital': '#f44336',
            'university': '#3f51b5',
            'port': '#00bcd4'
        }
        
        for landmark in self.landmarks:
            color = landmark_colors.get(landmark['type'], '#ff9800')
            rect = Rectangle((landmark['x'] - landmark['size']/2, 
                            landmark['y'] - landmark['size']/2),
                           landmark['size'], landmark['size'],
                           color=color, alpha=0.8, linewidth=3)
            ax.add_patch(rect)
            
            if show_labels:
                ax.text(landmark['x'], landmark['y'], landmark['name'],
                       ha='center', va='center', fontsize=12, fontweight='bold',
                       color='white')
        
        ax.set_title(f'GTA-Style Generated Map (Seed: {self.seed})', 
                    fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel('X Coordinate', fontsize=14)
        ax.set_ylabel('Y Coordinate', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Map visualization saved to {save_path}")
        
        plt.show()

if __name__ == "__main__":
    # Example usage
    generator = GTAMapGenerator(width=2000, height=2000, seed=42)
    map_data = generator.generate_complete_map()
    
    # Save the map data
    generator.save_map('generated_map.json')
    
    # Visualize the map
    generator.visualize_map('gta_map.png')
