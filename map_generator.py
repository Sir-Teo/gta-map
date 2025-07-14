import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon as MatplotlibPolygon, Patch
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
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
        self.road_styles = {
            'arterial': {'color': '#333333', 'width': 8},
            'collector': {'color': '#555555', 'width': 5},
            'highway':  {'color': '#000000', 'width': 6},
            'local':    {'color': '#777777', 'width': 3},
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

    def generate_complete_map(self):
        print(f"Generating new map with seed: {self.seed}")
        self._generate_heightmap()
        print("✓ Heightmap generated")
        self._define_land_and_water()
        print("✓ Land and water defined")
        self._place_districts()
        print("✓ Districts placed")
        # Generate primary arterial grid before detailed road network
        self._generate_arterial_grid()
        print("✓ Arterial grid generated")
        self._generate_road_network()
        print("✓ Road network generated")
        self._define_city_blocks()
        print("✓ City blocks defined")
        self._populate_city_blocks()
        print("✓ Buildings populated")
        print("Map generation complete!")
        return self.get_map_data()

    def _generate_heightmap(self):
        scale = self.grid_width * 0.005
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                nx = x / self.grid_width - 0.5
                ny = y / self.grid_height - 0.5
                e = (1 * pnoise2(1 * nx * scale, 1 * ny * scale, octaves=4, persistence=0.5, lacunarity=2.0, base=self.seed) +
                     0.5 * pnoise2(2 * nx * scale, 2 * ny * scale, octaves=4, persistence=0.5, lacunarity=2.0, base=self.seed+1) +
                     0.25 * pnoise2(4 * nx * scale, 4 * ny * scale, octaves=4, persistence=0.5, lacunarity=2.0, base=self.seed+2))
                e /= (1 + 0.5 + 0.25)
                # Create a coastal falloff
                d = min(1, (math.sqrt(nx**2 + ny**2) / math.sqrt(0.5**2 + 0.5**2))**0.5)
                e = (e + 1) / 2 # map to 0-1
                self.heightmap[y, x] = e * (1 - d * 0.5) # lower edges to form island

    def _define_land_and_water(self):
        self.land_mask = self.heightmap > self.water_level

    def _place_districts(self):
        land_points = np.argwhere(self.land_mask)
        if len(land_points) < self.district_count:
            print("Warning: Not enough land to place districts.")
            return

        # Prioritize flatter areas for certain districts
        gradient_x, gradient_y = np.gradient(self.heightmap)
        slope = np.sqrt(gradient_x**2 + gradient_y**2)
        flat_areas = np.argwhere((self.land_mask) & (slope < 0.01))

        district_types = ['downtown', 'commercial', 'industrial', 'residential', 'suburban', 'hills', 'beach', 'port', 'airport', 'park']
        random.shuffle(district_types)

        centers = []
        for i in range(self.district_count):
            district_type = district_types[i % len(district_types)]
            point = None
            if district_type in ['downtown', 'commercial', 'airport'] and len(flat_areas) > 0:
                idx = np.random.choice(len(flat_areas))
                point = flat_areas[idx]
            elif district_type == 'beach':
                beach_points = np.argwhere((self.heightmap > self.water_level) & (self.heightmap < self.beach_level))
                if len(beach_points) > 0:
                    idx = np.random.choice(len(beach_points))
                    point = beach_points[idx]

            if point is None:
                idx = np.random.choice(len(land_points))
                point = land_points[idx]

            centers.append(point * self.grid_size)

        vor = Voronoi([c.tolist() for c in centers])
        for i, center in enumerate(centers):
            self.districts.append({
                'center': center,
                'type': district_types[i % len(district_types)],
                'vertices': [],
                'radius': random.uniform(150, 300)
            })

    def _generate_arterial_grid(self):
        """Generate an organic, distorted grid of arterial roads using Perlin noise."""
        noise_scale = 0.001
        noise_strength = 150

        # Vertical arterials
        x = 0
        while x <= self.width:
            pts = []
            for y_step in range(0, self.height + self.grid_size, self.grid_size):
                offset = pnoise2(x * noise_scale, y_step * noise_scale, octaves=2, persistence=0.5, lacunarity=2.0, base=self.seed + 10) * noise_strength
                pts.append((x + offset, y_step))
            self._add_polyline_as_road(pts, 'arterial')
            x += self.arterial_spacing

        # Horizontal arterials
        y = 0
        while y <= self.height:
            pts = []
            for x_step in range(0, self.width + self.grid_size, self.grid_size):
                offset = pnoise2(x_step * noise_scale, y * noise_scale, octaves=2, persistence=0.5, lacunarity=2.0, base=self.seed + 20) * noise_strength
                pts.append((x_step, y + offset))
            self._add_polyline_as_road(pts, 'arterial')
            y += self.arterial_spacing

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

        # Draw terrain
        cmap = LinearSegmentedColormap.from_list('terrain', ['#a67c52', '#348c31', '#2d5016', '#6b6b6b'])
        ax.imshow(self.heightmap, cmap=cmap, extent=(0, self.width, self.height, 0), vmin=self.water_level, vmax=1.0)
        ax.imshow(np.invert(self.land_mask), cmap='Blues_r', extent=(0, self.width, self.height, 0), alpha=0.5, vmin=0, vmax=1)

        # Draw roads
        for road in self.roads:
            points = np.array(road['points'])
            color = self.road_styles.get(road['type'], {}).get('color', '#444')
            ax.plot(points[:, 0], points[:, 1], color=color, linewidth=road['width'] / 1.5, alpha=0.9, solid_capstyle='round')

        # Draw bridges
        for bridge in self.bridges:
            points = np.array(bridge['points'])
            # Draw bridge casing
            ax.plot(points[:, 0], points[:, 1], color='#666', linewidth=bridge['width'] / 1.5 + 2, alpha=0.7, solid_capstyle='round')
            # Draw bridge surface
            ax.plot(points[:, 0], points[:, 1], color='#bbbbbb', linewidth=bridge['width'] / 1.5, alpha=0.9, solid_capstyle='round')

        # Draw city blocks
        block_patches = []
        for block in self.blocks:
            poly = block['polygon']
            if poly.is_valid and not poly.is_empty:
                # Use matplotlib's Polygon patch directly, which is more robust
                patch = MatplotlibPolygon(list(poly.exterior.coords))
                block_patches.append(patch)

        ax.add_collection(PatchCollection(
            block_patches, 
            facecolor=[self.district_colors.get(b['type'], '#cccccc') for b in self.blocks],
            edgecolor='black',
            alpha=0.4,
            zorder=1
        ))

        # Draw buildings
        for building in self.buildings:
            rect = Rectangle((building['x'], building['y']), building['width'], building['length'], color='#888', alpha=0.8)
            ax.add_patch(rect)

        # Add legend for districts
        legend_elements = [Patch(facecolor=color, edgecolor='gray', label=dtype.replace('_', ' ').title()) for dtype, color in self.district_colors.items()]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 1.0))

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

if __name__ == "__main__":
    generator = GTAMapGenerator(width=2000, height=2000, seed=42)
    generator.generate_complete_map()
    generator.visualize_map('gta_map_improved.png')
    generator.save_map('gta_map_improved.json')

