"""
Map rendering module.
Handles visualization and export of generated maps.
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web compatibility
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle, Polygon as MatplotlibPolygon, Patch
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
import numpy as np
from typing import Optional, Dict, Any
import json

from ..core.map_data import MapData
from ..config.settings import MapConfig


class MapRenderer:
    """
    Handles rendering and visualization of generated maps.
    """
    
    def __init__(self):
        self.figure = None
        self.ax = None
    
    def render_map(
        self, 
        map_data: MapData, 
        config: MapConfig, 
        save_path: Optional[str] = None,
        show_legend: bool = True,
        dpi: int = 300
    ) -> str:
        """
        Render the complete map with all features.
        
        Args:
            map_data: The map data to render
            config: Map configuration for styling
            save_path: Optional path to save the image
            show_legend: Whether to show the legend
            dpi: Image resolution
            
        Returns:
            Base64 encoded image string
        """
        # Create figure and axis
        fig_width = 16
        fig_height = 12
        self.figure, self.ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        
        # Render terrain background with biomes
        self._render_terrain(map_data, config)
        
        # Render biomes
        self._render_biomes(map_data, config)
        
        # Render water bodies
        self._render_water_bodies(map_data)
        
        # Render districts
        self._render_districts(map_data, config)
        
        # Render roads and railways
        self._render_roads(map_data, config)
        self._render_railways(map_data, config)
        
        # Render bridges and tunnels
        self._render_bridges(map_data, config)
        self._render_tunnels(map_data, config)
        
        # Render buildings
        self._render_buildings(map_data, config)
        
        # Render parks and agricultural zones
        self._render_parks(map_data)
        self._render_agricultural_zones(map_data, config)
        
        # Render POIs
        self._render_pois(map_data)
        
        # Configure map appearance
        self._configure_map_appearance(map_data, show_legend, config)
        
        # Save or return image
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
        
        # Convert to base64 for web display
        import io
        import base64
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        
        plt.close(self.figure)
        
        return image_base64
    
    def _render_terrain(self, map_data: MapData, config: MapConfig):
        """Render the terrain background."""
        if map_data.heightmap is not None:
            # Create extent for the heightmap
            extent = [0, map_data.width, 0, map_data.height]
            
            # Use a terrain colormap
            cmap = plt.cm.terrain
            
            # Display heightmap as background
            self.ax.imshow(
                map_data.heightmap, 
                extent=extent, 
                origin='lower',
                cmap=cmap, 
                alpha=0.3,
                aspect='equal'
            )
    
    def _render_water_bodies(self, map_data: MapData):
        """Render water bodies like rivers and lakes."""
        for water_body in map_data.water_bodies.values():
            if water_body.water_type == 'river':
                # Render river as a line with width
                points = water_body.points
                if len(points) > 1:
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    
                    self.ax.plot(
                        x_coords, y_coords,
                        color=water_body.color,
                        linewidth=water_body.width / 5,  # Scale for visibility
                        alpha=0.8,
                        solid_capstyle='round',
                        zorder=2
                    )
            
            elif water_body.water_type == 'lake':
                # Render lake as a filled polygon
                coords = water_body.points
                if len(coords) > 2:
                    polygon = MatplotlibPolygon(
                        coords,
                        closed=True,
                        facecolor=water_body.color,
                        alpha=0.7,
                        zorder=2
                    )
                    self.ax.add_patch(polygon)
    
    def _render_districts(self, map_data: MapData, config: MapConfig):
        """Render district boundaries and colors."""
        for district in map_data.districts.values():
            if district.polygon and district.vertices:
                # Create polygon patch
                polygon = MatplotlibPolygon(
                    district.vertices,
                    closed=True,
                    facecolor=district.color,
                    alpha=0.3,
                    edgecolor='#555555',
                    linewidth=1,
                    zorder=3
                )
                self.ax.add_patch(polygon)
                
                # Add district label
                center_x, center_y = district.center
                self.ax.text(
                    center_x, center_y,
                    district.district_type.replace('_', ' ').title(),
                    fontsize=10,
                    ha='center',
                    va='center',
                    weight='bold',
                    color='#333333',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2),
                    zorder=11
                )
    
    def _render_roads(self, map_data: MapData, config: MapConfig):
        """Render road networks."""
        # Group roads by type for consistent rendering
        road_types = ['highway', 'arterial', 'collector', 'local']
        
        for road_type in road_types:
            roads = [r for r in map_data.roads.values() if r.road_type == road_type]
            
            for road in roads:
                if len(road.points) > 1:
                    x_coords = [p[0] for p in road.points]
                    y_coords = [p[1] for p in road.points]
                    
                    # Get road style
                    style = config.transportation.road_styles.get(
                        road_type, 
                        config.transportation.road_styles['local']
                    )
                    
                    # Render road with outline for better visibility
                    outline_width = style['width'] + 2
                    
                    # Road outline (darker)
                    self.ax.plot(
                        x_coords, y_coords,
                        color='#000000',
                        linewidth=outline_width,
                        solid_capstyle='round',
                        alpha=0.8,
                        zorder=6
                    )
                    
                    # Road surface
                    self.ax.plot(
                        x_coords, y_coords,
                        color=style['color'],
                        linewidth=style['width'],
                        solid_capstyle='round',
                        alpha=0.9,
                        zorder=7
                    )
                    
                    # Center line for highways
                    if road_type == 'highway':
                        self.ax.plot(
                            x_coords, y_coords,
                            color='#FFFF00',
                            linewidth=style['width'] * 0.2,
                            solid_capstyle='round',
                            alpha=0.7,
                            zorder=8,
                            linestyle=(0, (5, 10))
                        )
    
    def _render_buildings(self, map_data: MapData, config: MapConfig):
        """Render buildings with 3D-like appearance."""
        for building in map_data.buildings.values():
            # Building shadow for 3D effect
            shadow_offset = 2
            shadow_rect = Rectangle(
                (building.x + shadow_offset, building.y - shadow_offset),
                building.width,
                building.length,
                facecolor='black',
                alpha=0.2,
                zorder=9
            )
            self.ax.add_patch(shadow_rect)
            
            # Main building
            building_color = self._get_building_color(building.district_type)
            building_rect = Rectangle(
                (building.x, building.y),
                building.width,
                building.length,
                facecolor=building_color,
                edgecolor='#333333',
                linewidth=0.5,
                alpha=0.9,
                zorder=10
            )
            self.ax.add_patch(building_rect)
    
    def _render_pois(self, map_data: MapData):
        """Render Points of Interest."""
        for poi in map_data.pois.values():
            # POI rectangle
            poi_rect = Rectangle(
                (poi.x, poi.y),
                poi.width,
                poi.height,
                facecolor=poi.color,
                edgecolor='#000000',
                linewidth=2,
                alpha=0.8,
                zorder=12
            )
            self.ax.add_patch(poi_rect)
            
            # POI label
            center_x = poi.x + poi.width / 2
            center_y = poi.y + poi.height / 2
            
            self.ax.text(
                center_x, center_y,
                poi.name,
                fontsize=8,
                ha='center',
                va='center',
                weight='bold',
                color='white',
                bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=1),
                zorder=13
            )
    
    def _render_parks(self, map_data: MapData):
        """Render parks and green spaces."""
        for park in map_data.parks.values():
            coords = list(park.polygon.exterior.coords)
            
            # Park polygon
            park_polygon = MatplotlibPolygon(
                coords,
                closed=True,
                facecolor=park.color,
                edgecolor='#228B22',
                linewidth=1,
                alpha=0.6,
                zorder=4
            )
            self.ax.add_patch(park_polygon)
            
            # Park label for larger parks
            if park.polygon.area > 10000:  # Only label large parks
                centroid = park.polygon.centroid
                self.ax.text(
                    centroid.x, centroid.y,
                    park.name,
                    fontsize=8,
                    ha='center',
                    va='center',
                    weight='bold',
                    color='#006400',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1),
                    zorder=5
                )
    
    def _configure_map_appearance(self, map_data: MapData, show_legend: bool, config: MapConfig):
        """Configure the overall map appearance."""
        # Set map limits
        self.ax.set_xlim(0, map_data.width)
        self.ax.set_ylim(0, map_data.height)
        
        # Set equal aspect ratio
        self.ax.set_aspect('equal')
        
        # Remove axes
        self.ax.axis('off')
        
        # Add title
        title = f"GTA-Style Procedural Map (Seed: {map_data.generation_seed})"
        self.ax.set_title(title, fontsize=20, pad=20, fontweight='bold')
        
        # Add legend if requested
        if show_legend:
            self._add_legend(config)
    
    def _add_legend(self, config: MapConfig):
        """Add legend to the map."""
        # District legend elements
        district_elements = []
        for district_type, color in config.districts.colors.items():
            if district_type in ['downtown', 'commercial', 'residential', 'industrial', 'suburban']:
                district_elements.append(
                    Patch(facecolor=color, alpha=0.5, edgecolor='#555555', 
                         label=district_type.replace('_', ' ').title())
                )
        
        # Special district elements
        special_elements = []
        for district_type, color in config.districts.colors.items():
            if district_type in ['beach', 'airport', 'hills', 'port', 'park']:
                special_elements.append(
                    Patch(facecolor=color, alpha=0.5, edgecolor='#555555',
                         label=district_type.replace('_', ' ').title())
                )
        
        # Road elements
        road_elements = []
        for road_type, style in config.transportation.road_styles.items():
            if road_type in ['highway', 'arterial', 'collector', 'local']:
                road_elements.append(
                    Line2D([0], [0], color=style['color'], linewidth=4,
                          label=road_type.replace('_', ' ').title())
                )
        
        # Feature elements
        feature_elements = [
            Patch(facecolor='#4682B4', alpha=0.7, label='Water'),
            Patch(facecolor='#33cc33', alpha=0.6, label='Parks'),
            Patch(facecolor='#A0A0A0', alpha=0.9, label='Buildings')
        ]
        
        # Create legends
        legend1 = self.ax.legend(handles=district_elements, loc='upper left',
                               title='Urban Districts', framealpha=0.9, title_fontsize=10)
        
        legend2 = self.ax.legend(handles=special_elements, loc='upper left',
                               title='Special Areas', framealpha=0.9, title_fontsize=10,
                               bbox_to_anchor=(0.0, 0.7))
        
        legend3 = self.ax.legend(handles=road_elements, loc='upper right',
                               title='Transportation', framealpha=0.9, title_fontsize=10)
        
        legend4 = self.ax.legend(handles=feature_elements, loc='upper right',
                               title='Features', framealpha=0.9, title_fontsize=10,
                               bbox_to_anchor=(1.0, 0.7))
        
        # Add all legends to the plot
        self.ax.add_artist(legend1)
        self.ax.add_artist(legend2)
        self.ax.add_artist(legend3)
        
    def _get_building_color(self, district_type: str) -> str:
        """Get building color based on district type."""
        colors = {
            'downtown': '#A0A0A0',      # Gray for downtown skyscrapers
            'commercial': '#C0C0C0',    # Light gray for commercial buildings
            'industrial': '#909090',    # Dark gray for industrial
            'residential': '#D0D0D0',   # Light gray for residential
            'suburban': '#E0E0E0',      # Very light gray for suburban
            'airport': '#B0B0B0',       # Medium gray for airport buildings
            'port': '#808080',          # Darker gray for port facilities
            'hills': '#F0F0F0',         # Very light for hills
            'beach': '#D8D8D8',         # Light for beach buildings
        }
        return colors.get(district_type, '#D0D0D0')
    
    def _render_biomes(self, map_data: MapData, config: MapConfig):
        """Render biome backgrounds."""
        if hasattr(map_data, 'biome_map') and map_data.biome_map is not None:
            # Get biome colors from config
            biome_colors = getattr(config, 'biomes', None)
            if biome_colors and hasattr(biome_colors, 'biome_colors'):
                colors = biome_colors.biome_colors
            else:
                # Default biome colors
                colors = {
                    'water': '#0077cc',
                    'beach': '#F4E4BC',
                    'coastal_plains': '#90EE90',
                    'plains': '#9ACD32',
                    'agricultural': '#DAA520',
                    'wetlands': '#2E8B57',
                    'forest': '#228B22',
                    'hills': '#8FBC8F',
                    'mountain_forest': '#556B2F',
                    'mountain_peaks': '#A0522D'
                }
            
            # Create a color array for the biome map
            height, width = map_data.biome_map.shape
            color_array = np.zeros((height, width, 4))  # RGBA
            
            for y in range(height):
                for x in range(width):
                    biome = map_data.biome_map[y, x]
                    if biome in colors:
                        hex_color = colors[biome]
                        # Convert hex to RGBA
                        rgb = mcolors.hex2color(hex_color)
                        color_array[y, x] = (*rgb, 0.4)  # Semi-transparent
            
            # Display biome map
            extent = [0, map_data.width, 0, map_data.height]
            self.ax.imshow(
                color_array,
                extent=extent,
                origin='lower',
                aspect='equal',
                zorder=1
            )
    
    def _render_railways(self, map_data: MapData, config: MapConfig):
        """Render railway networks."""
        railways = [r for r in map_data.roads.values() if r.road_type == 'railway']
        
        for railway in railways:
            if len(railway.points) > 1:
                x_coords = [p[0] for p in railway.points]
                y_coords = [p[1] for p in railway.points]
                
                # Railway ties (sleepers)
                self.ax.plot(
                    x_coords, y_coords,
                    color='#8B4513',
                    linewidth=railway.width + 2,
                    solid_capstyle='round',
                    alpha=0.8,
                    zorder=7
                )
                
                # Railway tracks
                for offset in [-1, 1]:
                    offset_x = [x + offset for x in x_coords]
                    self.ax.plot(
                        offset_x, y_coords,
                        color='#C0C0C0',
                        linewidth=1,
                        solid_capstyle='round',
                        alpha=0.9,
                        zorder=8
                    )
    
    def _render_bridges(self, map_data: MapData, config: MapConfig):
        """Render bridges."""
        if hasattr(map_data, 'bridges'):
            for bridge in map_data.bridges:
                # Bridge deck
                x_coords = [bridge.start_point[0], bridge.end_point[0]]
                y_coords = [bridge.start_point[1], bridge.end_point[1]]
                
                # Bridge outline
                self.ax.plot(
                    x_coords, y_coords,
                    color='#654321',
                    linewidth=bridge.width + 4,
                    solid_capstyle='round',
                    alpha=0.9,
                    zorder=9
                )
                
                # Bridge surface
                bridge_color = '#8B7355' if bridge.bridge_type == 'railway' else '#D2691E'
                self.ax.plot(
                    x_coords, y_coords,
                    color=bridge_color,
                    linewidth=bridge.width,
                    solid_capstyle='round',
                    alpha=0.9,
                    zorder=10
                )
    
    def _render_tunnels(self, map_data: MapData, config: MapConfig):
        """Render tunnel portals."""
        if hasattr(map_data, 'tunnels'):
            for tunnel in map_data.tunnels:
                # Tunnel portals at start and end
                for point in [tunnel.start_point, tunnel.end_point]:
                    # Portal entrance
                    circle = plt.Circle(
                        point,
                        tunnel.width / 2,
                        facecolor='#404040',
                        edgecolor='#202020',
                        linewidth=2,
                        alpha=0.8,
                        zorder=8
                    )
                    self.ax.add_patch(circle)
    
    def _render_agricultural_zones(self, map_data: MapData, config: MapConfig):
        """Render agricultural zones."""
        agricultural_parks = [p for p in map_data.parks.values() if p.park_type == 'agricultural']
        
        for zone in agricultural_parks:
            if zone.polygon:
                # Agricultural field pattern
                polygon = MatplotlibPolygon(
                    list(zone.polygon.exterior.coords),
                    closed=True,
                    facecolor='#DAA520',
                    alpha=0.4,
                    edgecolor='#B8860B',
                    linewidth=1,
                    linestyle='--',
                    zorder=4
                )
                self.ax.add_patch(polygon)
                
                # Add simple field marker
                center = zone.center
                self.ax.text(
                    center[0], center[1],
                    'AG',
                    fontsize=8,
                    ha='center',
                    va='center',
                    weight='bold',
                    color='#8B7B00',
                    alpha=0.7,
                    zorder=5
                )
    
    def export_data(self, map_data: MapData, export_path: str, format: str = 'json'):
        """
        Export map data to various formats.
        
        Args:
            map_data: Map data to export
            export_path: Path to save the exported data
            format: Export format ('json', 'geojson')
        """
        if format.lower() == 'json':
            self._export_json(map_data, export_path)
        elif format.lower() == 'geojson':
            self._export_geojson(map_data, export_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self, map_data: MapData, export_path: str):
        """Export map data as JSON."""
        from ..core.map_data import NumpyJSONEncoder
        
        data = map_data.to_dict()
        
        with open(export_path, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyJSONEncoder)
    
    def _export_geojson(self, map_data: MapData, export_path: str):
        """Export map data as GeoJSON."""
        # This is a simplified GeoJSON export
        # A full implementation would include proper coordinate transformations
        
        features = []
        
        # Add districts as features
        for district in map_data.districts.values():
            if district.polygon:
                feature = {
                    "type": "Feature",
                    "properties": {
                        "type": "district",
                        "district_type": district.district_type,
                        "name": district.district_type.replace('_', ' ').title()
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [list(district.polygon.exterior.coords)]
                    }
                }
                features.append(feature)
        
        # Add roads as features
        for road in map_data.roads.values():
            feature = {
                "type": "Feature",
                "properties": {
                    "type": "road",
                    "road_type": road.road_type,
                    "width": road.width
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": road.points
                }
            }
            features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        with open(export_path, 'w') as f:
            json.dump(geojson, f, indent=2) 