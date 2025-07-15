"""
Core data structures for the GTA Map Generator.
Defines the base classes and data containers for all map elements.
"""
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from shapely.geometry import Polygon, LineString, Point
import numpy as np
import json


# Custom JSON encoder to handle Numpy types
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)


@dataclass
class District:
    """Represents a district/zone in the city."""
    id: str
    center: Tuple[float, float]
    district_type: str
    vertices: List[Tuple[float, float]]
    radius: float
    polygon: Optional[Polygon] = None
    color: str = "#ffffff"
    
    def __post_init__(self):
        """Create polygon from vertices if not provided."""
        if self.polygon is None and len(self.vertices) >= 3:
            try:
                self.polygon = Polygon(self.vertices)
            except Exception:
                self.polygon = None


@dataclass
class Road:
    """Represents a road segment."""
    id: str
    points: List[Tuple[float, float]]
    road_type: str  # 'highway', 'arterial', 'collector', 'local', etc.
    width: float
    color: str = "#333333"
    is_bridge: bool = False
    
    @property
    def linestring(self) -> Optional[LineString]:
        """Get the road as a Shapely LineString."""
        if len(self.points) >= 2:
            return LineString(self.points)
        return None


@dataclass
class Building:
    """Represents a building."""
    id: str
    x: float
    y: float
    width: float
    length: float
    building_type: str
    district_type: str = "generic"
    height: Optional[float] = None
    
    @property
    def polygon(self) -> Polygon:
        """Get the building as a Shapely Polygon."""
        return Polygon([
            (self.x, self.y),
            (self.x + self.width, self.y),
            (self.x + self.width, self.y + self.length),
            (self.x, self.y + self.length)
        ])
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get the center point of the building."""
        return (self.x + self.width/2, self.y + self.length/2)


@dataclass
class WaterBody:
    """Represents a water feature (river, lake, etc.)."""
    id: str
    water_type: str  # 'river', 'lake', 'ocean'
    geometry: Union[LineString, Polygon]
    color: str = "#0077cc"
    width: Optional[float] = None  # For rivers
    
    @property
    def points(self) -> List[Tuple[float, float]]:
        """Get points from the geometry."""
        if isinstance(self.geometry, LineString):
            return list(self.geometry.coords)
        elif isinstance(self.geometry, Polygon):
            return list(self.geometry.exterior.coords)
        return []


@dataclass
class POI:
    """Represents a Point of Interest (landmark)."""
    id: str
    name: str
    poi_type: str  # 'airport', 'stadium', 'hospital', etc.
    x: float
    y: float
    width: float
    height: float
    color: str = "#cc33cc"
    properties: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def polygon(self) -> Polygon:
        """Get the POI as a Shapely Polygon."""
        return Polygon([
            (self.x, self.y),
            (self.x + self.width, self.y),
            (self.x + self.width, self.y + self.height),
            (self.x, self.y + self.height)
        ])
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get the center point of the POI."""
        return (self.x + self.width/2, self.y + self.height/2)


@dataclass
class Park:
    """Represents a park or green space."""
    id: str
    name: str
    polygon: Polygon
    park_type: str = "public"
    color: str = "#33cc33"
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get the center point of the park."""
        centroid = self.polygon.centroid
        return (centroid.x, centroid.y)


@dataclass
class CityBlock:
    """Represents a city block defined by surrounding roads."""
    id: str
    polygon: Polygon
    district_type: str
    buildings: List[Building] = field(default_factory=list)
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get the center point of the block."""
        centroid = self.polygon.centroid
        return (centroid.x, centroid.y)
    
    @property
    def area(self) -> float:
        """Get the area of the block."""
        return self.polygon.area


class MapData:
    """
    Main container for all map data.
    This is the central data structure that holds all map elements.
    """
    
    def __init__(self, width: int, height: int, grid_size: int = 20):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.grid_width = width // grid_size
        self.grid_height = height // grid_size
        
        # Terrain data
        self.heightmap: Optional[np.ndarray] = None
        self.land_mask: Optional[np.ndarray] = None
        
        # Map elements
        self.districts: Dict[str, District] = {}
        self.roads: Dict[str, Road] = {}
        self.buildings: Dict[str, Building] = {}
        self.water_bodies: Dict[str, WaterBody] = {}
        self.pois: Dict[str, POI] = {}
        self.parks: Dict[str, Park] = {}
        self.city_blocks: Dict[str, CityBlock] = {}
        
        # Metadata
        self.generation_seed: Optional[int] = None
        self.generation_timestamp: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
    
    def add_district(self, district: District) -> None:
        """Add a district to the map."""
        self.districts[district.id] = district
    
    def add_road(self, road: Road) -> None:
        """Add a road to the map."""
        self.roads[road.id] = road
    
    def add_building(self, building: Building) -> None:
        """Add a building to the map."""
        self.buildings[building.id] = building
    
    def add_water_body(self, water_body: WaterBody) -> None:
        """Add a water body to the map."""
        self.water_bodies[water_body.id] = water_body
    
    def add_poi(self, poi: POI) -> None:
        """Add a POI to the map."""
        self.pois[poi.id] = poi
    
    def add_park(self, park: Park) -> None:
        """Add a park to the map."""
        self.parks[park.id] = park
    
    def add_city_block(self, city_block: CityBlock) -> None:
        """Add a city block to the map."""
        self.city_blocks[city_block.id] = city_block
    
    def get_district_by_point(self, x: float, y: float) -> Optional[District]:
        """Find which district contains the given point."""
        point = Point(x, y)
        for district in self.districts.values():
            if district.polygon and district.polygon.contains(point):
                return district
        return None
    
    def get_buildings_in_district(self, district_id: str) -> List[Building]:
        """Get all buildings in a specific district."""
        if district_id not in self.districts:
            return []
        
        district = self.districts[district_id]
        if not district.polygon:
            return []
        
        buildings = []
        for building in self.buildings.values():
            if district.polygon.contains(Point(building.center)):
                buildings.append(building)
        
        return buildings
    
    def get_roads_by_type(self, road_type: str) -> List[Road]:
        """Get all roads of a specific type."""
        return [road for road in self.roads.values() if road.road_type == road_type]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the map."""
        district_counts = {}
        for district in self.districts.values():
            district_type = district.district_type
            district_counts[district_type] = district_counts.get(district_type, 0) + 1
        
        road_counts = {}
        total_road_length = 0
        for road in self.roads.values():
            road_type = road.road_type
            road_counts[road_type] = road_counts.get(road_type, 0) + 1
            if road.linestring:
                total_road_length += road.linestring.length
        
        return {
            'dimensions': f"{self.width} x {self.height}",
            'districts': len(self.districts),
            'district_breakdown': district_counts,
            'roads': len(self.roads),
            'road_breakdown': road_counts,
            'total_road_length': total_road_length,
            'buildings': len(self.buildings),
            'water_bodies': len(self.water_bodies),
            'pois': len(self.pois),
            'parks': len(self.parks),
            'city_blocks': len(self.city_blocks),
            'seed': self.generation_seed
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the map data to a dictionary for serialization."""
        def serialize_geometry(geom):
            if geom is None:
                return None
            if isinstance(geom, Point):
                return list(geom.coords)
            elif isinstance(geom, LineString):
                return list(geom.coords)
            elif isinstance(geom, Polygon):
                if geom.is_valid:
                    return list(geom.exterior.coords)
                else:
                    return None
            return None
        
        return {
            'width': self.width,
            'height': self.height,
            'grid_size': self.grid_size,
            'generation_seed': self.generation_seed,
            'generation_timestamp': self.generation_timestamp,
            'metadata': self.metadata,
            'heightmap': self.heightmap.tolist() if self.heightmap is not None else None,
            'land_mask': self.land_mask.tolist() if self.land_mask is not None else None,
            'districts': {
                k: {
                    'id': v.id,
                    'center': v.center,
                    'district_type': v.district_type,
                    'vertices': v.vertices,
                    'radius': v.radius,
                    'color': v.color,
                    'polygon': serialize_geometry(v.polygon)
                } for k, v in self.districts.items()
            },
            'roads': {
                k: {
                    'id': v.id,
                    'points': v.points,
                    'road_type': v.road_type,
                    'width': v.width,
                    'color': v.color,
                    'is_bridge': v.is_bridge
                } for k, v in self.roads.items()
            },
            'buildings': {
                k: {
                    'id': v.id,
                    'x': v.x,
                    'y': v.y,
                    'width': v.width,
                    'length': v.length,
                    'building_type': v.building_type,
                    'district_type': v.district_type,
                    'height': v.height
                } for k, v in self.buildings.items()
            },
            'water_bodies': {
                k: {
                    'id': v.id,
                    'water_type': v.water_type,
                    'geometry': serialize_geometry(v.geometry),
                    'color': v.color,
                    'width': v.width
                } for k, v in self.water_bodies.items()
            },
            'pois': {
                k: {
                    'id': v.id,
                    'name': v.name,
                    'poi_type': v.poi_type,
                    'x': v.x,
                    'y': v.y,
                    'width': v.width,
                    'height': v.height,
                    'color': v.color,
                    'properties': v.properties
                } for k, v in self.pois.items()
            },
            'parks': {
                k: {
                    'id': v.id,
                    'name': v.name,
                    'park_type': v.park_type,
                    'color': v.color,
                    'polygon': serialize_geometry(v.polygon)
                } for k, v in self.parks.items()
            },
            'city_blocks': {
                k: {
                    'id': v.id,
                    'district_type': v.district_type,
                    'polygon': serialize_geometry(v.polygon),
                    'buildings': [b.id for b in v.buildings]
                } for k, v in self.city_blocks.items()
            }
        } 