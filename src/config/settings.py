"""
Configuration settings for the GTA Map Generator.
Centralized configuration for all map generation parameters.
"""
import random
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class TerrainConfig:
    """Configuration for terrain generation."""
    water_level: float = 0.35
    beach_level: float = 0.4
    mountain_count_range: Tuple[int, int] = (2, 4)
    terrain_scale: float = 0.005
    mountain_scale_multiplier: float = 2.0
    enable_erosion: bool = True
    enable_biomes: bool = True
    continental_scale: float = 0.6
    
    
@dataclass
class DistrictConfig:
    """Configuration for district placement and generation."""
    district_count: int = 6  # Reduced for more organic placement
    minimum_district_distance: float = 500.0  # Increased for more organic spacing
    colors: Dict[str, str] = field(default_factory=lambda: {
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
    })
    key_districts: list = field(default_factory=lambda: [
        'downtown', 'commercial', 'industrial', 'residential', 'suburban', 
        'hills', 'beach', 'port', 'airport', 'park'
    ])


@dataclass
class TransportationConfig:
    """Configuration for road and transportation networks."""
    arterial_spacing: int = 700  # Slightly denser, but still organic
    collector_spacing: int = 400  # Slightly denser
    local_road_density: float = 0.28  # More local roads for richer city fabric
    road_curve_factor: float = 0.85  # Much more natural curves
    highway_count_range: Tuple[int, int] = (2, 4)  # Slightly more highways for realism
    enable_bridges: bool = True
    enable_tunnels: bool = True
    enable_railways: bool = True
    elevation_cost_factor: float = 2.2  # Roads follow terrain even more
    
    road_styles: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'arterial': {'color': '#3a3a3a', 'width': 7},  # Wider, darker
        'collector': {'color': '#5a5a5a', 'width': 5},  # Wider
        'highway': {'color': '#222222', 'width': 11},   # Much wider, deep black
        'local': {'color': '#b0b0b0', 'width': 3.5},    # Wider, lighter for city feel
        'rural': {'color': '#cccccc', 'width': 2},      # Wider, pale
        'path': {'color': '#e0e0e0', 'width': 1.5},     # Slightly wider
        'railway': {'color': '#8B4513', 'width': 4}     # Slightly wider
    })
    
    bridge_styles: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'highway': {'color': '#8B4513', 'width': 14},
        'road': {'color': '#8B4513', 'width': 10},
        'railway': {'color': '#654321', 'width': 8}
    })


@dataclass
class BuildingConfig:
    """Configuration for building placement and generation."""
    density_map: Dict[str, float] = field(default_factory=lambda: {
        'downtown': 0.001,
        'commercial': 0.0008,
        'residential': 0.0005,
        'industrial': 0.0004,
        'default': 0.0003
    })
    
    size_ranges: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=lambda: {
        'downtown': {'width': (15, 50), 'length': (15, 50)},
        'commercial': {'width': (10, 40), 'length': (10, 40)},
        'residential': {'width': (8, 25), 'length': (8, 25)},
        'industrial': {'width': (20, 60), 'length': (20, 60)},
        'default': {'width': (10, 30), 'length': (10, 30)}
    })


@dataclass
class POIConfig:
    """Configuration for Points of Interest (landmarks, etc.)."""
    poi_types: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'airport': {
            'size': (150, 250),
            'color': '#cc33cc',
            'buffer': 100,
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
    })


@dataclass
class WaterConfig:
    """Configuration for water body generation."""
    river_count_range: Tuple[int, int] = (2, 6)
    lake_count_range: Tuple[int, int] = (3, 8)
    river_width_range: Tuple[float, float] = (20.0, 50.0)
    lake_radius_range: Tuple[float, float] = (40.0, 120.0)
    enable_drainage_networks: bool = True
    coastal_features: bool = True


@dataclass
class BiomeConfig:
    """Configuration for biome generation."""
    enable_forests: bool = True
    enable_agricultural_areas: bool = True
    enable_wetlands: bool = True
    forest_coverage: float = 0.15  # Percentage of map covered by forests
    agricultural_coverage: float = 0.20  # Percentage suitable for agriculture
    
    biome_colors: Dict[str, str] = field(default_factory=lambda: {
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
    })


@dataclass
class MapConfig:
    """Main configuration class that combines all other configs."""
    name: str = "Generated Region"
    width: int = 4000
    height: int = 4000
    seed: int = None
    grid_size: int = 20  # Keep as is unless performance is an issue
    enable_multi_city: bool = True
    
    # Sub-configurations
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    districts: DistrictConfig = field(default_factory=DistrictConfig)
    transportation: TransportationConfig = field(default_factory=TransportationConfig)
    buildings: BuildingConfig = field(default_factory=BuildingConfig)
    pois: POIConfig = field(default_factory=POIConfig)
    water: WaterConfig = field(default_factory=WaterConfig)
    biomes: BiomeConfig = field(default_factory=BiomeConfig)
    
    def __post_init__(self):
        """Initialize computed properties after creation."""
        if self.seed is None:
            self.seed = random.randint(0, 10000)
            
        self.grid_width = self.width // self.grid_size
        self.grid_height = self.height // self.grid_size
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MapConfig':
        """Create a MapConfig from a dictionary."""
        # Extract main parameters
        main_params = {
            k: v for k, v in config_dict.items() 
            if k in ['width', 'height', 'seed', 'grid_size']
        }
        
        # Create sub-configs
        terrain = TerrainConfig(**config_dict.get('terrain', {}))
        districts = DistrictConfig(**config_dict.get('districts', {}))
        transportation = TransportationConfig(**config_dict.get('transportation', {}))
        buildings = BuildingConfig(**config_dict.get('buildings', {}))
        pois = POIConfig(**config_dict.get('pois', {}))
        water = WaterConfig(**config_dict.get('water', {}))
        biomes = BiomeConfig(**config_dict.get('biomes', {}))
        
        return cls(
            **main_params,
            terrain=terrain,
            districts=districts,
            transportation=transportation,
            buildings=buildings,
            pois=pois,
            water=water,
            biomes=biomes
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert MapConfig to a dictionary."""
        return {
            'width': self.width,
            'height': self.height,
            'seed': self.seed,
            'grid_size': self.grid_size,
            'enable_multi_city': self.enable_multi_city,
            'terrain': self.terrain.__dict__,
            'districts': self.districts.__dict__,
            'transportation': self.transportation.__dict__,
            'buildings': self.buildings.__dict__,
            'pois': self.pois.__dict__,
            'water': self.water.__dict__,
            'biomes': self.biomes.__dict__
        }


def create_default_config(**overrides) -> MapConfig:
    """Create a default configuration with optional overrides."""
    return MapConfig(**overrides)


def create_preset_config(preset_name: str, **overrides) -> MapConfig:
    """Create predefined configuration presets."""
    presets = {
        'small_town': {
            'name': 'Small Town',
            'width': 1200,
            'height': 1200,
            'enable_multi_city': False,
            'districts': {'district_count': 4},
            'transportation': {'highway_count_range': (1, 2), 'enable_railways': False}
        },
        'large_metropolis': {
            'name': 'Large Metropolis', 
            'width': 3500,
            'height': 3500,
            'terrain': {'mountain_count_range': (3, 5)},
            'water': {'river_count_range': (4, 8)},
            'transportation': {'highway_count_range': (6, 10), 'enable_railways': True}
        },
        'island_archipelago': {
            'name': 'Island Archipelago',
            'width': 2500,
            'height': 2500,
            'terrain': {'water_level': 0.3, 'mountain_count_range': (3, 6), 'continental_scale': 0.4},
            'water': {'river_count_range': (3, 6), 'lake_count_range': (6, 12), 'coastal_features': True},
            'biomes': {'forest_coverage': 0.25}
        },
        'continental_region': {
            'name': 'Continental Region',
            'width': 4000,
            'height': 4000,
            'terrain': {'mountain_count_range': (4, 7), 'continental_scale': 0.8},
            'water': {'river_count_range': (6, 12)},
            'biomes': {'forest_coverage': 0.2, 'agricultural_coverage': 0.3},
            'transportation': {'highway_count_range': (8, 12), 'enable_railways': True}
        },
        'mountain_valleys': {
            'name': 'Mountain Valleys',
            'width': 2000,
            'height': 2000,
            'terrain': {'mountain_count_range': (4, 6), 'water_level': 0.4},
            'water': {'river_count_range': (4, 8)},
            'biomes': {'forest_coverage': 0.3},
            'transportation': {'enable_tunnels': True, 'elevation_cost_factor': 2.0}
        }
    }
    
    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
    
    preset_config = presets[preset_name]
    preset_config.update(overrides)
    
    return MapConfig.from_dict(preset_config) 