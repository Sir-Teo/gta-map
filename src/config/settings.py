"""
Configuration settings for the GTA Map Generator.
Centralized configuration for all map generation parameters.
"""
import random
from typing import Dict, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class TerrainConfig:
    """Configuration for terrain generation."""
    water_level: float = 0.35
    beach_level: float = 0.4
    mountain_count_range: Tuple[int, int] = (1, 3)
    terrain_scale: float = 0.005
    mountain_scale_multiplier: float = 2.0
    
    
@dataclass
class DistrictConfig:
    """Configuration for district placement and generation."""
    district_count: int = 10
    minimum_district_distance: float = 300.0
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
    arterial_spacing: int = 400
    collector_spacing: int = 200
    local_road_density: float = 0.3
    road_curve_factor: float = 0.2
    highway_count_range: Tuple[int, int] = (2, 4)
    
    road_styles: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'arterial': {'color': '#333333', 'width': 8},
        'collector': {'color': '#555555', 'width': 5},
        'highway': {'color': '#000000', 'width': 10},
        'local': {'color': '#777777', 'width': 3},
        'rural': {'color': '#999999', 'width': 2},
        'path': {'color': '#bbbbbb', 'width': 1}
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
    river_count_range: Tuple[int, int] = (0, 3)
    lake_count_range: Tuple[int, int] = (2, 5)
    river_width_range: Tuple[float, float] = (30.0, 60.0)
    lake_radius_range: Tuple[float, float] = (30.0, 100.0)


@dataclass
class MapConfig:
    """Main configuration class that combines all other configs."""
    name: str = "Generated City"
    width: int = 2000
    height: int = 2000
    seed: int = None
    grid_size: int = 20
    
    # Sub-configurations
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    districts: DistrictConfig = field(default_factory=DistrictConfig)
    transportation: TransportationConfig = field(default_factory=TransportationConfig)
    buildings: BuildingConfig = field(default_factory=BuildingConfig)
    pois: POIConfig = field(default_factory=POIConfig)
    water: WaterConfig = field(default_factory=WaterConfig)
    
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
        
        return cls(
            **main_params,
            terrain=terrain,
            districts=districts,
            transportation=transportation,
            buildings=buildings,
            pois=pois,
            water=water
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert MapConfig to a dictionary."""
        return {
            'width': self.width,
            'height': self.height,
            'seed': self.seed,
            'grid_size': self.grid_size,
            'terrain': self.terrain.__dict__,
            'districts': self.districts.__dict__,
            'transportation': self.transportation.__dict__,
            'buildings': self.buildings.__dict__,
            'pois': self.pois.__dict__,
            'water': self.water.__dict__
        }


def create_default_config(**overrides) -> MapConfig:
    """Create a default configuration with optional overrides."""
    return MapConfig(**overrides)


def create_preset_config(preset_name: str, **overrides) -> MapConfig:
    """Create predefined configuration presets."""
    presets = {
        'small_city': {
            'width': 1000,
            'height': 1000,
            'districts': {'district_count': 6}
        },
        'large_metropolis': {
            'width': 3000,
            'height': 3000,
            'districts': {'district_count': 15}
        },
        'island_paradise': {
            'width': 2000,
            'height': 2000,
            'terrain': {'water_level': 0.3, 'mountain_count_range': (2, 4)},
            'water': {'lake_count_range': (5, 8)}
        },
        'urban_sprawl': {
            'width': 2500,
            'height': 2500,
            'districts': {'district_count': 12},
            'buildings': {'density_map': {
                'residential': 0.0008,
                'suburban': 0.0006,
                'commercial': 0.001
            }}
        }
    }
    
    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
    
    preset_config = presets[preset_name]
    preset_config.update(overrides)
    
    return MapConfig.from_dict(preset_config) 