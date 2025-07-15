"""
Main map generator orchestrator.
Coordinates all the generation modules to create complete maps.
"""
import random
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

from ..config.settings import MapConfig
from .map_data import MapData


class MapGenerator:
    """
    Main orchestrator for map generation.
    Coordinates all the specialized generation modules.
    """
    
    def __init__(self, config: MapConfig):
        self.config = config
        self.map_data = MapData(config.width, config.height, config.grid_size)
        
        # Set random seeds for reproducibility
        random.seed(config.seed)
        np.random.seed(config.seed)
        
        # Initialize modules (will be set by dependency injection)
        self.terrain_generator = None
        self.district_generator = None
        self.transportation_generator = None
        self.urban_planner = None
        self.building_generator = None
        self.features_generator = None
        
        # Track generation state
        self._generation_steps = []
        self._current_step = 0
    
    def set_terrain_generator(self, terrain_generator):
        """Set the terrain generation module."""
        self.terrain_generator = terrain_generator
    
    def set_district_generator(self, district_generator):
        """Set the district generation module."""
        self.district_generator = district_generator
    
    def set_transportation_generator(self, transportation_generator):
        """Set the transportation generation module."""
        self.transportation_generator = transportation_generator
    
    def set_urban_planner(self, urban_planner):
        """Set the urban planning module."""
        self.urban_planner = urban_planner
    
    def set_building_generator(self, building_generator):
        """Set the building generation module."""
        self.building_generator = building_generator
    
    def set_features_generator(self, features_generator):
        """Set the features generation module."""
        self.features_generator = features_generator
    
    def generate_complete_map(self, progress_callback=None) -> MapData:
        """
        Generate a complete map with all features.
        
        Args:
            progress_callback: Optional callback function that receives progress updates
        
        Returns:
            MapData: The complete generated map
        """
        start_time = datetime.now()
        self.map_data.generation_seed = self.config.seed
        self.map_data.generation_timestamp = start_time.isoformat()
        
        steps = [
            ("Generating terrain and heightmap", self._generate_terrain),
            ("Placing districts and settlements", self._place_districts),
            ("Creating transportation network", self._generate_transportation),
            ("Building bridges and tunnels", self._generate_infrastructure),
            ("Creating railway network", self._generate_railways),
            ("Defining urban structure", self._plan_urban_layout),
            ("Placing buildings", self._generate_buildings),
            ("Adding natural features", self._generate_features),
        ]
        
        self._generation_steps = steps
        total_steps = len(steps)
        
        for i, (step_name, step_function) in enumerate(steps):
            self._current_step = i
            
            if progress_callback:
                progress_callback(step_name, i, total_steps)
            
            print(f"Step {i+1}/{total_steps}: {step_name}")
            step_function()
            print(f"✓ {step_name} completed")
        
        # After all features, connect features to roads
        if self.transportation_generator and hasattr(self.transportation_generator, 'connect_features_to_roads'):
            self.transportation_generator.connect_features_to_roads(self.map_data, self.config.transportation)
        # Finalize map data
        generation_time = (datetime.now() - start_time).total_seconds()
        self.map_data.metadata.update({
            'generation_time_seconds': generation_time,
            'config': self.config.to_dict(),
            'steps_completed': len(steps)
        })
        
        if progress_callback:
            progress_callback("Map generation complete!", total_steps, total_steps)
        
        print(f"Map generation completed in {generation_time:.2f} seconds!")
        return self.map_data
    
    def _generate_terrain(self):
        """Generate terrain features like heightmap and water bodies."""
        if not self.terrain_generator:
            raise ValueError("Terrain generator not set")
        
        self.terrain_generator.generate_heightmap(self.map_data, self.config.terrain)
        self.terrain_generator.define_land_and_water(self.map_data, self.config.terrain)
        self.terrain_generator.generate_water_bodies(self.map_data, self.config.water)
    
    def _place_districts(self):
        """Place and define city districts."""
        if not self.district_generator:
            raise ValueError("District generator not set")
        
        self.district_generator.place_districts(self.map_data, self.config.districts)
    
    def _generate_transportation(self):
        """Generate roads, highways, and basic transportation networks."""
        if not self.transportation_generator:
            raise ValueError("Transportation generator not set")
        
        self.transportation_generator.generate_highway_network(self.map_data, self.config.transportation)
        self.transportation_generator.generate_arterial_grid(self.map_data, self.config.transportation)
        self.transportation_generator.generate_local_roads(self.map_data, self.config.transportation)
    
    def _generate_infrastructure(self):
        """Generate bridges and tunnels for the transportation network."""
        if not self.transportation_generator:
            raise ValueError("Transportation generator not set")
        
        # Generate bridges and tunnels using the advanced transportation system
        if hasattr(self.transportation_generator, 'advanced_generator'):
            if self.config.transportation.enable_bridges:
                self.transportation_generator.advanced_generator.generate_bridges(self.map_data, self.config.transportation)
            
            if self.config.transportation.enable_tunnels:
                self.transportation_generator.advanced_generator.generate_tunnels(self.map_data, self.config.transportation)
    
    def _generate_railways(self):
        """Generate railway network."""
        if not self.transportation_generator:
            raise ValueError("Transportation generator not set")
        
        # Generate railways using the advanced transportation system
        if hasattr(self.transportation_generator, 'advanced_generator') and self.config.transportation.enable_railways:
            self.transportation_generator.advanced_generator.generate_railway_network(self.map_data, self.config.transportation)
    
    def _plan_urban_layout(self):
        """Define city blocks and urban structure."""
        if not self.urban_planner:
            raise ValueError("Urban planner not set")
        
        self.urban_planner.define_city_blocks(self.map_data, self.config)
        self.urban_planner.plan_zoning(self.map_data, self.config)
    
    def _generate_buildings(self):
        """Generate and place buildings."""
        if not self.building_generator:
            raise ValueError("Building generator not set")
        
        self.building_generator.populate_city_blocks(self.map_data, self.config.buildings)
    
    def _generate_features(self):
        """Generate POIs, landmarks, and parks, then new city features."""
        if not self.features_generator:
            raise ValueError("Features generator not set")
        self.features_generator.generate_parks(self.map_data, self.config)
        self.features_generator.generate_pois(self.map_data, self.config.pois)
        # Add new city features (plazas, playgrounds, sports fields, bus stops)
        if hasattr(self.features_generator, 'generate_all_city_features'):
            self.features_generator.generate_all_city_features(self.map_data, self.config)
    
    def get_generation_progress(self) -> Dict[str, Any]:
        """Get current generation progress information."""
        if not self._generation_steps:
            return {"status": "not_started", "progress": 0.0}
        
        total_steps = len(self._generation_steps)
        current_progress = (self._current_step / total_steps) * 100
        
        return {
            "status": "generating" if self._current_step < total_steps else "completed",
            "current_step": self._current_step,
            "total_steps": total_steps,
            "progress": current_progress,
            "current_step_name": self._generation_steps[min(self._current_step, total_steps-1)][0] if self._generation_steps else ""
        }
    
    def get_map_statistics(self) -> Dict[str, Any]:
        """Get statistics about the generated map."""
        return self.map_data.get_statistics()
    
    def export_map_data(self) -> Dict[str, Any]:
        """Export the map data as a dictionary."""
        return self.map_data.to_dict()


class MapGeneratorFactory:
    """
    Factory class for creating configured map generators.
    Handles dependency injection of all generation modules.
    """
    
    @staticmethod
    def create_generator(config: MapConfig) -> MapGenerator:
        """
        Create a fully configured map generator with all modules.
        
        Args:
            config: Map generation configuration
            
        Returns:
            MapGenerator: Configured generator ready to use
        """
        generator = MapGenerator(config)
        
        # Import and inject all the generation modules
        # (These will be created in subsequent steps)
        from ..modules.terrain import TerrainGenerator
        from ..modules.districts import DistrictGenerator  
        from ..modules.transportation import TransportationGenerator
        from ..modules.urban_planning import UrbanPlanner
        from ..modules.buildings import BuildingGenerator
        from ..modules.features import FeaturesGenerator
        
        generator.set_terrain_generator(TerrainGenerator())
        generator.set_district_generator(DistrictGenerator())
        generator.set_transportation_generator(TransportationGenerator())
        generator.set_urban_planner(UrbanPlanner())
        generator.set_building_generator(BuildingGenerator())
        generator.set_features_generator(FeaturesGenerator())
        
        return generator
    
    @staticmethod
    def create_from_preset(preset_name: str, **overrides) -> MapGenerator:
        """Create a generator from a configuration preset."""
        from ..config.settings import create_preset_config
        config = create_preset_config(preset_name, **overrides)
        return MapGeneratorFactory.create_generator(config)
    
    @staticmethod
    def create_default(**overrides) -> MapGenerator:
        """Create a generator with default configuration."""
        from ..config.settings import create_default_config
        config = create_default_config(**overrides)
        return MapGeneratorFactory.create_generator(config)


# Convenience functions for easy usage
def generate_map(width=2000, height=2000, seed=None, **config_overrides) -> MapData:
    """
    Convenience function to generate a map with simple parameters.
    
    Args:
        width: Map width
        height: Map height  
        seed: Random seed for reproducibility
        **config_overrides: Additional configuration overrides
        
    Returns:
        MapData: Generated map
    """
    from ..config.settings import create_default_config
    
    config = create_default_config(
        width=width, 
        height=height, 
        seed=seed, 
        **config_overrides
    )
    generator = MapGeneratorFactory.create_generator(config)
    return generator.generate_complete_map()


def generate_preset_map(preset_name: str, **overrides) -> MapData:
    """
    Convenience function to generate a map from a preset.
    
    Args:
        preset_name: Name of the preset configuration
        **overrides: Configuration overrides
        
    Returns:
        MapData: Generated map
    """
    generator = MapGeneratorFactory.create_from_preset(preset_name, **overrides)
    return generator.generate_complete_map() 