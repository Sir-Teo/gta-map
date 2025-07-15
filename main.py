#!/usr/bin/env python3
"""
GTA Map Generator - Unified System
==================================

A comprehensive city map generation system with modular architecture.
Supports both command-line and web interface modes.

Usage:
    python main.py                    # Start web interface
    python main.py --cli              # Run CLI mode
    python main.py --generate         # Generate a map and save
    python main.py --preset <name>    # Use a specific preset
    python main.py --config <file>    # Use custom config file
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.map_generator import MapGeneratorFactory
from src.config.settings import MapConfig, create_preset_config
from src.web.app import create_app
from src.core.map_data import NumpyJSONEncoder
import json


def generate_map(config=None, preset_name=None, output_dir="output"):
    """Generate a map with the given configuration"""
    if preset_name:
        config = create_preset_config(preset_name)
    elif config is None:
        config = MapConfig()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate map
    generator = MapGeneratorFactory.create_generator(config)
    map_data = generator.generate_complete_map()
    
    # Save outputs
    base_name = f"gta_map_{config.name.lower().replace(' ', '_')}"
    
    # Save JSON data
    json_path = os.path.join(output_dir, f"{base_name}.json")
    with open(json_path, 'w') as f:
        json.dump(map_data.to_dict(), f, indent=2, cls=NumpyJSONEncoder)
    
    # Save PNG image
    from src.rendering.map_renderer import MapRenderer
    renderer = MapRenderer()
    png_path = os.path.join(output_dir, f"{base_name}.png")
    renderer.render_map(map_data, config, save_path=png_path)
    
    print(f"Map generated successfully!")
    print(f"  JSON: {json_path}")
    print(f"  PNG:  {png_path}")
    
    return map_data


def run_cli():
    """Run the system in CLI mode"""
    parser = argparse.ArgumentParser(description="GTA Map Generator")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode")
    parser.add_argument("--generate", action="store_true", help="Generate a map")
    parser.add_argument("--preset", type=str, help="Use a specific preset")
    parser.add_argument("--config", type=str, help="Custom config file")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    
    args = parser.parse_args()
    
    if args.generate or args.preset or args.config:
        # CLI generation mode
        config = None
        if args.config:
            with open(args.config, 'r') as f:
                config_data = json.load(f)
                config = MapConfig(**config_data)
        
        generate_map(config=config, preset_name=args.preset, output_dir=args.output)
    else:
        # Web interface mode
        app = create_app()
        app.run(host='0.0.0.0', port=5001, debug=True)


def run_web():
    """Run the web interface"""
    app = create_app()
    app.run(host='0.0.0.0', port=5001, debug=True)


if __name__ == "__main__":
    from src.core.map_generator import generate_map
    from src.rendering.map_renderer import MapRenderer
    from src.config.settings import create_default_config, DistrictConfig, TransportationConfig, BuildingConfig, WaterConfig

    # Enhanced config for a large, organic, beautiful map
    config_overrides = {
        'width': 3500,
        'height': 3500,
        'districts': DistrictConfig(**{
            'district_count': 16,
            'minimum_district_distance': 350.0,
        }),
        'transportation': TransportationConfig(**{
            'arterial_spacing': 1100,  # Denser arterials
            'collector_spacing': 600,  # Denser collectors
            'local_road_density': 0.45,  # Much more local roads
            'road_curve_factor': 1.7,
            'enable_bridges': True,
            'enable_tunnels': True,
            'enable_railways': True,
        }),
        'buildings': BuildingConfig(**{
            'density_map': {
                'downtown': 0.0015,
                'commercial': 0.0011,
                'residential': 0.0007,
                'industrial': 0.0004,
                'default': 0.00025
            }
        }),
        'water': WaterConfig(**{
            'river_count_range': (4, 7),
            'lake_count_range': (6, 12),
            'river_width_range': (50.0, 100.0),
            'coastal_features': True,
        }),
        'seed': 98765,
        'enable_multi_city': True,
    }

    print("Generating organized GTA map...")
    config = create_default_config(**config_overrides)
    generator = MapGeneratorFactory.create_generator(config)
    map_data = generator.generate_complete_map()
    print("Rendering map...")
    renderer = MapRenderer()
    renderer.render_map(map_data, config, save_path="static/gta_map_generated_region.png")
    print("Map generation and rendering complete! Output saved to static/gta_map_generated_region.png")

    if len(sys.argv) > 1:
        run_cli()
    else:
        run_web() 