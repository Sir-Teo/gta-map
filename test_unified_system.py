#!/usr/bin/env python3
"""
Test script for the unified GTA Map Generator system.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.map_generator import MapGeneratorFactory
from src.config.settings import MapConfig, create_preset_config
from src.web.app import create_app


def test_basic_generation():
    """Test basic map generation"""
    print("Testing basic map generation...")
    
    config = MapConfig(
        name="Test City",
        width=1000,
        height=1000,
        seed=42
    )
    
    generator = MapGeneratorFactory.create_generator(config)
    map_data = generator.generate_complete_map()
    
    assert map_data is not None
    assert map_data.heightmap is not None
    assert map_data.districts is not None
    assert map_data.roads is not None
    assert map_data.buildings is not None
    assert map_data.pois is not None
    
    print("✅ Basic generation test passed")


def test_preset_generation():
    """Test preset-based generation"""
    print("Testing preset generation...")
    
    presets = ["small_city", "large_metropolis", "island_paradise"]
    
    for preset_name in presets:
        config = create_preset_config(preset_name)
        generator = MapGeneratorFactory.create_generator(config)
        map_data = generator.generate_complete_map()
        
        assert map_data is not None
        
        print(f"✅ {preset_name} preset test passed")


def test_web_app_creation():
    """Test web app creation"""
    print("Testing web app creation...")
    
    app = create_app()
    assert app is not None
    
    # Test basic routes exist
    with app.test_client() as client:
        response = client.get('/')
        assert response.status_code == 200
        
        response = client.get('/api/presets')
        assert response.status_code == 200
    
    print("✅ Web app test passed")


def test_export_formats():
    """Test export formats"""
    print("Testing export formats...")
    
    config = MapConfig(name="Export Test", width=500, height=500)
    generator = MapGeneratorFactory.create_generator(config)
    map_data = generator.generate_complete_map()
    
    # Test JSON export
    json_data = map_data.to_dict()
    assert json_data is not None
    assert "metadata" in json_data
    assert "terrain" in json_data
    assert "districts" in json_data
    
    # Test rendering
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        map_data.render_to_file(f.name)
        assert os.path.exists(f.name)
        os.unlink(f.name)
    
    print("✅ Export formats test passed")


def main():
    """Run all tests"""
    print("🧪 Testing Unified GTA Map Generator System")
    print("=" * 50)
    
    try:
        test_basic_generation()
        test_preset_generation()
        test_web_app_creation()
        test_export_formats()
        
        print("\n🎉 All tests passed! The unified system is working correctly.")
        print("\nTo start the web interface:")
        print("  python main.py")
        print("\nTo generate a map from command line:")
        print("  python main.py --generate")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 