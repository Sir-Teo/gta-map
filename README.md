# GTA 5-Style Map Generator

A sophisticated procedural map generator that creates GTA 5-inspired city maps with realistic urban features including road networks, districts, buildings, landmarks, and natural features.

## Features

🏙️ **Realistic City Generation**
- Procedural district generation with different zones (downtown, residential, industrial, commercial, etc.)
- Complex road network with highways, secondary roads, and local streets
- Thousands of procedurally placed buildings with appropriate sizing for each district type

🗺️ **Map Elements**
- **Districts**: Downtown, residential, industrial, commercial, suburban, beach, hills, airport
- **Roads**: Multi-level road hierarchy (highways, secondary, local, industrial)
- **Buildings**: Context-aware building placement with realistic sizing
- **Landmarks**: Airports, stadiums, malls, hospitals, universities, ports
- **Natural Features**: Rivers, lakes, parks, and green spaces
- **Water Bodies**: Procedural rivers and lakes

🎨 **Visualization & Export**
- Beautiful map visualization with color-coded districts and features
- Web interface for interactive map generation
- Export maps as high-resolution PNG images
- Export map data as JSON for further processing

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the web interface:
```bash
python web_interface.py
```

3. Open your browser and go to `http://localhost:5000`

## Usage

### Web Interface
1. Launch the web application
2. Adjust map parameters (width, height, seed)
3. Click "Generate Map" to create a new procedural city
4. Download the generated map as PNG or JSON

### Command Line
```python
from map_generator import GTAMapGenerator

# Create generator
generator = GTAMapGenerator(width=2000, height=2000, seed=42)

# Generate complete map
map_data = generator.generate_complete_map()

# Visualize the map
generator.visualize_map('my_map.png')

# Save map data
generator.save_map('my_map.json')
```

## Map Generation Process

1. **District Generation**: Creates city zones using Voronoi diagrams
2. **Road Network**: Generates hierarchical road system (highways → secondary → local)
3. **Water Bodies**: Places rivers and lakes using procedural algorithms
4. **Building Placement**: Distributes buildings based on district type and road proximity
5. **Landmarks**: Strategically places special buildings and points of interest
6. **Parks**: Adds green spaces throughout residential and urban areas

## Customization

### District Types
- **Downtown**: High-rise buildings, dense grid streets
- **Residential**: Medium-height buildings, organic street layouts
- **Industrial**: Large buildings, wide straight roads
- **Commercial**: Mixed-height buildings, grid patterns
- **Suburban**: Low buildings, curved residential streets

### Road Types
- **Highway**: Major ring roads and cross-city arteries (width: 15-20)
- **Secondary**: District connecting roads (width: 10)
- **Local**: Neighborhood streets (width: 5-6)
- **Industrial**: Wide industrial access roads (width: 8)

## Technical Details

- **Terrain Generation**: Uses Perlin noise for realistic elevation
- **Road Networks**: Implements curved roads using Bezier curves
- **Building Placement**: Validates positions to avoid road conflicts
- **Visualization**: Uses matplotlib for high-quality map rendering
- **Web Framework**: Flask-based interface for easy interaction

## Parameters

- `width`: Map width in units (default: 2000)
- `height`: Map height in units (default: 2000)
- `seed`: Random seed for reproducible generation
- `road_density`: Controls road network density
- `building_density`: Controls building placement density
- `district_count`: Number of city districts to generate

## Examples

Generate a small city:
```python
generator = GTAMapGenerator(width=1000, height=1000)
```

Generate with specific seed:
```python
generator = GTAMapGenerator(seed=12345)
```

Create a large metropolitan area:
```python
generator = GTAMapGenerator(width=3000, height=3000)
```

## Output Formats

### JSON Export
Contains complete map data including:
- All road segments with coordinates and types
- Building positions, sizes, and types
- District boundaries and classifications
- Landmark locations and properties
- Water body definitions

### PNG Export
High-resolution visual representation with:
- Color-coded districts and buildings
- Detailed road network visualization
- Landmark labels and icons
- Natural feature representation

## Performance

- Typical generation time: 5-15 seconds for 2000x2000 maps
- Memory usage: ~100-500MB depending on map size
- Supports maps up to 5000x5000 units

## Contributing

Feel free to extend the generator with:
- New district types
- Additional landmark categories
- Enhanced road generation algorithms
- Different architectural styles
- Terrain-aware building placement

## License

Open source - feel free to use and modify for your projects!
