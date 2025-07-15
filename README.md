# GTA Map Generator - Unified System

A comprehensive city map generation system with modular architecture, supporting both command-line and web interface modes.

## 🏗️ Architecture

The system is built with a clean, modular architecture:

```
src/
├── config/          # Configuration management
├── core/            # Core data structures and generator
├── modules/         # Specialized generation modules
├── rendering/       # Visualization and export
└── web/            # Web interface and API
```

### Core Components

- **MapGenerator**: Main orchestrator coordinating all modules
- **MapData**: Central data structure holding all map information
- **Configuration**: Typed settings with presets for different city types
- **Modules**: Specialized generators for terrain, districts, transportation, etc.
- **Rendering**: Multi-format export (PNG, JSON, GeoJSON)

## 🚀 Quick Start

### Web Interface (Recommended)
```bash
python main.py
```
Then open http://localhost:5001 in your browser.

### Command Line
```bash
# Generate with default settings
python main.py --generate

# Use a preset
python main.py --preset "Modern City"

# Custom config file
python main.py --config my_config.json

# Specify output directory
python main.py --generate --output my_maps/
```

## 📋 Requirements

```bash
pip install -r requirements.txt
```

## 🎯 Features

### City Generation
- **Terrain**: Procedural height maps with realistic features
- **Districts**: Zoned areas (residential, commercial, industrial)
- **Transportation**: Road networks, highways, public transit
- **Urban Planning**: City blocks, zoning, density management
- **Buildings**: Procedural building placement and types
- **Features**: POIs, parks, landmarks, special areas

### Visualization & Export
- **PNG Images**: High-quality map visualizations
- **JSON Data**: Complete map data for analysis
- **GeoJSON**: Geographic data for GIS applications
- **Statistics**: Detailed generation metrics

### Configuration
- **Presets**: Pre-configured city types
- **Custom Settings**: Full control over all parameters
- **Validation**: Type-safe configuration with validation

## 🏙️ City Types

### Available Presets
- **Modern City**: Contemporary urban layout
- **Historic City**: Traditional European-style
- **Industrial City**: Factory-focused layout
- **Resort City**: Tourist-friendly design
- **Suburban City**: Low-density residential

### Custom Configuration
All aspects can be customized:
- Map dimensions and resolution
- District types and proportions
- Road network density and types
- Building styles and placement
- Feature distribution and types

## 🔧 Development

### Project Structure
```
gta-map/
├── main.py              # Unified entry point
├── src/                 # Source code
│   ├── config/         # Configuration management
│   ├── core/           # Core data structures
│   ├── modules/        # Generation modules
│   ├── rendering/      # Visualization
│   └── web/           # Web interface
├── templates/          # Web templates
├── static/            # Web assets
└── output/            # Generated maps
```

### Adding New Features
1. **Modules**: Add new generation logic in `src/modules/`
2. **Configuration**: Extend settings in `src/config/settings.py`
3. **Rendering**: Add export formats in `src/rendering/`
4. **Web Interface**: Add endpoints in `src/web/app.py`

### Testing
```bash
# Run the test suite
python -m pytest tests/

# Generate test maps
python main.py --generate --preset "Test City"
```

## 📊 Output Formats

### PNG Images
- High-resolution map visualizations
- Color-coded districts and features
- Customizable styling and themes

### JSON Data
```json
{
  "metadata": { "name": "City Name", "version": "1.0" },
  "terrain": { "height_map": [...], "features": [...] },
  "districts": [{ "name": "Downtown", "type": "commercial", ... }],
  "transportation": { "roads": [...], "highways": [...] },
  "buildings": [{ "type": "skyscraper", "position": {...} }],
  "features": [{ "type": "park", "name": "Central Park", ... }]
}
```

### GeoJSON
- Geographic data for GIS applications
- Compatible with mapping software
- Feature collections for analysis

## 🎨 Customization

### Creating Custom Presets
```python
from src.config.settings import MapConfig

custom_config = MapConfig(
    name="My Custom City",
    width=2000,
    height=2000,
    district_config=DistrictConfig(
        residential_ratio=0.6,
        commercial_ratio=0.3,
        industrial_ratio=0.1
    ),
    # ... other settings
)
```

### Extending Generation Modules
```python
from src.modules.base import BaseModule

class CustomModule(BaseModule):
    def generate(self, map_data):
        # Your custom generation logic
        pass
```

## 🔄 API Endpoints

### Web Interface
- `GET /`: Main interface
- `POST /api/generate`: Generate new map
- `GET /api/presets`: List available presets
- `GET /api/download/<filename>`: Download generated files
- `GET /api/status`: Generation progress

### Configuration
- `GET /api/config`: Current configuration
- `POST /api/config`: Update configuration
- `GET /api/presets/<name>`: Get preset details

## 📈 Performance

- **Generation Time**: 5-30 seconds depending on complexity
- **Memory Usage**: Optimized for large maps
- **Scalability**: Modular design supports easy scaling
- **Caching**: Intelligent caching for repeated operations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed information

---

**Built with ❤️ using Python, Flask, and modern software architecture principles.**
