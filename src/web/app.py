"""
Refactored web interface for the GTA Map Generator.
Uses the new modular architecture for scalable map generation.
"""
from flask import Flask, render_template, request, jsonify, send_file
import os
import json
from datetime import datetime

from ..core.map_generator import MapGeneratorFactory
from ..config.settings import create_default_config, create_preset_config
from ..rendering.map_renderer import MapRenderer


class MapGeneratorWebApp:
    """Web application for the GTA Map Generator."""
    
    def __init__(self):
        # Get the project root directory (two levels up from this file)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        template_dir = os.path.join(project_root, 'templates')
        static_dir = os.path.join(project_root, 'static')
        
        self.app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
        self.renderer = MapRenderer()
        self._setup_routes()
        
        # Create static directory
        os.makedirs(static_dir, exist_ok=True)
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/api/presets')
        def get_presets():
            """Get available configuration presets."""
            presets = {
                'small_city': {
                    'name': 'Small City',
                    'description': 'Compact urban area with essential districts',
                    'width': 1000,
                    'height': 1000
                },
                'large_metropolis': {
                    'name': 'Large Metropolis',
                    'description': 'Expansive city with many districts',
                    'width': 3000,
                    'height': 3000
                },
                'island_paradise': {
                    'name': 'Island Paradise',
                    'description': 'Tropical island with beaches and mountains',
                    'width': 2000,
                    'height': 2000
                },
                'urban_sprawl': {
                    'name': 'Urban Sprawl',
                    'description': 'Large suburban area with mixed development',
                    'width': 2500,
                    'height': 2500
                }
            }
            return jsonify(presets)
        
        @self.app.route('/generate', methods=['POST'])
        def generate_map():
            """Generate a new map with specified parameters."""
            try:
                data = request.get_json()
                
                # Extract parameters
                preset = data.get('preset')
                width = data.get('width', 2000)
                height = data.get('height', 2000)
                seed = data.get('seed')
                custom_config = data.get('config', {})
                
                # Create configuration
                if preset and preset != 'custom':
                    config = create_preset_config(preset, width=width, height=height, seed=seed, **custom_config)
                else:
                    config = create_default_config(width=width, height=height, seed=seed, **custom_config)
                
                # Create generator and generate map
                generator = MapGeneratorFactory.create_generator(config)
                
                # Generate map with progress tracking
                progress_updates = []
                def progress_callback(step_name, current, total):
                    progress_updates.append({
                        'step': step_name,
                        'current': current,
                        'total': total,
                        'progress': (current / total) * 100
                    })
                
                map_data = generator.generate_complete_map(progress_callback)
                
                # Render the map
                image_base64 = self.renderer.render_map(map_data, config, show_legend=True)
                
                # Get statistics
                stats = map_data.get_statistics()
                
                # Save map data for potential download
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                map_filename = f"map_{stats['seed']}_{timestamp}.json"
                map_filepath = os.path.join('static', map_filename)
                self.renderer.export_data(map_data, map_filepath, 'json')
                
                return jsonify({
                    'success': True,
                    'image': image_base64,
                    'stats': stats,
                    'progress': progress_updates,
                    'map_file': map_filename,
                    'generation_time': map_data.metadata.get('generation_time_seconds', 0)
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/download/<filename>')
        def download_map(filename):
            """Download generated map data."""
            try:
                filepath = os.path.join('static', filename)
                if os.path.exists(filepath):
                    return send_file(filepath, as_attachment=True, download_name=filename)
                else:
                    return jsonify({'error': 'File not found'}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/config/defaults')
        def get_default_config():
            """Get the default configuration structure."""
            config = create_default_config()
            return jsonify(config.to_dict())
        
        @self.app.route('/api/health')
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '2.0.0'
            })
    
    def run(self, host='0.0.0.0', port=5001, debug=True):
        """Run the web application."""
        print(f"🚀 Starting GTA Map Generator Web Interface v2.0")
        print(f"📍 Server running at http://{host}:{port}")
        print(f"🎯 Open your browser and start generating maps!")
        
        self.app.run(host=host, port=port, debug=debug)


def create_app():
    """Factory function to create the Flask app."""
    web_app = MapGeneratorWebApp()
    return web_app.app


if __name__ == '__main__':
    web_app = MapGeneratorWebApp()
    web_app.run() 