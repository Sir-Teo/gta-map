from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from map_generator import GTAMapGenerator, NumpyJSONEncoder
from shapely.geometry import Polygon, LineString, Point

app = Flask(__name__)
app.json_encoder = NumpyJSONEncoder

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_map():
    try:
        data = request.get_json()
        
        # Get parameters from request
        width = data.get('width', 2000)
        height = data.get('height', 2000)
        seed = data.get('seed')
        if seed is None:
            seed = 42
        
        # Create generator and generate map
        generator = GTAMapGenerator(width=width, height=height, seed=seed)
        map_data = generator.generate_complete_map()
        json_compatible_data = generator._prepare_for_json(map_data)

        # Create visualization identical to local script
        static_dir = 'static'
        os.makedirs(static_dir, exist_ok=True)
        temp_img_path = os.path.join(static_dir, 'latest_map.png')
        generator.visualize_map(save_path=temp_img_path)
        
        # Load the saved image for base64 encoding
        with open(temp_img_path, 'rb') as img_f:
            img_bytes = img_f.read()
        img_base64 = base64.b64encode(img_bytes).decode()
        plt.close('all')
        
        # Prepare statistics
        stats = {
            'seed': generator.seed,
            'dimensions': f"{width} x {height}",
            'districts': len(generator.districts),
            'roads': len(generator.roads),
            'buildings': len(generator.buildings),
            'landmarks': len(generator.landmarks),
            'parks': len(generator.parks),
            'water_bodies': len(generator.water_bodies)
        }
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'stats': stats,
            'mapData': json_compatible_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/download/<format>')
def download_map(format):
    try:
        # Get the last generated map data from session or regenerate
        # For simplicity, we'll regenerate with default parameters
        generator = GTAMapGenerator(width=2000, height=2000)
        map_data = generator.generate_complete_map()
        
        if format == 'json':
            # Save as JSON
            filename = f'gta_map_seed_{generator.seed}.json'
            filepath = os.path.join('static', filename)
            
            # Prepare data for JSON serialization
            json_compatible_data = generator._prepare_for_json(map_data)
            
            with open(filepath, 'w') as f:
                json.dump(json_compatible_data, f, indent=2)
            
            return send_file(filepath, as_attachment=True, download_name=filename)
        
        elif format == 'png':
            # Generate high-res image
            filename = f'gta_map_seed_{generator.seed}.png'
            filepath = os.path.join('static', filename)
            
            generator.visualize_map(save_path=filepath)
            
            return send_file(filepath, as_attachment=True, download_name=filename)
        
        else:
            return jsonify({'error': 'Invalid format'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5001)
