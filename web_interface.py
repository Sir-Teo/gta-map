from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from map_generator import GTAMapGenerator

app = Flask(__name__)

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
        seed = data.get('seed', None)
        
        # Create generator and generate map
        generator = GTAMapGenerator(width=width, height=height, seed=seed)
        map_data = generator.generate_complete_map()
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.set_xlim(0, generator.width)
        ax.set_ylim(0, generator.height)
        ax.set_aspect('equal')
        ax.set_facecolor('#2d5016')  # Dark green background
        
        # Draw water bodies
        for water in generator.water_bodies:
            if water['type'] == 'river':
                points = water['points']
                for i in range(len(points) - 1):
                    x_vals = [points[i][0], points[i+1][0]]
                    y_vals = [points[i][1], points[i+1][1]]
                    ax.plot(x_vals, y_vals, color='#4a90e2', linewidth=water['width'], alpha=0.8)
            elif water['type'] == 'lake':
                circle = plt.Circle(water['center'], water['radius'], color='#4a90e2', alpha=0.8)
                ax.add_patch(circle)
        
        # Draw districts
        district_colors = {
            'downtown': '#ffeb3b',
            'residential': '#4caf50',
            'industrial': '#9e9e9e',
            'commercial': '#ff9800',
            'suburban': '#8bc34a',
            'beach': '#ffc107',
            'hills': '#795548',
            'airport': '#607d8b'
        }
        
        for district in generator.districts:
            color = district_colors.get(district['type'], '#cccccc')
            circle = plt.Circle(district['center'], district['radius'], 
                              color=color, alpha=0.2, linewidth=2, fill=True)
            ax.add_patch(circle)
        
        # Draw parks
        for park in generator.parks:
            circle = plt.Circle([park['x'], park['y']], park['size'], 
                              color='#4caf50', alpha=0.6)
            ax.add_patch(circle)
        
        # Draw roads
        road_colors = {
            'highway': '#333333',
            'secondary': '#555555',
            'local': '#777777',
            'industrial': '#666666'
        }
        
        for road in generator.roads:
            color = road_colors.get(road['type'], '#888888')
            points = road['points']
            
            if len(points) > 1:
                for i in range(len(points) - 1):
                    x_vals = [points[i][0], points[i+1][0]]
                    y_vals = [points[i][1], points[i+1][1]]
                    ax.plot(x_vals, y_vals, color=color, linewidth=road['width'], alpha=0.8)
        
        # Draw buildings (simplified for web)
        building_colors = {
            'downtown': '#1976d2',
            'commercial': '#f57c00',
            'residential': '#388e3c',
            'industrial': '#616161',
            'suburban': '#689f38'
        }
        
        # Only draw a subset of buildings for performance
        buildings_to_draw = generator.buildings[::5]  # Every 5th building
        for building in buildings_to_draw:
            color = building_colors.get(building['type'], '#757575')
            rect = plt.Rectangle((building['x'] - building['width']/2, 
                                building['y'] - building['length']/2),
                               building['width'], building['length'],
                               color=color, alpha=0.7)
            ax.add_patch(rect)
        
        # Draw landmarks
        landmark_colors = {
            'airport': '#ff5722',
            'stadium': '#e91e63',
            'mall': '#9c27b0',
            'hospital': '#f44336',
            'university': '#3f51b5',
            'port': '#00bcd4'
        }
        
        for landmark in generator.landmarks:
            color = landmark_colors.get(landmark['type'], '#ff9800')
            rect = plt.Rectangle((landmark['x'] - landmark['size']/2, 
                                landmark['y'] - landmark['size']/2),
                               landmark['size'], landmark['size'],
                               color=color, alpha=0.8, linewidth=3)
            ax.add_patch(rect)
        
        ax.set_title(f'GTA-Style Generated Map (Seed: {generator.seed})', 
                    fontsize=16, fontweight='bold')
        ax.axis('off')  # Remove axes for cleaner look
        
        # Convert plot to base64 string
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
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
            'map_data': map_data
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
            
            with open(filepath, 'w') as f:
                json.dump(map_data, f, indent=2)
            
            return send_file(filepath, as_attachment=True, download_name=filename)
        
        elif format == 'png':
            # Generate high-res image
            filename = f'gta_map_seed_{generator.seed}.png'
            filepath = os.path.join('static', filename)
            
            generator.visualize_map(save_path=filepath, show_labels=False)
            
            return send_file(filepath, as_attachment=True, download_name=filename)
        
        else:
            return jsonify({'error': 'Invalid format'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5001)
