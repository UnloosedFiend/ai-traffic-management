"""
Minimal Traffic Dashboard Server
=================================
A lightweight web dashboard that displays traffic signal status.
Works alongside run_multi_camera_inference.py to show real-time status.

Usage:
    python scripts/run_minimal_dashboard.py
    Then open http://localhost:5001 in browser

The dashboard receives updates via HTTP POST from the inference script.
"""

import sys
import time
import threading
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Flask app setup
app = Flask(__name__, 
            template_folder=str(project_root / 'src' / 'web' / 'templates'),
            static_folder=str(project_root / 'src' / 'web' / 'static'))
app.config['SECRET_KEY'] = 'traffic-dashboard-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')


# Global state
traffic_state = {
    'current_lane': 0,
    'remaining_time': 0,
    'emergency_active': False,
    'pi_connected': False,
    'lanes': {
        0: {'vehicles': 0, 'ambulance': 0, 'police': 0, 'signal': 'red'},
        1: {'vehicles': 0, 'ambulance': 0, 'police': 0, 'signal': 'red'},
        2: {'vehicles': 0, 'ambulance': 0, 'police': 0, 'signal': 'red'},
        3: {'vehicles': 0, 'ambulance': 0, 'police': 0, 'signal': 'red'},
    },
    'last_update': time.time()
}

state_lock = threading.Lock()


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard_minimal.html')


@app.route('/api/status')
def api_status():
    """Get current traffic status"""
    with state_lock:
        return jsonify(traffic_state)


@app.route('/api/update', methods=['POST'])
def api_update():
    """
    Receive status update from inference script.
    
    Expected JSON:
    {
        "current_lane": 0,
        "remaining_time": 15,
        "emergency_active": false,
        "pi_connected": true,
        "lanes": {
            "0": {"vehicles": 5, "ambulance": 0, "police": 0},
            "1": {"vehicles": 3, "ambulance": 1, "police": 0},
            ...
        }
    }
    """
    data = request.json
    
    with state_lock:
        if 'current_lane' in data:
            traffic_state['current_lane'] = data['current_lane']
        if 'remaining_time' in data:
            traffic_state['remaining_time'] = data['remaining_time']
        if 'emergency_active' in data:
            traffic_state['emergency_active'] = data['emergency_active']
        if 'pi_connected' in data:
            traffic_state['pi_connected'] = data['pi_connected']
        if 'lanes' in data:
            for lane_id, lane_data in data['lanes'].items():
                lane_idx = int(lane_id)
                if lane_idx in traffic_state['lanes']:
                    traffic_state['lanes'][lane_idx].update(lane_data)
        
        traffic_state['last_update'] = time.time()
    
    # Broadcast to all connected clients
    socketio.emit('traffic_update', {
        'current_lane': traffic_state['current_lane'],
        'remaining_time': traffic_state['remaining_time'],
        'emergency_active': traffic_state['emergency_active'],
        'pi_connected': traffic_state['pi_connected']
    })
    
    socketio.emit('detection_update', {
        'lanes': traffic_state['lanes']
    })
    
    return jsonify({'status': 'ok'})


@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    print("[DASHBOARD] Client connected")
    
    # Send current state
    with state_lock:
        socketio.emit('traffic_update', {
            'current_lane': traffic_state['current_lane'],
            'remaining_time': traffic_state['remaining_time'],
            'emergency_active': traffic_state['emergency_active'],
            'pi_connected': traffic_state['pi_connected']
        })
        
        socketio.emit('detection_update', {
            'lanes': traffic_state['lanes']
        })


@socketio.on('request_state')
def handle_request_state():
    """Handle state request from client"""
    with state_lock:
        socketio.emit('traffic_update', {
            'current_lane': traffic_state['current_lane'],
            'remaining_time': traffic_state['remaining_time'],
            'emergency_active': traffic_state['emergency_active'],
            'pi_connected': traffic_state['pi_connected']
        })
        
        socketio.emit('detection_update', {
            'lanes': traffic_state['lanes']
        })


def main():
    print("=" * 60)
    print("  Traffic Dashboard Server")
    print("=" * 60)
    print()
    print("Open http://localhost:5001 in your browser")
    print()
    print("The inference script will send updates to this dashboard.")
    print("Press Ctrl+C to stop.")
    print("=" * 60)
    
    socketio.run(app, host='0.0.0.0', port=5001, debug=False, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()
