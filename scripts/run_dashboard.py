#!/usr/bin/env python3
"""
Quick launcher for the AI Traffic Management Web Dashboard.

This script starts the web-based dashboard that shows:
- 4 camera feeds with real-time detection
- Vehicle, ambulance, police counts
- Current signal state and timing
- Emergency vehicle alerts

Usage:
    python scripts/run_dashboard.py
    
Then open http://localhost:5001 in your browser.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.web.dashboard import run_dashboard

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Traffic Management Dashboard")
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5001, help='Port to listen on (default: 5001)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    print()
    print("Starting AI Traffic Management Dashboard...")
    print(f"Open http://localhost:{args.port} in your browser")
    print()
    
    run_dashboard(host=args.host, port=args.port, debug=args.debug)
