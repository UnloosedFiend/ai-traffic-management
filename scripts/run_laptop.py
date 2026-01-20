#!/usr/bin/env python3
"""
Quick launcher for AI Traffic Management System on laptop/desktop.

This is a convenience script that starts the main application.

Usage:
    python scripts/run_laptop.py
    python scripts/run_laptop.py --show-video --debug
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.app import main

if __name__ == "__main__":
    main()
