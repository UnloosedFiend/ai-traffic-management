#!/usr/bin/env python3
"""
Send traffic signal commands to Raspberry Pi.

Usage:
    python send_signal.py                      # Default: NORTH, no emergency
    python send_signal.py --lane EAST          # Set East lane green
    python send_signal.py --lane SOUTH --emergency  # Emergency mode
    python send_signal.py --ip 192.168.1.100   # Specify Pi IP
"""

import argparse
import requests


def send_signal(pi_ip: str, lane: str, emergency: bool = False, port: int = 5000):
    """
    Send signal command to Raspberry Pi.
    
    Args:
        pi_ip: Raspberry Pi IP address
        lane: Lane name (NORTH, EAST, SOUTH, WEST)
        emergency: Whether this is an emergency override
        port: Server port (default 5000)
    
    Returns:
        Response JSON or error message
    """
    url = f"http://{pi_ip}:{port}/set_signal"
    
    data = {
        "lane": lane.upper(),
        "emergency": emergency
    }
    
    try:
        response = requests.post(url, json=data, timeout=5)
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": f"Cannot connect to Pi at {pi_ip}:{port}"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out"}
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Send traffic signal commands to Raspberry Pi")
    parser.add_argument("--ip", default="127.0.0.1", 
                        help="Raspberry Pi IP address")
    parser.add_argument("--lane", default="NORTH", 
                        choices=["NORTH", "EAST", "SOUTH", "WEST"],
                        help="Lane to set green")
    parser.add_argument("--emergency", action="store_true",
                        help="Enable emergency mode")
    parser.add_argument("--port", type=int, default=5000,
                        help="Server port (default: 5000)")
    
    args = parser.parse_args()
    
    result = send_signal(args.ip, args.lane, args.emergency, args.port)
    print(result)


if __name__ == "__main__":
    main()
