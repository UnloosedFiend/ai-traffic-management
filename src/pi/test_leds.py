#!/usr/bin/env python3
"""
Test script for traffic light LEDs on Raspberry Pi 4 Model B

Run on Raspberry Pi:
    python3 test_leds.py

Options:
    python3 test_leds.py --all      # Test all LEDs one by one
    python3 test_leds.py --lane 0   # Test specific lane
    python3 test_leds.py --cycle    # Simulate traffic cycle
    python3 test_leds.py --emergency # Test emergency mode
"""

import time
import argparse
import os

# Force real GPIO on Pi
os.environ['USE_GPIO_MOCK'] = '0'

try:
    import RPi.GPIO as GPIO
    ON_PI = True
except ImportError:
    print("Not running on Raspberry Pi - using mock mode")
    os.environ['USE_GPIO_MOCK'] = '1'
    ON_PI = False

from gpio_ctrl import GPIOController, PIN_MAP


def test_all_leds(gpio):
    """Test each LED individually"""
    print("\n=== Testing All LEDs ===\n")
    
    for lane in range(4):
        for color in ['red', 'yellow', 'green', 'blue']:
            key = f'lane{lane}_{color}'
            pin = PIN_MAP.get(key, '?')
            print(f"  Lane {lane} {color.upper():6} (GPIO {pin})")
            gpio.set_lane(lane, color, True)
            time.sleep(0.4)
            gpio.set_lane(lane, color, False)
    
    print("\n✓ All LEDs tested!")


def test_lane(gpio, lane_id):
    """Test a specific lane's LEDs"""
    print(f"\n=== Testing Lane {lane_id} ===\n")
    
    for color in ['red', 'yellow', 'green', 'blue']:
        key = f'lane{lane_id}_{color}'
        pin = PIN_MAP.get(key, '?')
        print(f"  {color.upper():6} (GPIO {pin})")
        gpio.set_lane(lane_id, color, True)
        time.sleep(0.5)
        gpio.set_lane(lane_id, color, False)
    
    print(f"\n✓ Lane {lane_id} tested!")


def test_cycle(gpio):
    """Simulate a complete traffic cycle"""
    print("\n=== Simulating Traffic Cycle ===")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            for green_lane in range(4):
                # Green phase
                print(f"Lane {green_lane}: GREEN (5s)")
                gpio.set_traffic_state(green_lane, emergency=False)
                time.sleep(5)
                
                # Yellow phase
                print(f"Lane {green_lane}: YELLOW (2s)")
                gpio.set_lane_state(green_lane, 'yellow')
                time.sleep(2)
                
                # All red briefly
                gpio.all_red()
                time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n\nStopping...")


def test_emergency(gpio):
    """Test emergency mode"""
    print("\n=== Testing Emergency Mode ===")
    print("Press Ctrl+C to stop\n")
    
    try:
        for emergency_lane in range(4):
            print(f"\n🚨 Emergency on Lane {emergency_lane}")
            print(f"   Lane {emergency_lane}: GREEN")
            print(f"   Other lanes: RED + BLUE")
            
            gpio.set_traffic_state(emergency_lane, emergency=True)
            time.sleep(5)
        
        gpio.all_red()
        print("\n✓ Emergency mode tested!")
    
    except KeyboardInterrupt:
        print("\n\nStopping...")


def main():
    parser = argparse.ArgumentParser(description="Test traffic light LEDs")
    parser.add_argument('--all', action='store_true', help='Test all LEDs')
    parser.add_argument('--lane', type=int, choices=[0,1,2,3], help='Test specific lane')
    parser.add_argument('--cycle', action='store_true', help='Simulate traffic cycle')
    parser.add_argument('--emergency', action='store_true', help='Test emergency mode')
    args = parser.parse_args()
    
    print("=" * 50)
    print("  Traffic Light LED Test - Raspberry Pi 4")
    print("=" * 50)
    
    gpio = GPIOController()
    
    try:
        if args.lane is not None:
            test_lane(gpio, args.lane)
        elif args.cycle:
            test_cycle(gpio)
        elif args.emergency:
            test_emergency(gpio)
        else:
            # Default: test all
            test_all_leds(gpio)
    
    finally:
        gpio.cleanup()


if __name__ == '__main__':
    main()
