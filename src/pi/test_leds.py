#!/usr/bin/env python3
"""
Test script for traffic light LEDs on Raspberry Pi 4 Model B
Uses pin mapping from signal_server_v2.py (the authoritative source).

Run on Raspberry Pi:
    python3 test_leds.py

Options:
    python3 test_leds.py --all       # Test all LEDs one by one
    python3 test_leds.py --lane 0    # Test specific lane
    python3 test_leds.py --cycle     # Simulate traffic cycle
    python3 test_leds.py --emergency # Test emergency mode
"""

import time
import argparse

try:
    import RPi.GPIO as GPIO
    ON_PI = True
except ImportError:
    print("Not running on Raspberry Pi - exiting.")
    exit(1)

# Pin mapping from signal_server_v2.py (authoritative source)
LANES = {
    "NORTH": {"R": 17, "Y": 27, "G": 22, "B": 23},
    "EAST":  {"R": 5,  "Y": 6,  "G": 13, "B": 19},
    "SOUTH": {"R": 12, "Y": 16, "G": 20, "B": 21},
    "WEST":  {"R": 24, "Y": 25, "G": 18, "B": 26},
}
LANE_ORDER = ["NORTH", "EAST", "SOUTH", "WEST"]
COLOR_KEYS = {"red": "R", "yellow": "Y", "green": "G", "blue": "B"}


def gpio_init():
    """Initialize all GPIO pins."""
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for lane in LANES.values():
        for pin in lane.values():
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)


def set_pin(lane_name, color_key, value):
    """Set a single LED on/off. color_key: 'R','Y','G','B'."""
    pin = LANES[lane_name][color_key]
    GPIO.output(pin, GPIO.HIGH if value else GPIO.LOW)


def all_off():
    """Turn off all LEDs."""
    for lane in LANES.values():
        for pin in lane.values():
            GPIO.output(pin, GPIO.LOW)


def all_red():
    """Set all lanes to red only."""
    for lane in LANES.values():
        GPIO.output(lane["G"], GPIO.LOW)
        GPIO.output(lane["Y"], GPIO.LOW)
        GPIO.output(lane["B"], GPIO.LOW)
        GPIO.output(lane["R"], GPIO.HIGH)


def set_lane_state(lane_name, state, blue_on=False):
    """Set a lane to red/yellow/green + optional blue."""
    lane = LANES[lane_name]
    GPIO.output(lane["R"], GPIO.HIGH if state == "red" else GPIO.LOW)
    GPIO.output(lane["Y"], GPIO.HIGH if state == "yellow" else GPIO.LOW)
    GPIO.output(lane["G"], GPIO.HIGH if state == "green" else GPIO.LOW)
    GPIO.output(lane["B"], GPIO.HIGH if blue_on else GPIO.LOW)


def set_traffic_state(green_lane_name, emergency=False):
    """Set complete intersection state: one lane green, others red."""
    for name in LANE_ORDER:
        if name == green_lane_name:
            set_lane_state(name, "green", blue_on=False)
        else:
            set_lane_state(name, "red", blue_on=emergency)


def test_all_leds():
    """Test each LED individually."""
    print("\n=== Testing All LEDs ===\n")

    for name in LANE_ORDER:
        lane = LANES[name]
        for color, key in COLOR_KEYS.items():
            pin = lane[key]
            print(f"  {name} {color.upper():6} (GPIO {pin})")
            GPIO.output(pin, GPIO.HIGH)
            time.sleep(0.4)
            GPIO.output(pin, GPIO.LOW)

    print("\n✓ All LEDs tested!")


def test_lane(lane_idx):
    """Test a specific lane's LEDs."""
    name = LANE_ORDER[lane_idx]
    lane = LANES[name]
    print(f"\n=== Testing Lane {lane_idx} ({name}) ===\n")

    for color, key in COLOR_KEYS.items():
        pin = lane[key]
        print(f"  {color.upper():6} (GPIO {pin})")
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(pin, GPIO.LOW)

    print(f"\n✓ Lane {lane_idx} ({name}) tested!")


def test_cycle():
    """Simulate a complete traffic cycle."""
    print("\n=== Simulating Traffic Cycle ===")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            for idx, name in enumerate(LANE_ORDER):
                print(f"Lane {idx} ({name}): GREEN (5s)")
                set_traffic_state(name, emergency=False)
                time.sleep(5)

                print(f"Lane {idx} ({name}): YELLOW (2s)")
                set_lane_state(name, "yellow")
                time.sleep(2)

                all_red()
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\nStopping...")


def test_emergency():
    """Test emergency mode."""
    print("\n=== Testing Emergency Mode ===")
    print("Press Ctrl+C to stop\n")

    try:
        for idx, name in enumerate(LANE_ORDER):
            print(f"\n🚨 Emergency on Lane {idx} ({name})")
            print(f"   {name}: GREEN + BLUE")
            print(f"   Other lanes: RED + BLUE")

            # Emergency lane green+blue, others red+blue
            for other in LANE_ORDER:
                if other == name:
                    set_lane_state(other, "green", blue_on=True)
                else:
                    set_lane_state(other, "red", blue_on=True)
            time.sleep(5)

        all_red()
        print("\n✓ Emergency mode tested!")
    except KeyboardInterrupt:
        print("\n\nStopping...")


def main():
    parser = argparse.ArgumentParser(description="Test traffic light LEDs")
    parser.add_argument('--all', action='store_true', help='Test all LEDs')
    parser.add_argument('--lane', type=int, choices=[0, 1, 2, 3], help='Test specific lane')
    parser.add_argument('--cycle', action='store_true', help='Simulate traffic cycle')
    parser.add_argument('--emergency', action='store_true', help='Test emergency mode')
    args = parser.parse_args()

    print("=" * 50)
    print("  Traffic Light LED Test - Raspberry Pi 4")
    print("  Pin mapping: signal_server_v2.py")
    print("=" * 50)

    gpio_init()

    try:
        if args.lane is not None:
            test_lane(args.lane)
        elif args.cycle:
            test_cycle()
        elif args.emergency:
            test_emergency()
        else:
            test_all_leds()
    finally:
        all_off()
        GPIO.cleanup()
        print("[GPIO] Cleanup complete")


if __name__ == '__main__':
    main()
