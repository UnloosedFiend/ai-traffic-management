# src/pi/gpio_ctrl.py
"""
GPIO Controller for Raspberry Pi 4 Model B
Controls 4 lanes × 4 lights (RED, YELLOW, GREEN, BLUE)
"""
import os

# GPIO BCM pin mapping for 4 lanes × 4 lights
# See docs/wiring.md for full circuit diagram
PIN_MAP = {
    # Lane 0 (North)
    'lane0_red': 17,
    'lane0_yellow': 27,
    'lane0_green': 22,
    'lane0_blue': 5,
    
    # Lane 1 (East)
    'lane1_red': 6,
    'lane1_yellow': 13,
    'lane1_green': 19,
    'lane1_blue': 26,
    
    # Lane 2 (South)
    'lane2_red': 12,
    'lane2_yellow': 16,
    'lane2_green': 20,
    'lane2_blue': 21,
    
    # Lane 3 (West)
    'lane3_red': 4,
    'lane3_yellow': 18,
    'lane3_green': 23,
    'lane3_blue': 24,
}

USE_MOCK = os.environ.get('USE_GPIO_MOCK', '1') == '1'

if USE_MOCK:
    class GPIOController:
        """Mock GPIO controller for development/testing"""
        def __init__(self, mapping=None):
            self.mapping = mapping or PIN_MAP
            self.state = {k: False for k in self.mapping}
            print('[GPIO MOCK] Initialized with pins:', list(self.mapping.keys()))
        
        def set(self, key, value: bool):
            if key in self.state:
                self.state[key] = bool(value)
                # Only print state changes to reduce noise
        
        def set_lane(self, lane_id: int, color: str, value: bool):
            """Set a specific light on a lane"""
            key = f'lane{lane_id}_{color}'
            self.set(key, value)
        
        def set_lane_state(self, lane_id: int, state: str, blue_on: bool = False):
            """
            Set complete state for a lane.
            state: 'red', 'yellow', 'green'
            blue_on: True to turn on blue emergency indicator
            """
            self.set_lane(lane_id, 'red', state == 'red')
            self.set_lane(lane_id, 'yellow', state == 'yellow')
            self.set_lane(lane_id, 'green', state == 'green')
            self.set_lane(lane_id, 'blue', blue_on)
            print(f'[GPIO MOCK] Lane {lane_id}: {state.upper()}{" + BLUE" if blue_on else ""}')
        
        def set_traffic_state(self, green_lane: int, emergency: bool = False):
            """
            Set complete traffic state.
            green_lane: which lane gets green (0-3)
            emergency: if True, blue lights on RED lanes
            """
            for lane in range(4):
                if lane == green_lane:
                    self.set_lane_state(lane, 'green', blue_on=False)
                else:
                    self.set_lane_state(lane, 'red', blue_on=emergency)
        
        def all_red(self):
            """Set all lanes to red (safe state)"""
            for lane in range(4):
                self.set_lane_state(lane, 'red', blue_on=False)
            print('[GPIO MOCK] ALL RED')
        
        def cleanup(self):
            self.all_red()
            print('[GPIO MOCK] Cleanup complete')

else:
    import RPi.GPIO as GPIO
    
    class GPIOController:
        """Real GPIO controller for Raspberry Pi"""
        def __init__(self, mapping=None):
            self.mapping = mapping or PIN_MAP
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # Setup all pins as output, initially LOW
            for key, pin in self.mapping.items():
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)
            
            print(f'[GPIO] Initialized {len(self.mapping)} pins')
        
        def set(self, key, value: bool):
            """Set a single GPIO pin"""
            if key in self.mapping:
                GPIO.output(self.mapping[key], GPIO.HIGH if value else GPIO.LOW)
        
        def set_lane(self, lane_id: int, color: str, value: bool):
            """Set a specific light on a lane"""
            key = f'lane{lane_id}_{color}'
            self.set(key, value)
        
        def set_lane_state(self, lane_id: int, state: str, blue_on: bool = False):
            """
            Set complete state for a lane.
            state: 'red', 'yellow', 'green'
            blue_on: True to turn on blue emergency indicator
            """
            self.set_lane(lane_id, 'red', state == 'red')
            self.set_lane(lane_id, 'yellow', state == 'yellow')
            self.set_lane(lane_id, 'green', state == 'green')
            self.set_lane(lane_id, 'blue', blue_on)
        
        def set_traffic_state(self, green_lane: int, emergency: bool = False):
            """
            Set complete traffic state.
            green_lane: which lane gets green (0-3)
            emergency: if True, blue lights on RED lanes
            """
            for lane in range(4):
                if lane == green_lane:
                    self.set_lane_state(lane, 'green', blue_on=False)
                else:
                    self.set_lane_state(lane, 'red', blue_on=emergency)
        
        def all_red(self):
            """Set all lanes to red (safe state)"""
            for lane in range(4):
                self.set_lane_state(lane, 'red', blue_on=False)
        
        def cleanup(self):
            """Cleanup GPIO on exit"""
            self.all_red()
            GPIO.cleanup()
            print('[GPIO] Cleanup complete')
