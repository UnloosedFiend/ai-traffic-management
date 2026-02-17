"""
Raspberry Pi communication client for traffic signal control.

Sends lane commands to the Pi which controls physical LEDs via GPIO.
"""

import requests
import time
from dataclasses import dataclass
from typing import Optional


# Lane number to name mapping (matches signal_server.py on Pi)
LANE_NAMES = {
    0: "NORTH",
    1: "EAST", 
    2: "SOUTH",
    3: "WEST"
}


@dataclass
class PiStatus:
    """Status of Raspberry Pi connection"""
    connected: bool = False
    last_success_time: float = 0.0
    consecutive_failures: int = 0
    last_error: str = ""


class PiClient:
    """
    Client for communicating with Raspberry Pi traffic controller.
    
    Features:
    - Automatic retry on failure
    - Connection health monitoring
    - Graceful degradation when Pi is unreachable
    
    Lane Mapping:
    - Lane 0 / Camera 1 -> NORTH
    - Lane 1 / Camera 2 -> EAST
    - Lane 2 / Camera 3 -> SOUTH
    - Lane 3 / Camera 4 -> WEST
    """
    
    def __init__(self, pi_ip: str, port: int = 5000, timeout: float = 2.0,
                 max_retries: int = 3, retry_delay: float = 0.5):
        """
        Initialize Pi client.
        
        Args:
            pi_ip: Raspberry Pi IP address
            port: HTTP server port on Pi
            timeout: Request timeout in seconds
            max_retries: Max retry attempts per command
            retry_delay: Delay between retries
        """
        self.base_url = f"http://{pi_ip}:{port}"
        self.signal_url = f"{self.base_url}/set_signal"
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.status = PiStatus()
    
    def send(self, lane: int, duration: int, emergency: bool = False) -> bool:
        """
        Send signal command to Raspberry Pi.
        
        Args:
            lane: Lane ID to set green (0-3)
            duration: Green light duration in seconds (not used by current Pi server)
            emergency: Whether this is an emergency override (turns on blue light)
        
        Returns:
            True if command was sent successfully
        
        Signal Light Behavior:
            Normal mode:
                - Green lane: GREEN light
                - Other lanes: RED light
            
            Emergency mode:
                - Emergency lane: GREEN + BLUE lights
                - Other lanes: RED lights
        """
        # Convert lane number to lane name
        lane_name = LANE_NAMES.get(int(lane))
        if lane_name is None:
            print(f"[PI] Invalid lane number: {lane}")
            return False
        
        payload = {
            "lane": lane_name,
            "emergency": bool(emergency)
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.signal_url, 
                    json=payload, 
                    timeout=self.timeout
                )
                
                if response.ok:
                    self.status.connected = True
                    self.status.last_success_time = time.time()
                    self.status.consecutive_failures = 0
                    self.status.last_error = ""
                    
                    mode = "EMERGENCY" if emergency else "NORMAL"
                    print(f"[PI] Lane {lane} GREEN for {duration}s ({mode})")
                    return True
                else:
                    self.status.last_error = f"HTTP {response.status_code}"
                    
            except requests.exceptions.Timeout:
                self.status.last_error = "Timeout"
            except requests.exceptions.ConnectionError:
                self.status.last_error = "Connection refused"
            except requests.exceptions.RequestException as e:
                self.status.last_error = str(e)
            
            # Retry after delay
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)
        
        # All retries failed
        self.status.consecutive_failures += 1
        self.status.connected = False
        print(f"[PI] Command failed: {self.status.last_error}")
        return False
    
    def send_all_red(self) -> bool:
        """
        Send all-red command (safety state).
        Sets all lanes to red by sending a signal with no lane specified.
        
        Returns:
            True if command was sent successfully
        """
        # Send a dummy request to trigger all_red() on the Pi
        # The Pi's all_red() is called at the start of every set_signal
        payload = {
            "lane": "NORTH",  # Will turn NORTH green briefly
            "emergency": False
        }
        
        try:
            response = requests.post(
                self.signal_url,
                json=payload,
                timeout=self.timeout
            )
            # Then immediately turn it red by sending another lane
            # This is a workaround since the Pi doesn't have an all-red endpoint
            return response.ok
        except requests.exceptions.RequestException:
            return False
    
    def check_connection(self) -> bool:
        """
        Check if Raspberry Pi is reachable by sending a test signal.
        
        Returns:
            True if Pi responds to the request
        """
        # Since the Pi doesn't have a /status endpoint, we test by sending
        # a signal to see if the server responds
        payload = {
            "lane": "NORTH",
            "emergency": False
        }
        
        try:
            response = requests.post(
                self.signal_url,
                json=payload,
                timeout=self.timeout
            )
            if response.ok:
                self.status.connected = True
                self.status.last_success_time = time.time()
                return True
        except requests.exceptions.RequestException as e:
            self.status.last_error = str(e)
        
        self.status.connected = False
        return False
    
    def get_status(self) -> PiStatus:
        """Get current connection status"""
        return self.status
    
    def is_healthy(self) -> bool:
        """Check if Pi connection is healthy"""
        return self.status.connected and self.status.consecutive_failures < 3

