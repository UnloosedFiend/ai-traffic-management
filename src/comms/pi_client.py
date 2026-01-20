"""
Raspberry Pi communication client for traffic signal control.

Sends lane commands to the Pi which controls physical LEDs via GPIO.
"""

import requests
import time
from dataclasses import dataclass
from typing import Optional


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
        self.status_url = f"{self.base_url}/status"
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.status = PiStatus()
    
    def send(self, lane: int, duration: int, emergency: bool = False) -> bool:
        """
        Send signal command to Raspberry Pi.
        
        Args:
            lane: Lane ID to set green (0-3)
            duration: Green light duration in seconds
            emergency: Whether this is an emergency override
        
        Returns:
            True if command was sent successfully
        
        Signal Light Behavior:
            Normal mode:
                - Green lane: GREEN light
                - Other lanes: RED light
            
            Emergency mode:
                - Emergency lane: GREEN light (let emergency vehicle pass)
                - Other lanes: RED + BLUE lights (stopped, blue shows emergency approaching)
        """
        payload = {
            "lane": int(lane),
            "green_duration": int(duration),
            "emergency": bool(emergency),
            "blue_on_stopped": bool(emergency)  # Turn on blue lights on stopped lanes
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
        
        Returns:
            True if command was sent successfully
        """
        payload = {
            "lane": -1,  # Special value for all-red
            "green_duration": 0,
            "emergency": False,
            "all_red": True
        }
        
        try:
            response = requests.post(
                self.signal_url,
                json=payload,
                timeout=self.timeout
            )
            return response.ok
        except requests.exceptions.RequestException:
            return False
    
    def check_connection(self) -> bool:
        """
        Check if Raspberry Pi is reachable.
        
        Returns:
            True if Pi responds to status check
        """
        try:
            response = requests.get(self.status_url, timeout=self.timeout)
            if response.ok:
                self.status.connected = True
                return True
        except requests.exceptions.RequestException:
            pass
        
        self.status.connected = False
        return False
    
    def get_status(self) -> PiStatus:
        """Get current connection status"""
        return self.status
    
    def is_healthy(self) -> bool:
        """Check if Pi connection is healthy"""
        return self.status.connected and self.status.consecutive_failures < 3

