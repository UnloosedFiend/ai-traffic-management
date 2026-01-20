"""
Traffic timing logic for 4-lane junction with fail-safe round-robin.

This module implements:
1. Priority-based lane selection (density + emergency)
2. Fail-safe round-robin when detection is unavailable
3. Emergency vehicle preemption with confirmation
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from enum import Enum


class TrafficMode(Enum):
    """Operating mode for traffic control"""
    NORMAL = "normal"           # Detection-based priority
    EMERGENCY = "emergency"     # Emergency vehicle preemption
    FAILSAFE = "failsafe"       # Round-robin fallback


@dataclass
class LaneState:
    """State of a single lane"""
    vehicle_count: int = 0
    ambulance_count: int = 0
    police_count: int = 0
    emergency_confirmed: bool = False
    emergency_confirm_counter: int = 0
    last_green_time: float = 0.0
    camera_ok: bool = True
    detection_ok: bool = True


@dataclass
class TrafficController:
    """
    Main traffic control logic with fail-safe round-robin.
    
    Features:
    - Density-based priority when detection is working
    - Emergency vehicle preemption with confirmation
    - Automatic round-robin failsafe when detection fails
    - Fair scheduling to prevent lane starvation
    """
    
    num_lanes: int = 4
    min_green: int = 5
    max_green: int = 30
    base_green: int = 10
    emergency_green: int = 30
    failsafe_cycle: int = 15
    emergency_confirm_required: int = 3
    emergency_cooldown: float = 10.0
    
    # Internal state
    lanes: Dict[int, LaneState] = field(default_factory=dict)
    current_lane: int = 0
    mode: TrafficMode = TrafficMode.NORMAL
    last_emergency_time: float = 0.0
    round_robin_index: int = 0
    cycle_count: int = 0
    
    def __post_init__(self):
        """Initialize lane states"""
        self.lanes = {i: LaneState() for i in range(self.num_lanes)}
    
    def update_lane(self, lane_id: int, vehicle_count: int, ambulance_count: int, 
                    police_count: int, camera_ok: bool = True, detection_ok: bool = True):
        """
        Update lane state with latest detection results.
        
        Args:
            lane_id: Lane index (0-3)
            vehicle_count: Number of normal vehicles detected
            ambulance_count: Number of ambulances detected
            police_count: Number of police vehicles detected
            camera_ok: Whether camera feed is available
            detection_ok: Whether detection ran successfully
        """
        if lane_id not in self.lanes:
            return
        
        lane = self.lanes[lane_id]
        lane.vehicle_count = vehicle_count
        lane.ambulance_count = ambulance_count
        lane.police_count = police_count
        lane.camera_ok = camera_ok
        lane.detection_ok = detection_ok
        
        # Emergency confirmation logic - require consecutive detections
        emergency_detected = (ambulance_count > 0 or police_count > 0)
        
        if emergency_detected:
            lane.emergency_confirm_counter += 1
            if lane.emergency_confirm_counter >= self.emergency_confirm_required:
                lane.emergency_confirmed = True
        else:
            # Reset counter if no emergency in this frame
            lane.emergency_confirm_counter = 0
            lane.emergency_confirmed = False
    
    def _check_system_health(self) -> bool:
        """Check if enough cameras/detection is working for normal operation"""
        working_lanes = sum(
            1 for lane in self.lanes.values() 
            if lane.camera_ok and lane.detection_ok
        )
        # Need at least 2 working lanes for normal operation
        return working_lanes >= 2
    
    def _get_emergency_lane(self) -> Optional[int]:
        """Get lane with confirmed emergency vehicle, respecting cooldown"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_emergency_time < self.emergency_cooldown:
            return None
        
        for lane_id, lane in self.lanes.items():
            if lane.emergency_confirmed:
                return lane_id
        
        return None
    
    def _get_priority_lane(self) -> Tuple[int, int]:
        """
        Select lane based on traffic density with fairness.
        
        Returns:
            (lane_id, green_duration)
        """
        # Calculate priority scores
        scores = {}
        current_time = time.time()
        
        for lane_id, lane in self.lanes.items():
            if not lane.camera_ok or not lane.detection_ok:
                # Skip broken lanes for priority calculation
                continue
            
            # Base score from vehicle count
            score = lane.vehicle_count + (lane.ambulance_count * 2) + (lane.police_count * 2)
            
            # Fairness bonus - lanes waiting longer get priority boost
            wait_time = current_time - lane.last_green_time
            fairness_bonus = min(wait_time / 30.0, 1.0) * 5  # Up to 5 bonus points
            
            scores[lane_id] = score + fairness_bonus
        
        if not scores:
            # All lanes broken, use round-robin
            return self._round_robin_next()
        
        # Select highest score
        selected_lane = max(scores, key=lambda k: scores[k])
        
        # Calculate green time based on density
        lane = self.lanes[selected_lane]
        total_vehicles = lane.vehicle_count + lane.ambulance_count + lane.police_count
        green_time = self.base_green + (total_vehicles * 2)
        green_time = max(self.min_green, min(green_time, self.max_green))
        
        return selected_lane, green_time
    
    def _round_robin_next(self) -> Tuple[int, int]:
        """Get next lane in round-robin sequence"""
        lane_id = self.round_robin_index
        self.round_robin_index = (self.round_robin_index + 1) % self.num_lanes
        return lane_id, self.failsafe_cycle
    
    def decide(self) -> Tuple[int, int, TrafficMode]:
        """
        Main decision function - determines which lane gets green.
        
        Returns:
            (lane_id, green_duration, mode)
        """
        self.cycle_count += 1
        
        # Check system health
        if not self._check_system_health():
            self.mode = TrafficMode.FAILSAFE
            lane_id, duration = self._round_robin_next()
            return lane_id, duration, self.mode
        
        # Check for emergency
        emergency_lane = self._get_emergency_lane()
        if emergency_lane is not None:
            self.mode = TrafficMode.EMERGENCY
            self.last_emergency_time = time.time()
            self.lanes[emergency_lane].last_green_time = time.time()
            # Clear emergency confirmation after granting green
            self.lanes[emergency_lane].emergency_confirmed = False
            self.lanes[emergency_lane].emergency_confirm_counter = 0
            return emergency_lane, self.emergency_green, self.mode
        
        # Normal priority-based selection
        self.mode = TrafficMode.NORMAL
        lane_id, duration = self._get_priority_lane()
        self.lanes[lane_id].last_green_time = time.time()
        
        return lane_id, duration, self.mode
    
    def get_status(self) -> dict:
        """Get current system status for logging/display"""
        return {
            "mode": self.mode.value,
            "current_lane": self.current_lane,
            "cycle_count": self.cycle_count,
            "lanes": {
                lane_id: {
                    "vehicles": lane.vehicle_count,
                    "ambulances": lane.ambulance_count,
                    "police": lane.police_count,
                    "emergency_confirmed": lane.emergency_confirmed,
                    "emergency_counter": lane.emergency_confirm_counter,
                    "camera_ok": lane.camera_ok,
                    "detection_ok": lane.detection_ok,
                }
                for lane_id, lane in self.lanes.items()
            }
        }


# Legacy function for backward compatibility
def decide_lane(lane_data: dict) -> Tuple[int, int, bool]:
    """
    Legacy interface for simple lane decision.
    
    Args:
        lane_data: {lane_id: {"count": int, "emergency": bool}, ...}
    
    Returns:
        (lane_id, green_duration, emergency)
    """
    MIN_GREEN = 5
    MAX_GREEN = 30
    BASE_GREEN = 10
    
    # Emergency preemption
    for lane_id, data in lane_data.items():
        if data.get("emergency", False):
            return lane_id, MAX_GREEN, True
    
    # Density-based selection
    selected_lane = max(lane_data, key=lambda k: lane_data[k].get("count", 0))
    
    # Compute green time
    green_time = BASE_GREEN + lane_data[selected_lane].get("count", 0)
    green_time = max(MIN_GREEN, min(green_time, MAX_GREEN))
    
    return selected_lane, green_time, False

