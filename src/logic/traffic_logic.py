# Traffic timing logic for 4-lane junction

MIN_GREEN = 5
MAX_GREEN = 30
BASE_GREEN = 10


def decide_lane(lane_data):
    """
    lane_data format:
    {
        0: {"count": int, "emergency": bool},
        1: {"count": int, "emergency": bool},
        2: {"count": int, "emergency": bool},
        3: {"count": int, "emergency": bool}
    }

    Returns:
        lane_id (int)
        green_duration (int)
        emergency (bool)
    """

    # 1. Emergency preemption (highest priority)
    for lane_id, data in lane_data.items():
        if data["emergency"]:
            return lane_id, MAX_GREEN, True

    # 2. Density-based selection
    selected_lane = max(lane_data, key=lambda k: lane_data[k]["count"])

    # 3. Compute green time proportional to traffic
    green_time = BASE_GREEN + lane_data[selected_lane]["count"]

    # 4. Clamp to safe bounds
    green_time = max(MIN_GREEN, min(green_time, MAX_GREEN))

    return selected_lane, green_time, False
