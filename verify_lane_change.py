from kalman_filter import KalmanFilter
import numpy as np

# Mock of the get_curve_status logic
def mock_get_curve_status(vp_x, lane_center_x, lane_width_px, lateral_velocity):
    if abs(lateral_velocity) > 1.5:
        return "Straight", 0
    
    offset = vp_x - lane_center_x
    if offset > 50: return "Curve Left", offset
    if offset < -50: return "Curve Right", offset
    return "Straight", offset

def test_lane_change_logic():
    print("Testing Lane Change Logic...")
    
    # 1. Normal Curve Scenario
    # VP is far right of center (Offset -100), Zero Velocity
    status, _ = mock_get_curve_status(640, 740, 600, 0.0)
    print(f"Scenario 1 (Curve Right): status='{status}' (Expected: Curve Right)")
    
    # 2. Lane Change Scenario
    # Same visual offset (-100), but High Lateral Velocity (2.0)
    status, _ = mock_get_curve_status(640, 740, 600, 2.0)
    print(f"Scenario 2 (Lane Change): status='{status}' (Expected: Straight)")
    
    if status == "Straight":
        print("SUCCESS: Lane change suppressed curve warning.")
    else:
        print("FAILURE: Lane change did NOT suppress curve warning.")

if __name__ == "__main__":
    test_lane_change_logic()
