import cv2
import numpy as np
import time
import os
import csv
import math
from datetime import datetime
from tracker import LaneTracker, LANE_WIDTH_PX
from sensor_bridge import SensorBridge
from start_screen import StartScreen
from gps_utils import GPSCurvatureEstimator


# --- CONFIGURATION ---
VIDEO_PATH = "drive.mp4" # Set to None or "" to use the live webcam feed
HISTORY_LENGTH = 10 
USE_SENSORS = False # Overridden by UI 
USE_WEBCAM = False # Overridden by UI 

SAFE_ROAD_WIDTH = 800 # Adjusted for narrow Kerala roads

# Display Settings
DISPLAY_WIDTH = 1280  # Output window width
DISPLAY_HEIGHT = 720  # Output window height

def region_of_interest(image, vp_x=None):
    height = image.shape[0]
    width = image.shape[1]
    
    # Dynamic Apex: Use VP if available, else Center
    apex_x = vp_x if vp_x is not None else width // 2
    
    # Clamp apex to screen bounds (optional safety)
    apex_x = max(0, min(width, apex_x))

    polygons = np.array([
        [
            (0, height), 
            (apex_x, int(height * 0.55)), # Horizon (Dynamic)
            (width, height) 
        ]
    ])
    
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def detect_lane_pixels(frame, gray=None):
    """
    Detect white and yellow lane marking pixels using color filtering.
    Returns a binary image with lane pixels highlighted.
    """
    # Convert to HLS color space
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    
    # Define range for white color
    lower_white = np.array([0, 200, 0])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(hls, lower_white, upper_white)
    
    # Define range for yellow color (Kerala roads often have yellow center lines)
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)
    
    # Also use grayscale thresholding as backup
    if gray is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, gray_thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Combine masks: White OR Yellow OR Grayscale
    color_mask = cv2.bitwise_or(white_mask, yellow_mask)
    combined = cv2.bitwise_or(color_mask, gray_thresh)
    
    return combined

def find_lane_boundaries(binary_lane_img, search_offset=0):
    """
    Use sliding window to find left and right lane boundaries.
    search_offset: Shift the midpoint split (Positive = Look more to right)
    """
    height, width = binary_lane_img.shape
    
    # Take histogram of bottom half
    histogram = np.sum(binary_lane_img[height//2:, :], axis=0)
    
    # Find peaks for left and right lanes
    midpoint = (width // 2) + search_offset
    
    # Clamp midpoint to prevent argmax on empty sequences
    midpoint = max(10, min(width - 10, midpoint))
    
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # If no strong peaks found, use defaults
    if histogram[left_base] < 100:
        left_base = width // 4
    if histogram[right_base] < 100:
        right_base = 3 * width // 4
    
    return left_base, right_base

# --- LOGIC & VISUALIZATION ---

def generate_cubic_bezier_points(p0, p1, p2, p3, num_points=20, cutoff=0.9):
    """
    Generates points for a Cubic Bezier curve (4 points).
    This creates a 'whip' effect where the base stays straight and the top bends.
    """
    t = np.linspace(0, cutoff, num_points)
    x = ((1-t)**3 * p0[0] + 3*(1-t)**2 * t * p1[0] + 3*(1-t) * t**2 * p2[0] + t**3 * p3[0]).astype(int)
    y = ((1-t)**3 * p0[1] + 3*(1-t)**2 * t * p1[1] + 3*(1-t) * t**2 * p2[1] + t**3 * p3[1]).astype(int)
    return list(zip(x, y))

# Professional ADAS-style Display
# Professional Minimalist AR Display
def draw_minimalist_hud(image, speed, status, optimal_speed, left_pts, right_pts, vp_coord, lane_width_px):
    height, width = image.shape[:2]
    vp_x, vp_y = vp_coord
    
    # --- 1. THE AR LANE (Wireframe Grid) ---
    # Instead of a solid carpet, we draw a "Ladder" or "Grid"
    overlay = image.copy()
    
    # Create the polygon points
    poly_points = left_pts + right_pts[::-1]
    
    # A. The Subtle Fill (Very transparent)
    # We use a much lower alpha (0.2) so you can see potholes through it
    color_fill = (0, 255, 100) # Cyber Green
    if "Mild" in status: color_fill = (0, 255, 255) # Yellow
    elif "Curve" in status: color_fill = (0, 165, 255) # Amber
    if "Sharp" in status: color_fill = (0, 0, 255)   # Red
    
    cv2.fillPoly(overlay, [np.array(poly_points, dtype=np.int32)], color_fill)
    cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image) # 20% opacity
    
    # B. The "Ladder" Effect (Horizontal rungs)
    # Draw lines connecting left and right boundaries every 10th point
    # This gives the "3D Terrain" look
    num_pts = len(left_pts)
    step = 4 # Draw a rung every 4 points
    
    for i in range(0, num_pts, step):
        pt_l = left_pts[i]
        pt_r = right_pts[i]
        # Make lines thinner as they get further away (depth perception)
        thickness = 2 if i > num_pts // 2 else 1 
        cv2.line(image, pt_l, pt_r, color_fill, thickness, cv2.LINE_AA)

    # C. The Glow Borders
    cv2.polylines(image, [np.array(left_pts, dtype=np.int32)], False, color_fill, 2, cv2.LINE_AA)
    cv2.polylines(image, [np.array(right_pts, dtype=np.int32)], False, color_fill, 2, cv2.LINE_AA)

    # --- 2. THE FLOATING HUD (No Box) ---
    # We place the data near the bottom center (Heads Up style)
    # or "Floating" near the VP.
    
    center_x = width // 2
    hud_y = height - 120 # Just above the hood
    
    # Font Settings
    font = cv2.FONT_HERSHEY_DUPLEX
    
    # A. Speed (Big, Centered)
    speed_text = f"{speed}"
    text_size = cv2.getTextSize(speed_text, font, 2.5, 3)[0]
    text_x = center_x - (text_size[0] // 2)
    
    # Drop Shadow for readability against any road color
    cv2.putText(image, speed_text, (text_x + 2, hud_y + 2), font, 2.5, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(image, speed_text, (text_x, hud_y), font, 2.5, (255, 255, 255), 3, cv2.LINE_AA)
    
    # "km/h" label next to it
    cv2.putText(image, "km/h", (text_x + text_size[0] + 10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1, cv2.LINE_AA)

    # Output Optimal Speed limit to the HUD
    opt_text = f"TARGET: {optimal_speed} km/h"
    opt_size = cv2.getTextSize(opt_text, font, 0.6, 1)[0]
    cv2.putText(image, opt_text, (center_x - (opt_size[0] // 2), hud_y - 80), font, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

    # B. Status / Curve Warning (Below Speed)
    if status != "Straight":
        status_text = status.upper()
        s_size = cv2.getTextSize(status_text, font, 0.8, 2)[0]
        s_x = center_x - (s_size[0] // 2)
        
        # Background pill for status
        cv2.rectangle(image, (s_x - 10, hud_y + 20), (s_x + s_size[0] + 10, hud_y + 55), (0, 0, 0), -1)
        # Text
        cv2.putText(image, status_text, (s_x, hud_y + 45), font, 0.8, color_fill, 1, cv2.LINE_AA)
    
    # C. Dynamic Brackets (The "Target" Lock)
    # Visual cues that hug the lane center
    b_color = (255, 255, 255)
    
    # Left Bracket
    l_x = int(vp_x - lane_width_px * 0.4)
    r_x = int(vp_x + lane_width_px * 0.4)
    
    # Draw simple vertical marks at the horizon line
    cv2.line(image, (l_x, vp_y - 20), (l_x, vp_y + 20), b_color, 1, cv2.LINE_AA)
    cv2.line(image, (r_x, vp_y - 20), (r_x, vp_y + 20), b_color, 1, cv2.LINE_AA)


def main():
    # Pass an unstarted bridge so the UI can enable sensor checkboxes
    # Connection logic runs asynchronously if requested later
    bridge = SensorBridge(port='COM10', baud=115200)

    # Launch Start Screen
    screen = StartScreen(bridge=bridge)
    start_mode, use_sensors, use_webcam, webcam_index = screen.show()
    
    if start_mode == "None":
        if bridge:
            bridge.stop()
        print("Application closed from launcher.")
        return
        
    global USE_SENSORS
    global USE_WEBCAM
    USE_SENSORS = use_sensors
    USE_WEBCAM = use_webcam
    
    # If user chose not to use sensors, shut down the bridge we span up for calib
    if not USE_SENSORS:
        bridge = None
    else:
        print("Connecting to sensors...")
        bridge.start()
        time.sleep(0.5)

    if start_mode == "Pothole":
        import pothole_analysis
        print("Launching Pothole Scanner...")
        pothole_analysis.run_pothole_detection(VIDEO_PATH, bridge, USE_SENSORS, USE_WEBCAM, webcam_index)
        if bridge:
            bridge.stop()
        return

    if USE_WEBCAM:
        print(f"Reading video from: Live Webcam (Index {webcam_index})")
        cap = cv2.VideoCapture(webcam_index)
    elif VIDEO_PATH:
        print(f"Reading video from: {VIDEO_PATH}")
        cap = cv2.VideoCapture(VIDEO_PATH)
    else:
        print(f"No video source selected (Webcam disabled and VIDEO_PATH is None). Attempting webcam {webcam_index} as fallback.")
        cap = cv2.VideoCapture(webcam_index)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        if bridge:
            bridge.stop()
        return

    current_speed = 0 
    
    # Create resizable window
    cv2.namedWindow('SafeTurn+ Main', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('SafeTurn+ Main', DISPLAY_WIDTH, DISPLAY_HEIGHT)
    
    # Initialize Tracker locally
    tracker = LaneTracker()
    gps_estimator = GPSCurvatureEstimator()
    
    consecutive_lost_frames = 0
    tracker_initialized = False # To ignore first-frame jump
    
    last_frame_time = time.perf_counter()
    
    # Track previous state for Teleport Check
    prev_l_bot = 0
    prev_r_bot = 0
    
    # Breadcrumb Logging Setup
    last_log_time = 0
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    breadcrumb_filename = f"route_breadcrumbs_{timestamp_str}.csv"
    breadcrumb_file = None
    crumb_writer = None
    
    # Write header if file doesn't exist yet
    if USE_SENSORS:
        breadcrumb_file = open(breadcrumb_filename, "w", newline="")
        crumb_writer = csv.writer(breadcrumb_file)
        crumb_writer.writerow(["Timestamp", "Latitude", "Longitude", "Speed_kmh", "Yaw_Rate", "Curve"])

    while True:
        ret, frame = cap.read()
        if not ret: 
            print("Video feed ended.")
            break
            
        now = time.perf_counter()
        dt = now - last_frame_time
        last_frame_time = now
        
        # Crop to target aspect ratio before resizing to avoid stretching
        orig_h, orig_w = frame.shape[:2]
        target_ar = DISPLAY_WIDTH / DISPLAY_HEIGHT
        orig_ar = orig_w / orig_h
        
        if orig_ar > target_ar:
            # Crop width
            new_w = int(orig_h * target_ar)
            start_x = (orig_w - new_w) // 2
            frame = frame[:, start_x:start_x+new_w]
        elif orig_ar < target_ar:
            # Crop height
            new_h = int(orig_w / target_ar)
            start_y = (orig_h - new_h) // 2
            frame = frame[start_y:start_y+new_h, :]
            
        # Resize frame to display dimensions
        frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        
        # Performance Optimization: Process at half resolution
        PROC_WIDTH, PROC_HEIGHT = 640, 360
        proc_frame = cv2.resize(frame, (PROC_WIDTH, PROC_HEIGHT))
        
        # Apply blur to color image to benefit both HLS and Gray pipelines
        proc_frame_blurred = cv2.GaussianBlur(proc_frame, (5, 5), 0)
        gray = cv2.cvtColor(proc_frame_blurred, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Dynamic ROI: Focus where the tracker thinks the road is (scaled)
        current_vp_x_proc = int(tracker.avg_vp[0] * PROC_WIDTH / DISPLAY_WIDTH)
        cropped_edges = region_of_interest(edges, vp_x=current_vp_x_proc)

        # Adjust Hough parameters for half resolution
        lines = cv2.HoughLinesP(cropped_edges, 2, np.pi/180, 80, np.array([]), minLineLength=20, maxLineGap=50)
        
        # Scale lines back to DISPLAY_WIDTH
        if lines is not None:
            lines = lines * np.array([DISPLAY_WIDTH/PROC_WIDTH, DISPLAY_HEIGHT/PROC_HEIGHT, 
                                      DISPLAY_WIDTH/PROC_WIDTH, DISPLAY_HEIGHT/PROC_HEIGHT], dtype=np.float32)
            lines = lines.astype(np.int32)
            
        width = frame.shape[1]
        height = frame.shape[0]
        
        # Get Sensor Data
        yaw_rate = 0.0
        current_speed = 0
        lat, lng = 0.0, 0.0
        gps_curvature = 0.0
        
        if bridge and USE_SENSORS:
            sensor_data = bridge.get_latest_data()
            yaw_rate = sensor_data['yaw']
            current_speed = int(sensor_data.get('spd', 0))
            lat = sensor_data.get('lat', 0.0)
            lng = sensor_data.get('lng', 0.0)
            
            gps_curvature = gps_estimator.update(lat, lng, current_speed)
        
        # Calculate search offset based on yaw (Biasing the sliding window)
        search_bias = int(yaw_rate * 5)

        # --- FUSE WITH PIXEL DETECTION (Fix & Fuse - Moved Upstream) ---
        # 1. Pixel-based detection (Raw) - use proc_frame & precomputed gray
        binary_lane = detect_lane_pixels(proc_frame_blurred, gray=gray)
        l_base_proc, r_base_proc = find_lane_boundaries(binary_lane, search_offset=search_bias//2)
        
        # Scale bases back to display dimensions
        l_base = int(l_base_proc * DISPLAY_WIDTH / PROC_WIDTH)
        r_base = int(r_base_proc * DISPLAY_WIDTH / PROC_WIDTH)
        
        # --- SANITY MONITOR ---
        
        # 1. Timeout Check (Blindness > 1 sec)
        # If no Hough lines found, we might be blind.
        if lines is None:
            consecutive_lost_frames += 1
        else:
            consecutive_lost_frames = 0
            
        if consecutive_lost_frames > 30: # Approx 1 sec at 30 FPS
            print(">> SANITY FAIL: Blind for 1s. Resetting Tracker.")
            tracker = LaneTracker()
            consecutive_lost_frames = 0
            tracker_initialized = False

        # --- UPDATE TRACKER ---
        # Pass pixel bases for internal fusion & smoothing
        l_bottom, r_bottom, vp_coord, left_angle, right_angle, lat_vel = tracker.update(
            lines, width, height, pixel_bases=(l_base, r_base), yaw_rate=yaw_rate, current_speed=current_speed, dt=dt)
        
        # 2. Teleport Check (Physics Impossibility)
        # Check if lane jumped > 100px in one frame (0.03s)
        if tracker_initialized:
            shift_l = abs(l_bottom - prev_l_bot)
            shift_r = abs(r_bottom - prev_r_bot)
            
            if shift_l > 100 or shift_r > 100:
                print(f">> SANITY FAIL: Teleport Detected (L:{shift_l} R:{shift_r}). Increasing covariance.")
                # Gentler approach: increase Kalman covariance instead of full wipe
                tracker.kf_left.P *= 10
                tracker.kf_right.P *= 10
                # Do not re-initialize tracker_initialized to False, keep it running
        
        # Update history for next check
        prev_l_bot = l_bottom
        prev_r_bot = r_bottom
        tracker_initialized = True
        
        vp_x, vp_y = int(vp_coord[0]), int(vp_coord[1])
        lane_center_x = (l_bottom + r_bottom) // 2
        
        # Calculate dynamic lane width
        current_lane_width = r_bottom - l_bottom
        
        # 1. Calc Status (Fixed Signature)
        # Passing CORRECT arguments: vp_x, lane_center_x, current_lane_width, lat_vel, yaw_rate
        status, raw_curve_val = tracker.get_curve_status(vp_x, lane_center_x, current_lane_width, lat_vel, yaw_rate=yaw_rate, speed=current_speed, gps_curvature=gps_curvature, use_sensors=USE_SENSORS)
        
        # --- Continuous Breadcrumb Logging (1Hz) ---
        if crumb_writer:
            current_time_sec = time.time()
            if current_time_sec - last_log_time >= 1.0:
                if lat != 0.0 or lng != 0.0:
                    try:
                        crumb_writer.writerow([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                            lat, 
                            lng, 
                            current_speed, 
                            yaw_rate,
                            status
                        ])
                        breadcrumb_file.flush()
                        last_log_time = current_time_sec
                    except Exception as e:
                        print(f"Failed to write breadcrumb: {e}")

        # Smooth the curve value (offset)
        smoothed_curve_val = tracker.update_curve(raw_curve_val)
        
        # 2. Optimal Speed Calculation
        base_advisory = 100
        if "Sharp" in status: base_advisory = 40
        elif "Curve" in status: base_advisory = 70
        
        # Output a target speed that is safe but realistic to brake towards
        optimal_speed = min(max(current_speed, 30), base_advisory) if current_speed > 5 else 0

        # 4. Draw MINIMALIST AR HUD
        
        # Smart Control Point Logic (Clamping)
        lane_half_width = current_lane_width // 2
        max_shift = int(lane_half_width * 0.5) # Reduced maximum lateral shift
        
        if "Straight" in status:
            control_shift_x = 0
        else:
            # Clamp the shift to prevent overshooting lane boundaries (decreased threshold shift)
            raw_shift = int(smoothed_curve_val * 0.5) 
            control_shift_x = int(np.clip(raw_shift, -max_shift, max_shift))
        
        # Ensure bases don't go wildly out of bounds from Kalman teleport bounces
        l_bottom = int(np.clip(l_bottom, 0, DISPLAY_WIDTH - 1))
        r_bottom = int(np.clip(r_bottom, 0, DISPLAY_WIDTH - 1))
        
        # --- THE PROGRESSIVE RIBBON TRANSFORMATION ---
        
        # 1. Anchor the Left Path 
        p0_l = (l_bottom, height) # Bottom Anchor
        
        # Control 1: Lower-Mid (Stays straight, pushes out 30% towards horizon)
        p1_l = (int(l_bottom + (vp_x - l_bottom) * 0.3), int(height - (height - vp_y) * 0.3))
        
        # Control 2: Upper-Mid (Takes the hard curve shift, 70% towards horizon)
        p2_l = (int(l_bottom + (vp_x - l_bottom) * 0.7) + control_shift_x, int(height - (height - vp_y) * 0.7))
        
        p3_l = (vp_x, vp_y) # Top Anchor
        
        # 2. Derive the Right Path using actual constraints
        actual_width = r_bottom - l_bottom
        # Trust actual width but enforce sane minimums/maximums, don't force a phantom wide lane
        safe_r_bottom = l_bottom + np.clip(actual_width, 300, 1000)
        safe_r_bottom = int(np.clip(safe_r_bottom, 0, DISPLAY_WIDTH - 1))
        
        p0_r = (safe_r_bottom, height)
        p1_r = (int(safe_r_bottom + (vp_x - safe_r_bottom) * 0.3), int(height - (height - vp_y) * 0.3))
        p2_r = (int(safe_r_bottom + (vp_x - safe_r_bottom) * 0.7) + control_shift_x, int(height - (height - vp_y) * 0.7))
        p3_r = (vp_x, vp_y)
        
        # Generate Progressive Points
        left_curve_pts = generate_cubic_bezier_points(p0_l, p1_l, p2_l, p3_l, 40)
        right_curve_pts = generate_cubic_bezier_points(p0_r, p1_r, p2_r, p3_r, 40)

        draw_minimalist_hud(
            frame, 
            current_speed, 
            status, 
            optimal_speed, 
            left_curve_pts, 
            right_curve_pts, 
            (vp_x, vp_y),
            current_lane_width
        )

        cv2.imshow('SafeTurn+ Main', frame)

        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
            
    if breadcrumb_file:
        breadcrumb_file.close()
    if bridge:
        bridge.stop()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
