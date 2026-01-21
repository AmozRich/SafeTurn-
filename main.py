import cv2
import numpy as np
from collections import deque
from sensor_bridge import SensorBridge
from kalman_filter import KalmanFilter
import time

# --- CONFIGURATION ---
VIDEO_PATH = "drive.mp4" 
LANE_WIDTH_PX = 600 # Approx lane width at bottom of screen
HISTORY_LENGTH = 10 
USE_SENSORS = False # Set to True when hardware is connected 

# Display Settings
DISPLAY_WIDTH = 1280  # Output window width
DISPLAY_HEIGHT = 720  # Output window height 

# State Persistence
class LaneTracker:
    def __init__(self):
        # Initial States (Center of screen approx)
        self.avg_left_bottom = 200
        self.avg_right_bottom = 200 + LANE_WIDTH_PX
        self.avg_vp = (640, 360) 
        self.avg_left_angle = -45
        self.avg_right_angle = 45

        # Initialize Kalman Filters
        # Tunning: Smoother (Lower Q, Higher R)
        q_pos = 0.005   # Process Noise (Trust model more)
        r_pos = 200.0   # Measurement Noise (Trust input less, heavily smoothed)

        self.kf_left = KalmanFilter(process_noise=q_pos, measurement_noise=r_pos, initial_state=self.avg_left_bottom)
        self.kf_right = KalmanFilter(process_noise=q_pos, measurement_noise=r_pos, initial_state=self.avg_right_bottom)
        
        # VP needs separate X and Y filters
        self.kf_vp_x = KalmanFilter(process_noise=q_pos, measurement_noise=r_pos, initial_state=640)
        self.kf_vp_y = KalmanFilter(process_noise=q_pos, measurement_noise=r_pos, initial_state=360)
        
        # Angles
        self.kf_left_angle = KalmanFilter(process_noise=0.1, measurement_noise=10.0, initial_state=-45)
        self.kf_right_angle = KalmanFilter(process_noise=0.1, measurement_noise=10.0, initial_state=45)

        self.curve_history = deque(maxlen=15) # Keep for curve smoothing as it's a derived value
        self.lane_width_history = deque(maxlen=50) # Keep for width logic

    def update(self, lines, frame_width, frame_height, pixel_bases=None, yaw_rate=0.0):
        # PREDICTION STEP (Physics)
        self.kf_left.predict()
        self.kf_right.predict()
        self.kf_vp_x.predict()
        self.kf_vp_y.predict()
        self.kf_left_angle.predict()
        self.kf_right_angle.predict()

        # CONTROL INPUT (Yaw Rate)
        # Shift expectations based on car turning. 
        # If we turn Left (+yaw), objects move Right on screen (-x).
        yaw_shift = int(yaw_rate * 300) 
        
        # Direct modification of State Position (External Force)
        self.kf_left.x[0, 0] -= yaw_shift
        self.kf_right.x[0, 0] -= yaw_shift
        self.kf_vp_x.x[0, 0] -= yaw_shift
        # VP Y is mostly unaffected by yaw, maybe pitch, but we ignore for now.

            
        # 1. Separate Lines into Left and Right Candidates
        left_lines = []
        right_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x1 == x2: continue
                
                poly = np.polyfit((x1, x2), (y1, y2), 1)
                slope = poly[0]
                intercept = poly[1]
                
                # Check 1: Slope Filtering
                if slope < -0.3: 
                    left_lines.append((slope, intercept))
                elif slope > 0.3:
                    right_lines.append((slope, intercept))
        
        # 2. Get Raw Averages for this frame
        curr_left_params = np.mean(left_lines, axis=0) if left_lines else None
        curr_right_params = np.mean(right_lines, axis=0) if right_lines else None
        
        # 3. Calculate Velocities/Angles
        curr_left_angle = None
        curr_right_angle = None
        
        if curr_left_params is not None:
            slope = curr_left_params[0]
            # Convert to degrees. Vertical is -90/90. Horizontal is 0.
            # Typical Left Lane: Slope -0.7 -> -35 degrees
            curr_left_angle = np.degrees(np.arctan(slope))
            
        if curr_right_params is not None:
            slope = curr_right_params[0]
            # Typical Right Lane: Slope 0.7 -> +35 degrees
            curr_right_angle = np.degrees(np.arctan(slope))
            
        # 3b. Calculate Current Bottom Positions (Hough Only)
        hough_left_bot = None
        hough_right_bot = None
        
        if curr_left_params is not None:
            slope, intercept = curr_left_params
            hough_left_bot = int((frame_height - intercept) / slope)
            
        if curr_right_params is not None:
            slope, intercept = curr_right_params
            hough_right_bot = int((frame_height - intercept) / slope)

        # --- FUSION STEP (Raw Hough + Raw Pixel) ---
        curr_left_bot = hough_left_bot
        curr_right_bot = hough_right_bot
        
        if pixel_bases is not None:
            px_left, px_right = pixel_bases
            
            # Fuse Left
            if hough_left_bot is not None and px_left is not None:
                 # 70% Hough (Trajectory), 30% Pixel (Base)
                 curr_left_bot = int(0.7 * hough_left_bot + 0.3 * px_left)
            elif px_left is not None:
                 curr_left_bot = px_left # Fallback to pixel if no Hough
            
            # Fuse Right
            if hough_right_bot is not None and px_right is not None:
                 curr_right_bot = int(0.7 * hough_right_bot + 0.3 * px_right)
            elif px_right is not None:
                 curr_right_bot = px_right

        # 4. Vanishing Point (VP) Calculation
        curr_vp = None
        
        if curr_left_params is not None and curr_right_params is not None:
            m1, b1 = curr_left_params
            m2, b2 = curr_right_params
            
            if abs(m1 - m2) > 0.1: # Prevent division by zero
                vp_x = int((b2 - b1) / (m1 - m2))
                vp_y = int(m1 * vp_x + b1)
                curr_vp = (vp_x, vp_y)

        # 5. Sanity Checks & Measurement Update
        vp_ok = False
        if curr_vp is not None:
            vx, vy = curr_vp
            if frame_height * 0.2 < vy < frame_height * 0.8:
                # Basic bounds check vs predicted state
                pred_vx = self.kf_vp_x.get_position()
                if abs(vx - pred_vx) < frame_width * 0.2: # Allow 20% jump, else ignore
                    vp_ok = True
                # Start up condition
                if self.kf_vp_x.P[0,0] > 500: vp_ok = True

        if vp_ok:
            self.kf_vp_x.update(curr_vp[0])
            self.kf_vp_y.update(curr_vp[1])
        
        if curr_left_angle is not None:
             self.kf_left_angle.update(curr_left_angle)
        if curr_right_angle is not None:
             self.kf_right_angle.update(curr_right_angle)
            
        if curr_left_bot is not None:
             self.kf_left.update(curr_left_bot)
        if curr_right_bot is not None:
             self.kf_right.update(curr_right_bot)

        # Retrieve Smoothed States
        self.avg_vp = (int(self.kf_vp_x.get_position()), int(self.kf_vp_y.get_position()))
        self.avg_left_angle = self.kf_left_angle.get_position()
        self.avg_right_angle = self.kf_right_angle.get_position()
        self.avg_left_bottom = int(self.kf_left.get_position())
        self.avg_right_bottom = int(self.kf_right.get_position())

        # 6. Fallback / Hallucination Logic (Adaptive Width)
        current_width_in_history = self.avg_right_bottom - self.avg_left_bottom
        if current_width_in_history > 200:
             self.lane_width_history.append(current_width_in_history)
             
        # Use average tracked width instead of constant if possible
        avg_lane_width = LANE_WIDTH_PX
        if len(self.lane_width_history) > 5:
            avg_lane_width = int(np.mean(self.lane_width_history))
            
        current_width = self.avg_right_bottom - self.avg_left_bottom
        if current_width < 200: 
            self.avg_right_bottom = self.avg_left_bottom + avg_lane_width
        
        # 7. Bounds Checking - Keep lanes within frame
        # Constrain to reasonable bounds (with margin)
        margin = 50
        if self.avg_left_bottom < margin:
            self.avg_left_bottom = margin
        if self.avg_right_bottom > frame_width - margin:
            self.avg_right_bottom = frame_width - margin
        
        # Ensure minimum lane width is maintained after bounds check
        if self.avg_right_bottom - self.avg_left_bottom < 200:
            center = (self.avg_left_bottom + self.avg_right_bottom) // 2
            self.avg_left_bottom = center - 100
            self.avg_right_bottom = center + 100
        
        # We don't return lines anymore, we return the raw coordinates
        # Also return average lateral velocity (pixel/frame)
        lat_vel = (self.kf_left.get_velocity() + self.kf_right.get_velocity()) / 2.0
        return self.avg_left_bottom, self.avg_right_bottom, self.avg_vp, self.avg_left_angle, self.avg_right_angle, lat_vel
    
    def update_curve(self, raw_curve_val):
        self.curve_history.append(raw_curve_val)
        self.avg_curve = int(np.mean(self.curve_history))
        return self.avg_curve

tracker = LaneTracker()

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

def detect_lane_pixels(frame):
    """
    Detect white lane marking pixels using color filtering.
    Returns a binary image with lane pixels highlighted.
    """
    # Convert to HLS color space (better for white detection)
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    
    # Define range for white color
    lower_white = np.array([0, 200, 0])
    upper_white = np.array([255, 255, 255])
    
    # Create mask for white pixels
    white_mask = cv2.inRange(hls, lower_white, upper_white)
    
    # Also use grayscale thresholding as backup
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, gray_thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Combine both masks
    combined = cv2.bitwise_or(white_mask, gray_thresh)
    
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
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # If no strong peaks found, use defaults
    if histogram[left_base] < 100:
        left_base = width // 4
    if histogram[right_base] < 100:
        right_base = 3 * width // 4
    
    return left_base, right_base

# --- LOGIC & VISUALIZATION ---

# Hysteresis state
last_status = "Straight"
status_counter = 0

def get_curve_status(vp_x, lane_center_x, lane_width_px, lateral_velocity):
    """
    Decides turning based on relative position of VP vs Lane Center.
    Uses lateral velocity to detect lane changes.
    """
    global last_status, status_counter
    
    # 1. Lane Change Detection
    # If the lanes are sliding sideways fast (e.g. > 1.5 px/frame), it's a lane change.
    if abs(lateral_velocity) > 1.5:
        # Reset curve status to straight during merge so we don't show confusing arrows
        return "Straight", 0
    
    # VP is smoothed, so this is more stable than raw line angles
    offset = vp_x - lane_center_x
    
    # Dynamic Thresholds
    dead_zone = int(lane_width_px * 0.05) # 5%
    mild_zone = int(lane_width_px * 0.20) # 20% (reduced slightly)
    
    current_status = "Straight"
    
    # Logic: Change #5 Fixed Logic (VP Right = Curve Left)
    if offset > dead_zone:
        current_status = "Curve Left" if offset < mild_zone else "Sharp Left"
    elif offset < -dead_zone:
        current_status = "Curve Right" if offset > -mild_zone else "Sharp Right"
        
    # Hysteresis
    if current_status != last_status:
        status_counter += 1
        if status_counter > 3: 
            last_status = current_status
            status_counter = 0
    else:
        status_counter = 0
        
    return last_status, offset

def generate_bezier_points(p0, p1, p2, num_points=20, cutoff=0.9):
    """
    Generates points for a Quadratic Bezier curve.
    cutoff: Stop at this percentage of the curve (0.9 = 90% to VP)
    """
    points = []
    for t in np.linspace(0, cutoff, num_points):
        x = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
        y = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
        points.append((int(x), int(y)))
    return points

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
    if "Curve" in status: color_fill = (0, 165, 255) # Amber
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
    bracket_w = 40
    bracket_h = 200
    b_color = (255, 255, 255)
    
    # Left Bracket
    l_x = int(vp_x - lane_width_px * 0.4)
    r_x = int(vp_x + lane_width_px * 0.4)
    
    # Draw simple vertical marks at the horizon line
    cv2.line(image, (l_x, vp_y - 20), (l_x, vp_y + 20), b_color, 1, cv2.LINE_AA)
    cv2.line(image, (r_x, vp_y - 20), (r_x, vp_y + 20), b_color, 1, cv2.LINE_AA)


def main():
    # Initialize Sensor Bridge
    bridge = None
    if USE_SENSORS:
        try:
            bridge = SensorBridge(port='COM3', baud=115200) # Adjust COM port as needed
            bridge.start()
            time.sleep(1) # Wait for connection
        except Exception as e:
            print(f"Sensor Warning: {e}")

    if VIDEO_PATH:
        print(f"Reading video from: {VIDEO_PATH}")
        cap = cv2.VideoCapture(VIDEO_PATH)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    current_speed = 0 
    
    # Create resizable window
    cv2.namedWindow('SafeTurn+ Main', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('SafeTurn+ Main', DISPLAY_WIDTH, DISPLAY_HEIGHT)
    
    # Initialize Tracker locally
    tracker = LaneTracker()
    
    consecutive_lost_frames = 0
    tracker_initialized = False # To ignore first-frame jump
    
    # Track previous state for Teleport Check
    prev_l_bot = 0
    prev_r_bot = 0

    while True:
        ret, frame = cap.read()
        if not ret: 
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # Resize frame to display dimensions
        frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            
        width = frame.shape[1]
        height = frame.shape[0]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        # Dynamic ROI: Focus where the tracker thinks the road is
        current_vp_x = int(tracker.avg_vp[0])
        cropped_edges = region_of_interest(edges, vp_x=current_vp_x)

        lines = cv2.HoughLinesP(cropped_edges, 2, np.pi/180, 80, np.array([]), minLineLength=40, maxLineGap=100)
        
        # Get Sensor Data
        yaw_rate = 0.0
        current_speed = 0
        
        if bridge and USE_SENSORS:
            sensor_data = bridge.get_latest_data()
            yaw_rate = sensor_data['yaw']
            current_speed = int(sensor_data['spd'])
        
        # Calculate search offset based on yaw (Biasing the sliding window)
        # If yaw > 0 (Left Turn), assume lanes shift.
        search_bias = int(yaw_rate * 200)

        # --- FUSE WITH PIXEL DETECTION (Fix & Fuse - Moved Upstream) ---
        # 1. Pixel-based detection (Raw)
        binary_lane = detect_lane_pixels(frame)
        l_base, r_base = find_lane_boundaries(binary_lane, search_offset=search_bias)
        
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
        l_bottom, r_bottom, vp_coord, left_angle, right_angle, lat_vel = tracker.update(lines, width, height, pixel_bases=(l_base, r_base), yaw_rate=yaw_rate)
        
        # 2. Teleport Check (Physics Impossibility)
        # Check if lane jumped > 100px in one frame (0.03s)
        if tracker_initialized:
            shift_l = abs(l_bottom - prev_l_bot)
            shift_r = abs(r_bottom - prev_r_bot)
            
            if shift_l > 100 or shift_r > 100:
                print(f">> SANITY FAIL: Teleport Detected (L:{shift_l} R:{shift_r}). Re-acquiring.")
                tracker = LaneTracker()
                tracker_initialized = False
                continue # Skip drawing this frame to avoid glitch
        
        # Update history for next check
        prev_l_bot = l_bottom
        prev_r_bot = r_bottom
        tracker_initialized = True
        
        vp_x, vp_y = int(vp_coord[0]), int(vp_coord[1])
        lane_center_x = (l_bottom + r_bottom) // 2
        
        # Calculate dynamic lane width
        current_lane_width = r_bottom - l_bottom
        
        # 1. Calc Status (Fixed Signature)
        # Passing CORRECT arguments: vp_x, lane_center_x, current_lane_width
        status, raw_curve_val = get_curve_status(vp_x, lane_center_x, current_lane_width, lat_vel)
        
        # Smooth the curve value (offset)
        smoothed_curve_val = tracker.update_curve(raw_curve_val)
        
        # 2. Optimal Speed
        if "Sharp" in status: optimal_speed = 40
        elif "Curve" in status: optimal_speed = 70
        else: optimal_speed = 100

        # 4. Draw MINIMALIST AR HUD
        
        # Smart Control Point Logic (Clamping)
        lane_half_width = current_lane_width // 2
        max_shift = int(lane_half_width * 0.6) # Limit shift to 60% of half-width
        
        if "Straight" in status:
            control_shift_x = 0
        else:
            # Clamp the shift to prevent overshooting lane boundaries
            raw_shift = int(smoothed_curve_val * 0.9)
            control_shift_x = int(np.clip(raw_shift, -max_shift, max_shift))
        
        # Control Point Y (60% depth)
        control_y = int(height - (height - vp_y) * 0.4) 
        
        p0_l = (l_bottom, height)
        p2_l = (vp_x, vp_y)
        p1_l = ((l_bottom + vp_x)//2 + control_shift_x, control_y)
        
        p0_r = (r_bottom, height)
        p2_r = (vp_x, vp_y)
        p1_r = ((r_bottom + vp_x)//2 + control_shift_x, control_y)
        
        # Generate Points
        left_curve_pts = generate_bezier_points(p0_l, p1_l, p2_l, 40)
        right_curve_pts = generate_bezier_points(p0_r, p1_r, p2_r, 40)

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
            
    if bridge:
        bridge.stop()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
