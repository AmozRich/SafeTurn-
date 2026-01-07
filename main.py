import cv2
import numpy as np
from collections import deque

# --- CONFIGURATION ---
VIDEO_PATH = "drive.mp4" 
LANE_WIDTH_PX = 600 # Approx lane width at bottom of screen
HISTORY_LENGTH = 10 

# Display Settings
DISPLAY_WIDTH = 1280  # Output window width
DISPLAY_HEIGHT = 720  # Output window height 

# State Persistence
class LaneTracker:
    def __init__(self):
        self.vp_history = deque(maxlen=HISTORY_LENGTH)
        self.left_bottom_history = deque(maxlen=HISTORY_LENGTH)
        self.right_bottom_history = deque(maxlen=HISTORY_LENGTH)
        self.curve_history = deque(maxlen=15) # Smooth the visual bending
        self.lane_width_history = deque(maxlen=50) # Adapt to road width
        
        # New: Angle History
        self.left_angle_history = deque(maxlen=HISTORY_LENGTH)
        self.right_angle_history = deque(maxlen=HISTORY_LENGTH)
        
        # Defaults (Center of screen approx)
        self.avg_vp = (640, 360) 
        self.avg_left_bottom = 200
        self.avg_right_bottom = 200 + LANE_WIDTH_PX
        self.avg_curve = 0
        
        # Defaults for angles
        self.avg_left_angle = -45
        self.avg_right_angle = 45

    def update(self, lines, frame_width, frame_height, pixel_bases=None):
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

        # 5. Sanity Checks & History Update
        vp_ok = False
        if curr_vp is not None:
            vx, vy = curr_vp
            # Reject if VP is too high (sky) or too low (hood)
            if frame_height * 0.2 < vy < frame_height * 0.8:
                # Reject sudden huge horizontal jumps (>15% width)
                if len(self.vp_history) > 0:
                    last_vp = self.vp_history[-1]
                    if abs(vx - last_vp[0]) < frame_width * 0.15:
                        vp_ok = True
                else:
                    vp_ok = True
                    
        if vp_ok:
            self.vp_history.append(curr_vp)
        
        if curr_left_angle is not None:
            self.left_angle_history.append(curr_left_angle)
        if curr_right_angle is not None:
            self.right_angle_history.append(curr_right_angle)
            
        if curr_left_bot is not None:
            self.left_bottom_history.append(curr_left_bot)
        if curr_right_bot is not None:
            self.right_bottom_history.append(curr_right_bot)

        # Calculate Smoothed Averages
        if len(self.vp_history) > 0:
            self.avg_vp = np.mean(self.vp_history, axis=0).astype(int)
            
        if len(self.left_angle_history) > 0:
            self.avg_left_angle = np.mean(self.left_angle_history)
        if len(self.right_angle_history) > 0:
            self.avg_right_angle = np.mean(self.right_angle_history)

        if len(self.left_bottom_history) > 0:
            self.avg_left_bottom = int(np.mean(self.left_bottom_history))
        if len(self.right_bottom_history) > 0:
            self.avg_right_bottom = int(np.mean(self.right_bottom_history))

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
        return self.avg_left_bottom, self.avg_right_bottom, self.avg_vp, self.avg_left_angle, self.avg_right_angle
    
    def update_curve(self, raw_curve_val):
        self.curve_history.append(raw_curve_val)
        self.avg_curve = int(np.mean(self.curve_history))
        return self.avg_curve

tracker = LaneTracker()

def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    
    polygons = np.array([
        [
            (0, height), 
            (width // 2, int(height * 0.55)), # Horizon
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

def find_lane_boundaries(binary_lane_img):
    """
    Use sliding window to find left and right lane boundaries.
    Returns bottom positions for left and right lanes.
    """
    height, width = binary_lane_img.shape
    
    # Take histogram of bottom half
    histogram = np.sum(binary_lane_img[height//2:, :], axis=0)
    
    # Find peaks for left and right lanes
    midpoint = width // 2
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

def get_curve_status(vp_x, lane_center_x, lane_width_px=600):
    """
    Decides turning based on relative position of VP vs Lane Center.
    Uses dynamic thresholds based on lane width (Robust VP Offset method).
    """
    global last_status, status_counter
    
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

def generate_bezier_points(p0, p1, p2, num_points=20, cutoff=0.8):
    """
    Generates points for a Quadratic Bezier curve.
    cutoff: Stop at this percentage of the curve (0.8 = 80% to VP)
    """
    points = []
    for t in np.linspace(0, cutoff, num_points):
        x = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
        y = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
        points.append((int(x), int(y)))
    return points

def draw_info_panel(image, speed, status, optimal_speed, lane_center_x, vp_coord):
    height = image.shape[0]
    width = image.shape[1]
    
    # Enhanced Color System (Forza-style)
    speed_diff = speed - optimal_speed
    if speed_diff <= 0:
        # Safe - Bright Blue
        color = (255, 200, 0)  # Cyan/Blue
        status_color = (255, 200, 0)
    elif speed_diff <= 10:
        # Caution - Yellow/Orange
        color = (0, 200, 255)  # Orange
        status_color = (0, 200, 255)
    elif speed_diff <= 20:
        # Warning - Orange/Red
        color = (0, 100, 255)  # Deep Orange
        status_color = (0, 100, 255)
    else:
        # Danger - Red
        color = (0, 0, 255)  # Red
        status_color = (0, 0, 255)

    # Modern HUD Panel Background
    panel_overlay = image.copy()
    cv2.rectangle(panel_overlay, (10, 20), (420, 240), (20, 20, 20), -1)
    cv2.addWeighted(panel_overlay, 0.7, image, 0.3, 0, image)
    
    # Panel Border
    cv2.rectangle(image, (10, 20), (420, 240), (60, 60, 60), 3)
    
    # Title Bar
    cv2.rectangle(image, (10, 20), (420, 60), color, -1)
    cv2.putText(image, "SAFETURN+", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)

    # Speed Display (Large and Prominent)
    font = cv2.FONT_HERSHEY_DUPLEX
    # Text shadow for depth
    cv2.putText(image, f"{speed}", (32, 125), font, 2.5, (0, 0, 0), 4)
    cv2.putText(image, f"{speed}", (30, 123), font, 2.5, color, 3)
    cv2.putText(image, "km/h", (200, 125), font, 0.9, (180, 180, 180), 2)
    
    # Optimal Speed Limit
    cv2.putText(image, f"LIMIT: {optimal_speed} km/h", (30, 165), font, 0.7, (200, 200, 200), 2)
    
    # Status with color coding
    cv2.putText(image, f"{status.upper()}", (30, 210), font, 0.8, status_color, 2)
    
    # Speed Bar Indicator (Visual gauge)
    bar_x, bar_y = 30, 225
    bar_width = 360
    bar_height = 8
    
    # Background bar
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (60, 60, 60), -1)
    
    # Fill bar based on speed vs optimal
    fill_ratio = min(speed / (optimal_speed + 30), 1.0)
    fill_width = int(bar_width * fill_ratio)
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)

    # AR Arrow (Enhanced with glow effect)
    arrow_base = (lane_center_x, height - 80)
    vp_x, vp_y = vp_coord
    
    dx = vp_x - lane_center_x
    dy = vp_y - (height - 80)
    length = np.sqrt(dx**2 + dy**2)
    
    if length > 0:
        ratio = (length - 120) / length  # Reduced arrow length
        target_x = int(lane_center_x + dx * ratio)
        target_y = int((height - 80) + dy * ratio)
        
        # Glow effect (thicker, semi-transparent)
        cv2.arrowedLine(image, arrow_base, (target_x, target_y), color, 12, tipLength=0.25, line_type=cv2.LINE_AA)
        # Main arrow
        cv2.arrowedLine(image, arrow_base, (target_x, target_y), (255, 255, 255), 6, tipLength=0.25, line_type=cv2.LINE_AA)
    
    # Base circle with glow
    cv2.circle(image, arrow_base, 15, color, -1)
    cv2.circle(image, arrow_base, 10, (255, 255, 255), -1)

def draw_debug_gauge(image, angle_balance):
    """
    Draws a visual gauge to show Angle Balance (degrees).
    Straight = 0 deg.
    Left = Negative.
    Right = Positive.
    """
    h, w = image.shape[:2]
    center_x = w // 2
    gauge_y = h - 60
    gauge_width = 300
    gauge_height = 20
    
    # Thresholds (matching get_curve_status logic)
    dead_zone = 5.0
    mild_zone = 15.0
    
    # Background
    cv2.rectangle(image, (center_x - gauge_width//2, gauge_y), 
                  (center_x + gauge_width//2, gauge_y + gauge_height), (40, 40, 40), -1)
    
    # Center Line (Straight)
    cv2.line(image, (center_x, gauge_y - 5), (center_x, gauge_y + gauge_height + 5), (255, 255, 255), 2)
    
    # Dead Zone Area (Green) - Map degrees to pixels
    # Scale: gauge_width/2 covers approx 20 degrees?
    max_deg = 20.0
    scale = (gauge_width / 2) / max_deg
    
    dz_px = int(dead_zone * scale)
    cv2.rectangle(image, (center_x - dz_px, gauge_y + 2), 
                  (center_x + dz_px, gauge_y + gauge_height - 2), (0, 100, 0), -1)
    
    # Threshold Markers (Mild Zone)
    mild_px = int(mild_zone * scale)
    thresh_x_left = center_x - mild_px
    thresh_x_right = center_x + mild_px
    
    cv2.line(image, (thresh_x_left, gauge_y), (thresh_x_left, gauge_y + gauge_height), (0, 255, 255), 2)
    cv2.line(image, (thresh_x_right, gauge_y), (thresh_x_right, gauge_y + gauge_height), (0, 255, 255), 2)
    
    # Current Value Indicator
    val_px = int(angle_balance * scale)
    val_px = np.clip(val_px, -gauge_width//2, gauge_width//2)
    indicator_x = center_x + val_px
    
    indicator_color = (0, 255, 0) # Green (Straight)
    if abs(angle_balance) > dead_zone:
        indicator_color = (0, 255, 255) # Yellow (Curve)
    if abs(angle_balance) > mild_zone:
        indicator_color = (0, 0, 255) # Red (Sharp)
        
    cv2.circle(image, (indicator_x, gauge_y + gauge_height//2), 8, indicator_color, -1)
    cv2.circle(image, (indicator_x, gauge_y + gauge_height//2), 6, (255, 255, 255), 1)
    
    # Text labels
    cv2.putText(image, f"ANGLE BAL: {angle_balance:.1f}", (center_x - 60, gauge_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

def main():
    if VIDEO_PATH:
        print(f"Reading video from: {VIDEO_PATH}")
        cap = cv2.VideoCapture(VIDEO_PATH)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    current_speed = 60 
    
    # Create resizable window
    cv2.namedWindow('SafeTurn+ Main', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('SafeTurn+ Main', DISPLAY_WIDTH, DISPLAY_HEIGHT)
    
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
        cropped_edges = region_of_interest(edges)

        lines = cv2.HoughLinesP(cropped_edges, 2, np.pi/180, 80, np.array([]), minLineLength=40, maxLineGap=100)
        
        # --- FUSE WITH PIXEL DETECTION (Fix & Fuse - Moved Upstream) ---
        # 1. Pixel-based detection (Raw)
        binary_lane = detect_lane_pixels(frame)
        l_base, r_base = find_lane_boundaries(binary_lane)
        
        # --- UPDATE TRACKER ---
        # Pass pixel bases for internal fusion & smoothing
        l_bottom, r_bottom, vp_coord, left_angle, right_angle = tracker.update(lines, width, height, pixel_bases=(l_base, r_base))
        
        vp_x, vp_y = int(vp_coord[0]), int(vp_coord[1])
        lane_center_x = (l_bottom + r_bottom) // 2
        
        # Calculate dynamic lane width
        current_lane_width = r_bottom - l_bottom
        
        # 1. Calc Status (Fixed Signature)
        # Passing CORRECT arguments: vp_x, lane_center_x, current_lane_width
        status, raw_curve_val = get_curve_status(vp_x, lane_center_x, current_lane_width)
        
        # Smooth the curve value (offset)
        smoothed_curve_val = tracker.update_curve(raw_curve_val)
        
        # 2. Optimal Speed
        if "Sharp" in status: optimal_speed = 40
        elif "Curve" in status: optimal_speed = 70
        else: optimal_speed = 100

        # 3. Dynamic Lane Colors (Forza-style)
        speed_diff = current_speed - optimal_speed
        if speed_diff <= 0:
            # Safe - Blue/Cyan
            lane_fill_color = (255, 150, 0)  # Cyan
            lane_border_color = (255, 200, 0)  # Bright Cyan
        elif speed_diff <= 10:
            # Caution - Yellow
            lane_fill_color = (0, 180, 200)  # Yellow
            lane_border_color = (0, 220, 255)  # Bright Yellow
        elif speed_diff <= 20:
            # Warning - Orange
            lane_fill_color = (0, 100, 200)  # Orange
            lane_border_color = (0, 150, 255)  # Bright Orange
        else:
            # Danger - Red
            lane_fill_color = (0, 0, 180)  # Dark Red
            lane_border_color = (0, 0, 255)  # Bright Red

        # 4. Draw CURVED LANES
        line_image = np.zeros_like(frame)
        
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
        
        # Create Polygon: Left Points -> Reverse(Right Points)
        poly_points = left_curve_pts + right_curve_pts[::-1]
        cv2.fillPoly(line_image, [np.array(poly_points, dtype=np.int32)], lane_fill_color)
        
        # Draw Borders with glow effect
        cv2.polylines(line_image, [np.array(left_curve_pts, dtype=np.int32)], False, lane_border_color, 8, lineType=cv2.LINE_AA)
        cv2.polylines(line_image, [np.array(right_curve_pts, dtype=np.int32)], False, lane_border_color, 8, lineType=cv2.LINE_AA)
        

        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        draw_info_panel(combo_image, current_speed, status, optimal_speed, lane_center_x, (vp_x, vp_y))
        
        cv2.imshow('SafeTurn+ Main', combo_image)

        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('w'): 
            current_speed = min(current_speed + 2, 140)
        elif key == ord('s'): 
            current_speed = max(current_speed - 5, 0)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
