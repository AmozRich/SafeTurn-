import cv2
import numpy as np
from collections import deque
import sys
import torch
sys.path.insert(0, 'enet')
from enet_loader import ENetLaneDetector


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

    def update(self, lines, frame_width, frame_height):
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
            
        # 3b. Calculate Current Bottom Positions
        curr_left_bot = None
        curr_right_bot = None
        
        if curr_left_params is not None:
            slope, intercept = curr_left_params
            curr_left_bot = int((frame_height - intercept) / slope)
            
        if curr_right_params is not None:
            slope, intercept = curr_right_params
            curr_right_bot = int((frame_height - intercept) / slope)

        # 4. Vanishing Point (VP) Calculation
        curr_vp = None
        
        if curr_left_params is not None and curr_right_params is not None:
            m1, b1 = curr_left_params
            m2, b2 = curr_right_params
            
            if abs(m1 - m2) > 0.1: # Prevent division by zero
                vp_x = int((b2 - b1) / (m1 - m2))
                vp_y = int(m1 * vp_x + b1)
                curr_vp = (vp_x, vp_y)

        # 5. Update Histories & Smoothing
        if curr_vp is not None:
            if 0 < curr_vp[0] < frame_width and 0 < curr_vp[1] < frame_height:
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

        # 6. Fallback / Hallucination Logic
        current_width = self.avg_right_bottom - self.avg_left_bottom
        if current_width < 200: 
            self.avg_right_bottom = self.avg_left_bottom + LANE_WIDTH_PX
        
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
    
    # Professional Automotive Color System (ADAS Standard)
    speed_diff = speed - optimal_speed
    if speed_diff <= 0:
        # Safe - Green
        color = (0, 200, 0)  # Green
        status_color = (0, 200, 0)
    elif speed_diff <= 10:
        # Caution - Amber
        color = (0, 180, 255)  # Amber/Yellow
        status_color = (0, 180, 255)
    elif speed_diff <= 20:
        # Warning - Orange
        color = (0, 120, 255)  # Orange
        status_color = (0, 120, 255)
    else:
        # Danger - Red
        color = (0, 0, 255)  # Red
        status_color = (0, 0, 255)

    # Clean HUD Panel Background
    panel_overlay = image.copy()
    cv2.rectangle(panel_overlay, (10, 20), (380, 220), (30, 30, 30), -1)
    cv2.addWeighted(panel_overlay, 0.75, image, 0.25, 0, image)
    
    # Simple Panel Border
    cv2.rectangle(image, (10, 20), (380, 220), (80, 80, 80), 2)
    
    # Clean Title Bar
    cv2.rectangle(image, (10, 20), (380, 55), (50, 50, 50), -1)
    cv2.putText(image, "SafeTurn+", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)

    # Speed Display (Clean, No Shadows)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f"{speed}", (30, 115), font, 2.2, color, 3)
    cv2.putText(image, "km/h", (180, 115), font, 0.7, (180, 180, 180), 2)
    
    # Optimal Speed Limit
    cv2.putText(image, f"Limit: {optimal_speed} km/h", (30, 150), font, 0.6, (200, 200, 200), 1)
    
    # Status with color coding
    cv2.putText(image, f"{status}", (30, 185), font, 0.7, status_color, 2)
    
    # Clean Speed Bar Indicator
    bar_x, bar_y = 30, 200
    bar_width = 320
    bar_height = 6
    
    # Background bar
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (60, 60, 60), -1)
    
    # Fill bar based on speed vs optimal
    fill_ratio = min(speed / (optimal_speed + 30), 1.0)
    fill_width = int(bar_width * fill_ratio)
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)

    # Simple Direction Arrow (No Glow)
    arrow_base = (lane_center_x, height - 80)
    vp_x, vp_y = vp_coord
    
    dx = vp_x - lane_center_x
    dy = vp_y - (height - 80)
    length = np.sqrt(dx**2 + dy**2)
    
    if length > 0:
        ratio = (length - 120) / length
        target_x = int(lane_center_x + dx * ratio)
        target_y = int((height - 80) + dy * ratio)
        
        # Clean arrow
        cv2.arrowedLine(image, arrow_base, (target_x, target_y), color, 5, tipLength=0.3, line_type=cv2.LINE_AA)
    
    # Simple base circle
    cv2.circle(image, arrow_base, 8, color, -1)
    cv2.circle(image, arrow_base, 10, color, 2)

# Debug gauge removed for cleaner professional display

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
    
    # Initialize variables to prevent scope errors
    frame_id = 0
    enet_enabled = False
    last_lane_mask = None

    # Initialize ENet for dense lane detection
    print("\n[ENet] Initializing lane detector...")
    try:
        # Try GPU first, fallback to CPU if unavailable
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ENet] Using device: {device}")
        enet = ENetLaneDetector("enet/ENET.pth", device=device)
        enet_enabled = True
        last_lane_mask = None
        frame_id = 0
        print("[ENet] Activated successfully!")
    except Exception as e:
        print(f"[ENet] Could not load ENet: {e}")
        print("[ENet] Continuing with Hough-based detection only.")
        enet_enabled = False
    
    # Create resizable window
    cv2.namedWindow('SafeTurn+ Main', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('SafeTurn+ Main', DISPLAY_WIDTH, DISPLAY_HEIGHT)
    
    # Optional: Create ENet visualization window (press 'e' to toggle)
    show_enet = False
    
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
        
        # --- ENET INFERENCE (Hybrid Approach) ---
        # Run ENet every 2nd frame for performance (cache results)
        if enet_enabled and frame_id % 2 == 0:
            binary_lane = enet.infer(frame)
            if binary_lane is not None:
                last_lane_mask = binary_lane
        elif enet_enabled:
            binary_lane = last_lane_mask
        
        frame_id += 1

        lines = cv2.HoughLinesP(cropped_edges, 2, np.pi/180, 80, np.array([]), minLineLength=40, maxLineGap=100)
        
        # Optional: Visualize ENet mask (press 'e' to toggle)
        if enet_enabled and show_enet and last_lane_mask is not None:
            # Show raw binary mask (what ENet actually detected)
            debug_mask = last_lane_mask.copy()
            
            # Add text overlay with statistics
            white_pixels = np.sum(debug_mask == 255)
            total_pixels = debug_mask.shape[0] * debug_mask.shape[1]
            detection_percentage = (white_pixels / total_pixels) * 100
            
            # Convert to color for better visibility
            debug_color = cv2.cvtColor(debug_mask, cv2.COLOR_GRAY2BGR)
            
            # Green overlay on detected lanes
            green_overlay = np.zeros_like(frame)
            green_overlay[debug_mask == 255] = [0, 255, 0]  # Green where lanes detected
            enet_overlay = cv2.addWeighted(frame, 0.7, green_overlay, 0.3, 0)
            
            # Add debug info
            cv2.putText(enet_overlay, f"ENet Detection: {detection_percentage:.1f}%", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(enet_overlay, f"White Pixels: {white_pixels}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('ENet Lane Mask', enet_overlay)
            cv2.imshow('ENet Raw Binary', debug_mask)
        
        # --- UPDATE TRACKER ---
        # Now returns coordinates directly AND improved angles
        l_bottom, r_bottom, vp_coord, left_angle, right_angle = tracker.update(lines, width, height)
        
        vp_x, vp_y = int(vp_coord[0]), int(vp_coord[1])
        lane_center_x = (l_bottom + r_bottom) // 2
        
        # 1. Calc Status (New Angle Method)
        status, raw_curve_val = get_curve_status(left_angle, right_angle)
        
        # Smooth the curve value (Angle Balance)
        smoothed_curve_val = tracker.update_curve(raw_curve_val)
        
        # 2. Optimal Speed
        if "Sharp" in status: optimal_speed = 40
        elif "Curve" in status: optimal_speed = 70
        else: optimal_speed = 100

        # 3. Professional Lane Colors (Automotive Safety Standard)
        speed_diff = current_speed - optimal_speed
        if speed_diff <= 0:
            # Safe - Green
            lane_fill_color = (0, 150, 0)  # Green
            lane_border_color = (0, 200, 0)  # Bright Green
        elif speed_diff <= 10:
            # Caution - Amber
            lane_fill_color = (0, 140, 200)  # Amber
            lane_border_color = (0, 180, 255)  # Bright Amber
        elif speed_diff <= 20:
            # Warning - Orange
            lane_fill_color = (0, 100, 200)  # Orange
            lane_border_color = (0, 120, 255)  # Bright Orange
        else:
            # Danger - Red
            lane_fill_color = (0, 0, 180)  # Dark Red
            lane_border_color = (0, 0, 255)  # Bright Red

        # 4. Draw CURVED LANES
        line_image = np.zeros_like(frame)
        
        # Snap to Straight
        if "Straight" in status:
            control_shift_x = 0
        else:
            # Reverted to smoothed VP offset approach
            # Multiplier 0.5 was too stiff? User said 1.2 "overshot".
            # User said "stiff and overshoots".
            # Let's try 0.9 (Middle ground) + Better Control Point Height
            control_shift_x = int(smoothed_curve_val * 0.9) 
        
        # Reverted Control Point Logic (Previous Stable Version)
        # 40% up from bottom
        control_y = int(height - (height - vp_y) * 0.4)
        
        # Standard Multiplier (Removed dynamic speed damping)
        control_shift_x = int(smoothed_curve_val * 0.9) if "Straight" not in status else 0
        
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
        
        # Draw clean lane borders
        cv2.polylines(line_image, [np.array(left_curve_pts, dtype=np.int32)], False, lane_border_color, 6, lineType=cv2.LINE_AA)
        cv2.polylines(line_image, [np.array(right_curve_pts, dtype=np.int32)], False, lane_border_color, 6, lineType=cv2.LINE_AA)
        

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
        elif key == ord('e'):
            # Toggle ENet visualization
            show_enet = not show_enet
            if not show_enet:
                cv2.destroyWindow('ENet Lane Mask')
            print(f"[ENet] Visualization: {'ON' if show_enet else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
