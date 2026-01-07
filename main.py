import cv2
import numpy as np
from collections import deque

# --- CONFIGURATION ---
VIDEO_PATH = "drive.mp4" 
LANE_WIDTH_PX = 600 # Approx lane width at bottom of screen
HISTORY_LENGTH = 10 

# State Persistence
class LaneTracker:
    def __init__(self):
        self.vp_history = deque(maxlen=HISTORY_LENGTH)
        self.left_bottom_history = deque(maxlen=HISTORY_LENGTH)
        self.right_bottom_history = deque(maxlen=HISTORY_LENGTH)
        self.curve_history = deque(maxlen=15) # Smooth the visual bending
        
        # Defaults (Center of screen approx)
        self.avg_vp = (640, 360) 
        self.avg_left_bottom = 200
        self.avg_right_bottom = 200 + LANE_WIDTH_PX
        self.avg_curve = 0

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
        
        # 3. Calculate Current Bottom Positions (x at y=height)
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
        
        if len(self.vp_history) > 0:
            self.avg_vp = np.mean(self.vp_history, axis=0).astype(int)

        if curr_left_bot is not None:
            self.left_bottom_history.append(curr_left_bot)
        if curr_right_bot is not None:
            self.right_bottom_history.append(curr_right_bot)

        if len(self.left_bottom_history) > 0:
            self.avg_left_bottom = int(np.mean(self.left_bottom_history))
        if len(self.right_bottom_history) > 0:
            self.avg_right_bottom = int(np.mean(self.right_bottom_history))

        # 6. Fallback / Hallucination Logic
        current_width = self.avg_right_bottom - self.avg_left_bottom
        if current_width < 200: 
            self.avg_right_bottom = self.avg_left_bottom + LANE_WIDTH_PX
        
        # We don't return lines anymore, we return the raw coordinates
        return (self.avg_left_bottom, self.avg_right_bottom, self.avg_vp)
    
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

# --- LOGIC & VISUALIZATION ---

def get_curve_status(vp_x, lane_center_x):
    """
    Decides turning based on relative position of VP vs Lane Center.
    """
    offset = vp_x - lane_center_x
    
    # Tuned Thresholds for Calibration
    dead_zone = 25 
    mild_zone = 150 
    
    status = "Straight"
    if offset > dead_zone:
        status = "Curve Right" if offset < mild_zone else "Sharp Right"
    elif offset < -dead_zone:
        status = "Curve Left" if offset > -mild_zone else "Sharp Left"
        
    return status, offset

def generate_bezier_points(p0, p1, p2, num_points=20):
    """
    Generates points for a Quadratic Bezier curve.
    P0: Start (Bottom)
    P1: Control Point (Controls the bend)
    P2: End (VP)
    """
    points = []
    for t in np.linspace(0, 1, num_points):
        # Quadratic Bezier formula: (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2
        x = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
        y = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
        points.append((int(x), int(y)))
    return points

def draw_info_panel(image, speed, status, optimal_speed, lane_center_x, vp_coord):
    height = image.shape[0]
    
    # Colors
    if speed <= optimal_speed:
        color = (0, 255, 0)
    elif speed <= optimal_speed + 15:
        color = (0, 165, 255)
    else:
        color = (0, 0, 255)

    # Info Text
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, f"SPEED: {speed} km/h", (30, 80), font, 1.2, color, 2)
    cv2.putText(image, f"LIMIT: {optimal_speed} km/h", (30, 130), font, 0.8, (200, 200, 200), 1)
    cv2.putText(image, f"STATUS: {status}", (30, 180), font, 0.8, (255, 255, 255), 1)

    # AR Arrow
    arrow_base = (lane_center_x, height - 80)
    vp_x, vp_y = vp_coord
    
    dx = vp_x - lane_center_x
    dy = vp_y - (height - 80)
    length = np.sqrt(dx**2 + dy**2)
    
    if length > 0:
        ratio = (length - 60) / length 
        target_x = int(lane_center_x + dx * ratio)
        target_y = int((height - 80) + dy * ratio)
        
        cv2.arrowedLine(image, arrow_base, (target_x, target_y), color, 8, tipLength=0.2)
    
    cv2.circle(image, arrow_base, 10, color, -1)

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
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        width = frame.shape[1]
        height = frame.shape[0]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        cropped_edges = region_of_interest(edges)

        lines = cv2.HoughLinesP(cropped_edges, 2, np.pi/180, 80, np.array([]), minLineLength=40, maxLineGap=100)
        
        # --- UPDATE TRACKER ---
        # Now returns coordinates directly
        l_bottom, r_bottom, vp_coord = tracker.update(lines, width, height)
        
        vp_x, vp_y = int(vp_coord[0]), int(vp_coord[1])
        lane_center_x = (l_bottom + r_bottom) // 2
        
        # 1. Calc Status 
        status, raw_curve_val = get_curve_status(vp_x, lane_center_x)
        
        # Smooth the curve value
        smoothed_curve_val = tracker.update_curve(raw_curve_val)
        
        # 2. Optimal Speed
        if "Sharp" in status: optimal_speed = 40
        elif "Curve" in status: optimal_speed = 70
        else: optimal_speed = 100

        # 3. Draw CURVED LANES
        line_image = np.zeros_like(frame)
        
        # FIX: Snap to Straight
        if "Straight" in status:
            control_shift_x = 0
        else:
            control_shift_x = int(smoothed_curve_val * 0.5) 
        
        # Control Point is Midpoint + Shift
        p0_l = (l_bottom, height)
        p2_l = (vp_x, vp_y)
        p1_l = ((l_bottom + vp_x)//2 + control_shift_x, (height + vp_y)//2)
        
        p0_r = (r_bottom, height)
        p2_r = (vp_x, vp_y)
        p1_r = ((r_bottom + vp_x)//2 + control_shift_x, (height + vp_y)//2)
        
        # Generate Points
        left_curve_pts = generate_bezier_points(p0_l, p1_l, p2_l, 40)
        right_curve_pts = generate_bezier_points(p0_r, p1_r, p2_r, 40)
        
        # Create Polygon: Left Points -> Reverse(Right Points)
        poly_points = left_curve_pts + right_curve_pts[::-1]
        cv2.fillPoly(line_image, [np.array(poly_points, dtype=np.int32)], (0, 50, 0))
        
        # Draw Borders
        cv2.polylines(line_image, [np.array(left_curve_pts, dtype=np.int32)], False, (0, 255, 0), 10)
        cv2.polylines(line_image, [np.array(right_curve_pts, dtype=np.int32)], False, (0, 255, 0), 10)
        
        # Draw VP Dot
        cv2.circle(line_image, (vp_x, vp_y), 15, (0, 255, 255), -1)

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
