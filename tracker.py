import numpy as np
from collections import deque
from kalman_filter import KalmanFilter

# --- CONFIGURATION ---
LANE_WIDTH_PX = 600 # Approx lane width at bottom of screen

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
