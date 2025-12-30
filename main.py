import cv2
import numpy as np

# --- CONFIGURATION ---
VIDEO_PATH = "drive.mp4" 

def make_coordinates(image, line_parameters):
    if line_parameters is None:
        return None
        
    slope, intercept = line_parameters
    y1 = image.shape[0] 
    # FIX 1: Shorten the lines so they don't cross in the middle
    # Was 0.6 (too long), now 0.7 (shorter, stays in lower part of screen)
    y2 = int(y1 * 0.63)
    
    try:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
    except ZeroDivisionError:
        return None
        
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    
    if lines is None:
        return None

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Polyfit can crash on vertical lines, so we wrap it
        if x1 == x2: continue 
        
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]

        # FILTER: Tweak slope to ignore horizontal-ish lines
        if slope < -0.4: # Negative slope = Left Lane
            left_fit.append((slope, intercept))
        elif slope > 0.4: # Positive slope = Right Lane
            right_fit.append((slope, intercept))

    # Average the lines
    left_fit_average = np.average(left_fit, axis=0) if len(left_fit) > 0 else None
    right_fit_average = np.average(right_fit, axis=0) if len(right_fit) > 0 else None

    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)

    # Return only valid lines
    return np.array([l for l in [left_line, right_line] if l is not None])

def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    
    # FOCUS AREA: Keep ignoring the hood (bottom 50px)
    polygons = np.array([
        [
            (200, height - 50),
            (width // 2, int(height * 0.6)), # Horizon
            (width - 200, height - 50) 
        ]
    ])
    
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def main():
    if VIDEO_PATH:
        print(f"Reading video from: {VIDEO_PATH}")
        cap = cv2.VideoCapture(VIDEO_PATH)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    return masked_image

# --- LOGIC & VISUALIZATION ---

def get_curve_status(lines, width):
    """
    Decides if we are going Straight, turning Left, or turning Right.
    Returns: status_string, color_code (for future use), offset
    """
    if lines is None or len(lines) != 2:
        return "Detecting...", 0

    # lines[0] is Left, lines[1] is Right (usually, based on our previous function)
    left_line = lines[0]
    right_line = lines[1]

    # Calculate bottom x-coordinates to find lane center
    # [x1, y1, x2, y2] - we care about the bottom point (y1 is bottom)
    l_x_bottom = left_line[0]
    r_x_bottom = right_line[0]

    lane_center = (l_x_bottom + r_x_bottom) // 2
    image_center = width // 2
    
    # Offset: Positive = Right, Negative = Left
    offset = lane_center - image_center
    
    # Thresholds (Tweak these based on camera mounting/resolution!)
    # Assuming slight deviations needed to count as a curve
    dead_zone = 50 

    if abs(offset) < dead_zone:
        return "Straight", offset
    elif offset > dead_zone:
        status = "Sharp Right" if offset > 150 else "Mild Right"
        return status, offset
    elif offset < -dead_zone:
        status = "Sharp Left" if offset < -150 else "Mild Left"
        return status, offset
    
    return "Unknown", 0

def draw_info_panel(image, speed, status, optimal_speed):
    """
    Draws the Speed and Warning Arrows.
    """
    # 1. Determine Color based on Speed Limit
    # Blue (Safe), Orange (Caution), Red (Danger)
    if speed <= optimal_speed:
        color = (255, 100, 0) # Blue-ish (BGR)
    elif speed <= optimal_speed + 20:
        color = (0, 165, 255) # Orange
    else:
        color = (0, 0, 255) # Red

    # 2. Draw Speedometer Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f"SPEED: {speed} km/h", (50, 100), font, 1, color, 3)
    cv2.putText(image, f"LIMIT: {optimal_speed} km/h", (50, 150), font, 0.7, (200, 200, 200), 2)
    cv2.putText(image, f"ROAD: {status}", (50, 50), font, 1, (255, 255, 255), 3)

    # 3. Draw Direction Arrow (Simple Visual)
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    
    if "Left" in status:
        # Arrow pointing Left
        pt1 = (center_x + 50, center_y)
        pt2 = (center_x - 50, center_y)
    elif "Right" in status:
        # Arrow pointing Right
        pt1 = (center_x - 50, center_y)
        pt2 = (center_x + 50, center_y)
    else: 
        # Arrow pointing Up (Straight)
        pt1 = (center_x, center_y + 50)
        pt2 = (center_x, center_y - 50)

    cv2.arrowedLine(image, pt1, pt2, color, 10, tipLength=0.5)

def main():
    if VIDEO_PATH:
        print(f"Reading video from: {VIDEO_PATH}")
        cap = cv2.VideoCapture(VIDEO_PATH)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # SIMULATION VARIABLES
    current_speed = 60 # Start at 60 km/h
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        cropped_edges = region_of_interest(edges)

        lines = cv2.HoughLinesP(cropped_edges, 2, np.pi/180, 80, np.array([]), minLineLength=40, maxLineGap=50)
        averaged_lines = average_slope_intercept(frame, lines)

        # --- LOGIC UPDATE ---
        width = frame.shape[1]
        
        # 1. Curve Classification
        status, offset = get_curve_status(averaged_lines, width)
        
        # 2. Assign Optimal Speed
        if "Sharp" in status:
            optimal_speed = 40
        elif "Mild" in status:
            optimal_speed = 70
        else: # Straight or Detecting
            optimal_speed = 100

        # 3. Visualization
        line_image = np.zeros_like(frame)
        if averaged_lines is not None:
            for line in averaged_lines:
                x1, y1, x2, y2 = line
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)

        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        
        # Draw the HUD
        draw_info_panel(combo_image, current_speed, status, optimal_speed)

        cv2.imshow('SafeTurn+ Main', combo_image)

        # 4. Input Handling (Simulate Speed)
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
        elif key == 2490368 or key == 82: # Up Arrow (varies by OS, simple check usually helps to use libraries but raw key codes often 82/84)
            # Simple fallback for standard letters if arrow keys fail to map reliably in cv2 only
            pass
            
        # Hard to map Arrows cross-platform in OpenCv waitKey without curses/etc.
        # Let's use 'w' and 's' for Speed Control as they are reliable!
        if key == ord('w'): # W for Accelerate
            current_speed = min(current_speed + 2, 140)
        elif key == ord('s'): # S for Brake
            current_speed = max(current_speed - 5, 0)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
