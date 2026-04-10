import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
from hazard_manager import HazardManager

def run_pothole_detection(video_source, bridge, use_sensors=False, use_webcam=False, webcam_index=0):
    """
    Standalone Pothole Detection module for SafeTurn+.
    Integrates YOLOv8 inference with the ESP32 SensorBridge.
    """
    print("Loading YOLOv8 Pothole Model...")
    # Your Predator Helios Neo 16 should handle this inference easily, 
    # especially if PyTorch is utilizing the dedicated GPU.
    model = YOLO("Yolov8-fintuned-on-potholes.pt")
    hazard_manager = HazardManager()

    # 1. Setup Video Capture
    if use_webcam:
        cap = cv2.VideoCapture(webcam_index)
    else:
        cap = cv2.VideoCapture(video_source if video_source else "drive.mp4")

    cv2.namedWindow('SafeTurn+ Pothole Scanner', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('SafeTurn+ Pothole Scanner', 1280, 720)

    # Initialize variables for HUD
    current_speed = 0
    lat, lng = 0.0, 0.0
    last_log_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Loop video if running a test file
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
            continue

        frame = cv2.resize(frame, (1280, 720))

        # 2. Sensor Data Retrieval (SafeTurn Integration)
        if use_sensors and bridge:
            sensor_data = bridge.get_latest_data()
            current_speed = int(sensor_data.get('spd', 0))
            lat = sensor_data.get('lat', 0.0)
            lng = sensor_data.get('lng', 0.0)

        # 3. YOLO Inference
        # verbose=False keeps the terminal clean from constant YOLO spam
        results = model.predict(source=frame, conf=0.4, verbose=False)
        
        pothole_detected = False
        
        # 4. Process Results & Draw Bounding Boxes
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Further optimize by checking bounding box dimensions
                # A pothole cannot be the size of the entire frame (false positive filter)
                box_width = x2 - x1
                box_height = y2 - y1
                frame_area = 1280 * 720
                box_area = box_width * box_height
                
                if conf > 0.4 and (box_area < frame_area * 0.4): # Ignore boxes over 40% screen size
                    pothole_detected = True
                    
                    # Compute Safe Speed based on Bounding Box Area
                    # If box > 15% screen -> Massive (15 km/h)
                    # If box <= 15% screen -> Tiny (25 km/h)
                    ratio = box_area / frame_area
                    safe_speed = 15 if ratio > 0.15 else 25
                    
                    # Draw warning box (Red for hazard)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"POTHOLE {conf:.2f} ({safe_speed} km/h)"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 5. SafeTurn Standard HUD Overlay
        # Display Speed
        cv2.putText(frame, f"{current_speed} km/h", (50, 680), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 2)
        
        # Alert & GPS Logging Display
        if pothole_detected:
            cv2.putText(frame, "HAZARD DETECTED", (50, 80), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
            
            if use_sensors:
                # Unified Database Logging
                coord_text = f"Logged at: {lat:.6f}, {lng:.6f} ({safe_speed}km/h)"
                cv2.putText(frame, coord_text, (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                current_time = time.time()
                if current_time - last_log_time > 3.0: # 3 second cooldown
                    try:
                        hazard_manager.add_hazard(lat, lng, "Pothole", safe_speed)
                        last_log_time = current_time
                        print(f"Logged Pothole: Lat={lat}, Lng={lng}, Speed={safe_speed}")
                    except Exception as e:
                        print(f"Failed to log pothole: {e}")

        cv2.imshow('SafeTurn+ Pothole Scanner', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()