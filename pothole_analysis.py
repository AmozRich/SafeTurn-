import cv2
import numpy as np
import os
import csv
import time
from ultralytics import YOLO

def run_pothole_detection(video_source, bridge, use_sensors=False, use_webcam=False, webcam_index=0):
    """
    Standalone Pothole Detection module for SafeTurn+.
    Integrates YOLOv8 inference with the ESP32 SensorBridge.
    """
    print("Loading YOLOv8 Pothole Model...")
    # Your Predator Helios Neo 16 should handle this inference easily, 
    # especially if PyTorch is utilizing the dedicated GPU.
    model = YOLO("Yolov8-fintuned-on-potholes.pt")

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
                    # Draw warning box (Red for hazard)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"POTHOLE {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 5. SafeTurn Standard HUD Overlay
        # Display Speed
        cv2.putText(frame, f"{current_speed} km/h", (50, 680), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 2)
        
        # Alert & GPS Logging Display
        if pothole_detected:
            cv2.putText(frame, "HAZARD DETECTED", (50, 80), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
            
            if use_sensors:
                # Logging precise coordinates is going to be incredibly useful for mapping out 
                # changing road conditions around Thiruvananthapuram.
                coord_text = f"Logged at: {lat:.6f}, {lng:.6f}"
                cv2.putText(frame, coord_text, (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Simple CSV writer to save the lat/lng coordinates 
                # whenever a pothole is detected. Includes a 5-second cooldown to prevent log spamming.
                current_time = time.time()
                if current_time - last_log_time > 5.0:
                    file_exists = os.path.isfile("potholes_logged.csv")
                    try:
                        with open("potholes_logged.csv", "a", newline="") as f:
                            writer = csv.writer(f)
                            if not file_exists:
                                writer.writerow(["Timestamp", "Latitude", "Longitude", "Speed_kmh"])
                            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), lat, lng, current_speed])
                        last_log_time = current_time
                    except Exception as e:
                        print(f"Failed to log pothole: {e}")

        cv2.imshow('SafeTurn+ Pothole Scanner', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()