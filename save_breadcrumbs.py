import time
import math
import csv
import os
from datetime import datetime
from sensor_bridge import SensorBridge
from gps_utils import GPSCurvatureEstimator, classify_curve
from config import SENSOR_PORT, SENSOR_BAUD_RATE

# --- CONFIGURATION ---
SERIAL_PORT = SENSOR_PORT
BAUD_RATE = SENSOR_BAUD_RATE
# -------------------

def log_breadcrumbs():
    print(f"Starting Sensor Bridge on {SERIAL_PORT} at {BAUD_RATE} baud...")
    bridge = SensorBridge(port=SERIAL_PORT, baud=BAUD_RATE)
    bridge.start()
    
    gps_estimator = GPSCurvatureEstimator()
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"route_breadcrumbs_{timestamp_str}.csv"
    
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Latitude", "Longitude", "Speed_kmh", "Yaw_Rate", "GPS_Curvature", "Status", "Heading"])
        
    print(f"Started logging to {csv_filename}. Press Ctrl+C to stop.")
    
    try:
        csv_file = open(csv_filename, "a", newline="")
        writer = csv.writer(csv_file)
        
        while True:
            data = bridge.get_latest_data()
            
            yaw_rate = data.get('yaw', 0.0)
            current_speed = int(data.get('spd', 0))
            lat = data.get('lat', 0.0)
            lng = data.get('lng', 0.0)
            
            gps_curvature = gps_estimator.update(lat, lng, current_speed)
                    
            # --- SENSOR FUSION LOGIC ---
            current_status = classify_curve(yaw_rate, gps_curvature)
            
            # Write breadcrumb (1Hz logging to match main.py)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                lat, 
                lng, 
                current_speed, 
                yaw_rate,
                gps_curvature,
                current_status,
                gps_estimator.current_heading
            ])
            csv_file.flush()
                
            print(f"Logged point: Lat={lat}, Lng={lng}, Speed={current_speed}km/h, Yaw={yaw_rate:.2f}, Curv={gps_curvature:.4f}, Status={current_status}, Head={gps_estimator.current_heading:.1f}")
            time.sleep(1.0)
                        
    except KeyboardInterrupt:
        pass
    finally:
        if 'csv_file' in locals():
            csv_file.close()
        print("\nLogging stopped by user.")
        bridge.stop()
        print("Done.")

if __name__ == "__main__":
    log_breadcrumbs()
