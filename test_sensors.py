import time
import math
import csv
import os
from datetime import datetime
from sensor_bridge import SensorBridge
from gps_utils import GPSCurvatureEstimator

# Change this to your actual COM port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux/Mac)
PORT = 'COM10'
BAUD = 115200

def test_sensors():
    print(f"Starting Sensor Bridge on {PORT} at {BAUD} baud...")
    bridge = SensorBridge(port=PORT, baud=BAUD)
    
    bridge.start()
    
    gps_estimator = GPSCurvatureEstimator()
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"sensor_test_log_{timestamp_str}.csv"
    
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Latitude", "Longitude", "Speed_kmh", "Yaw_Rate", "GPS_Curvature", "Status"])
        
    print(f"Logging data to {csv_filename}")
    
    try:
        print("Waiting for data. Press Ctrl+C to stop.")
        while True:
            data = bridge.get_latest_data()
            
            yaw_rate = data.get('yaw', 0.0)
            current_speed = int(data.get('spd', 0))
            lat = data.get('lat', 0.0)
            lng = data.get('lng', 0.0)
            
            gps_curvature = gps_estimator.update(lat, lng, current_speed)
                    
            # --- SENSOR FUSION LOGIC ---
            abs_yaw = abs(yaw_rate)
            direction = "Left" if yaw_rate > 0 else "Right"
            
            # IMU is the primary classifier (fast, reliable)
            if abs_yaw < 5.0:
                current_status = "Straight"
            elif abs_yaw < 12.0:
                current_status = f"Mild Curve {direction}"
            elif abs_yaw < 22.0:
                current_status = f"Curve {direction}"
            else:
                current_status = f"Sharp {direction}"
                
            # GPS curvature can only UPGRADE the status, never downgrade it
            if gps_curvature > 0.008 and "Mild" in current_status:
                current_status = f"Curve {direction}"
            if gps_curvature > 0.02 and "Sharp" not in current_status:
                current_status = f"Sharp {direction}"
                
            # Safety Warning Logic
            if current_speed > 60 and "Sharp" in current_status: # High speed threshold
                print(f"!!! WARNING: Sharp curve ahead, slow down! (Speed: {current_speed}km/h, Yaw: {abs_yaw:.1f}, GPS Curv: {gps_curvature:.4f}) !!!")
                    
            print(f"Raw Data: {data}")
            print(f"Calculated -> Speed: {current_speed}km/h | Yaw: {yaw_rate:.2f} | GPS Curv: {gps_curvature:.4f} | Status: {current_status}")
            print("-" * 50)
            
            # Log to CSV
            with open(csv_filename, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], 
                    lat, 
                    lng, 
                    current_speed, 
                    yaw_rate,
                    gps_curvature,
                    current_status
                ])
                
            time.sleep(0.25)  # Print 4 times a second
            
    except KeyboardInterrupt:
        print("\nStopping test...")
    finally:
        bridge.stop()
        print("Done.")

if __name__ == "__main__":
    test_sensors()
