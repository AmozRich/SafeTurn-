import time
import math
import csv
import os
from datetime import datetime
from sensor_bridge import SensorBridge
from gps_utils import GPSCurvatureEstimator, classify_curve
from config import SENSOR_PORT, SENSOR_BAUD_RATE

PORT = SENSOR_PORT
BAUD = SENSOR_BAUD_RATE

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
        csv_file = open(csv_filename, "a", newline="")
        writer = csv.writer(csv_file)
        
        print("Waiting for data. Press Ctrl+C to stop.")
        while True:
            data = bridge.get_latest_data()
            
            yaw_rate = data.get('yaw', 0.0)
            current_speed = int(data.get('spd', 0))
            lat = data.get('lat', 0.0)
            lng = data.get('lng', 0.0)
            
            gps_curvature = gps_estimator.update(lat, lng, current_speed)
                    
            # --- SENSOR FUSION LOGIC ---
            current_status = classify_curve(yaw_rate, gps_curvature)
                
            # Safety Warning Logic
            if current_speed > 60 and "Sharp" in current_status: # High speed threshold
                print(f"!!! WARNING: Sharp curve ahead, slow down! (Speed: {current_speed}km/h, Yaw: {yaw_rate:.1f}, GPS Curv: {gps_curvature:.4f}) !!!")
                    
            print(f"Raw Data: {data}")
            print(f"Calculated -> Speed: {current_speed}km/h | Yaw: {yaw_rate:.2f} | GPS Curv: {gps_curvature:.4f} | Status: {current_status}")
            print("-" * 50)
            
            # Log to CSV
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], 
                lat, 
                lng, 
                current_speed, 
                yaw_rate,
                gps_curvature,
                current_status
            ])
            csv_file.flush()
                
            time.sleep(0.25)  # Print 4 times a second
            
    except KeyboardInterrupt:
        print("\nStopping test...")
    finally:
        if 'csv_file' in locals():
            csv_file.close()
        bridge.stop()
        print("Done.")

if __name__ == "__main__":
    test_sensors()
