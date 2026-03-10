import time
import math
import csv
import os
from datetime import datetime
from sensor_bridge import SensorBridge

# --- CONFIGURATION ---
SERIAL_PORT = 'COM10'  # Replace with the ESP32's COM port
BAUD_RATE = 115200    # Must match Serial.begin() in ESP32 sketch
# -------------------

def log_breadcrumbs():
    print(f"Starting Sensor Bridge on {SERIAL_PORT} at {BAUD_RATE} baud...")
    bridge = SensorBridge(port=SERIAL_PORT, baud=BAUD_RATE)
    bridge.start()
    
    gps_buffer = []
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"route_breadcrumbs_{timestamp_str}.csv"
    
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Latitude", "Longitude", "Speed_kmh", "Yaw_Rate", "GPS_Curvature", "Status"])
        
    print(f"Started logging to {csv_filename}. Press Ctrl+C to stop.")
    
    try:
        while True:
            data = bridge.get_latest_data()
            
            yaw_rate = data.get('yaw', 0.0)
            current_speed = int(data.get('spd', 0))
            lat = data.get('lat', 0.0)
            lng = data.get('lng', 0.0)
            
            gps_curvature = 0.0
            
            # NEO-6M Noise Mitigation: Filter stationary drift and small jumps
            if lat != 0.0 and lng != 0.0:
                add_to_buffer = False
                
                # Only buffer points if we are moving significantly (filtering 0-5 km/h false speed)
                if current_speed >= 5: 
                    if not gps_buffer:
                        add_to_buffer = True
                    else:
                        last_lat, last_lng = gps_buffer[-1]
                        # Compute literal distance in meters from the last buffered point
                        dx = (lng - last_lng) * 111320 * math.cos(math.radians((lat + last_lat) / 2))
                        dy = (lat - last_lat) * 110540
                        dist_to_last = math.sqrt(dx**2 + dy**2)
                        
                        # Only accept points at least 5 meters apart to avoid high localized curvature from jitter
                        if dist_to_last >= 5.0:
                            add_to_buffer = True

                if add_to_buffer:
                    gps_buffer.append((lat, lng))
                    if len(gps_buffer) > 3:
                        gps_buffer.pop(0)

            # Only calculate trajectory curvature if we have 3 solid points and actually moving
            if len(gps_buffer) == 3 and current_speed >= 5:
                lat1, lon1 = gps_buffer[0]
                lat2, lon2 = gps_buffer[1]
                lat3, lon3 = gps_buffer[2]

                x1 = lon1 * 111320 * math.cos(math.radians(lat1))
                y1 = lat1 * 110540
                x2 = lon2 * 111320 * math.cos(math.radians(lat2))
                y2 = lat2 * 110540
                x3 = lon3 * 111320 * math.cos(math.radians(lat3))
                y3 = lat3 * 110540

                heading1 = math.degrees(math.atan2(y2 - y1, x2 - x1))
                heading2 = math.degrees(math.atan2(y3 - y2, x3 - x2))

                delta_heading = heading2 - heading1
                delta_heading = (delta_heading + 180) % 360 - 180

                distance = math.sqrt((x3 - x2)**2 + (y3 - y2)**2)
                if distance > 0:
                    gps_curvature = abs(delta_heading) / distance
                    
            # --- SENSOR FUSION LOGIC ---
            abs_yaw = abs(yaw_rate)
            direction = "Left" if yaw_rate > 0 else "Right"
            
            current_status = "Straight"
            
            # Fuse IMU Yaw and GPS Curvature
            if abs_yaw < 2.0 and gps_curvature < 0.001:
                current_status = "Straight"
            elif abs_yaw < 10.0 and gps_curvature < 0.005:
                current_status = f"Mild Curve {direction}"
            elif abs_yaw < 20.0:
                current_status = f"Curve {direction}"
            else:
                current_status = f"Sharp {direction}"
            
            # Write breadcrumb (1Hz logging to match main.py)
            with open(csv_filename, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                    lat, 
                    lng, 
                    current_speed, 
                    yaw_rate,
                    gps_curvature,
                    current_status
                ])
                
            print(f"Logged point: Lat={lat}, Lng={lng}, Speed={current_speed}km/h, Yaw={yaw_rate:.2f}, Curv={gps_curvature:.4f}, Status={current_status}")
            time.sleep(1.0)
                        
    except KeyboardInterrupt:
        print("\nLogging stopped by user.")
    finally:
        bridge.stop()
        print("Done.")

if __name__ == "__main__":
    log_breadcrumbs()
