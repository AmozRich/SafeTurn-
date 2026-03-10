import math

class GPSCurvatureEstimator:
    def __init__(self):
        self.buffer = []

    def update(self, lat, lng, current_speed):
        if lat == 0.0 or lng == 0.0 or current_speed < 5:
            return 0.0

        add_to_buffer = False
        if not self.buffer:
            add_to_buffer = True
        else:
            last_lat, last_lng = self.buffer[-1]
            dx = (lng - last_lng) * 111320 * math.cos(math.radians((lat + last_lat) / 2))
            dy = (lat - last_lat) * 110540
            dist_to_last = math.sqrt(dx**2 + dy**2)
            if dist_to_last >= 5.0:
                add_to_buffer = True

        if add_to_buffer:
            self.buffer.append((lat, lng))
            if len(self.buffer) > 3:
                self.buffer.pop(0)

        if len(self.buffer) == 3:
            lat1, lon1 = self.buffer[0]
            lat2, lon2 = self.buffer[1]
            lat3, lon3 = self.buffer[2]

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
                rad_delta = math.radians(abs(delta_heading))
                if rad_delta > 0:
                    radius = distance / (2 * math.sin(rad_delta / 2))
                    return 1.0 / radius
        return 0.0

def classify_curve(yaw_rate, gps_curvature):
    abs_yaw = abs(yaw_rate)
    direction = "Left" if yaw_rate > 0 else "Right"
    
    # IMU is the primary classifier (fast, reliable)
    if abs_yaw < 5.0:
        status = "Straight"
    elif abs_yaw < 12.0:
        status = f"Mild Curve {direction}"
    elif abs_yaw < 22.0:
        status = f"Curve {direction}"
    else:
        status = f"Sharp {direction}"
        
    # GPS curvature can only UPGRADE the status, never downgrade it
    if gps_curvature > 0.008 and "Mild" in status:
        status = f"Curve {direction}"
    if gps_curvature > 0.02 and "Sharp" not in status:
        status = f"Sharp {direction}"
        
    return status
