import json
import math
import os
from datetime import datetime

class HazardManager:
    """
    Manages a local JSON database of road hazards (sharp curves).
    Handles dynamic querying of upcoming hazards using physics-based warning distances.
    """
    def __init__(self, db_path="curve_hazards.json"):
        self.db_path = db_path
        self.hazards = []
        self.load_hazards()
        
    def load_hazards(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    self.hazards = json.load(f)
            except json.JSONDecodeError:
                self.hazards = []
                
    def save_hazards(self):
        with open(self.db_path, 'w') as f:
            json.dump(self.hazards, f, indent=4)
            
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate the great circle distance between two points on Earth in meters."""
        R = 6371000  # radius of Earth in meters
        phi_1 = math.radians(lat1)
        phi_2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = math.sin(delta_phi / 2.0) ** 2 + \
            math.cos(phi_1) * math.cos(phi_2) * \
            math.sin(delta_lambda / 2.0) ** 2
            
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
        
    def _angular_difference(self, heading1, heading2):
        """Calculate the smallest angular difference between two headings (0-360)."""
        diff = abs(heading1 - heading2) % 360
        if diff > 180:
            diff = 360 - diff
        return diff
        
    def add_hazard(self, lat, lon, curve_status, optimal_speed, heading=0.0):
        """
        Record a curve into the database. 
        Will deduplicate if within 10 meters of an existing curve.
        """
        if lat == 0.0 or lon == 0.0:
            return  # Ignore invalid coordinates
            
        # Deduplication check (10 meters)
        for h in self.hazards:
            dist = self._haversine_distance(lat, lon, h['lat'], h['lon'])
            if dist <= 10.0:
                # Only deduplicate if travelling in roughly the same direction (+/- 60 deg)
                # This ensures we can log the curve separately for the opposite direction
                if self._angular_difference(heading, h.get('heading', heading)) <= 60:
                    # Update existing if this newly driven pass suggests a lower (safer) optimal speed
                    if optimal_speed < h['optimal_speed']:
                        h['optimal_speed'] = optimal_speed
                        h['status'] = curve_status
                        h['heading'] = heading # Update with latest heading
                        h['updated_at'] = datetime.now().isoformat()
                        self.save_hazards()
                    return # Already exists or updated within 10m in the same direction
                
        # If not deduplicated, add as new
        new_hazard = {
            "lat": lat,
            "lon": lon,
            "status": curve_status,
            "optimal_speed": optimal_speed,
            "heading": heading,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        self.hazards.append(new_hazard)
        self.save_hazards()
        
    def calculate_warning_dist(self, speed_kmh):
        """
        Calculate dynamic warning radius based on stopping distance physics.
        Reaction time distance (~2.0 seconds) + Braking distance (assuming average road friction f=0.7)
        """
        speed_ms = speed_kmh * 0.278
        reaction_dist = speed_ms * 2.0 
        
        # v^2 / 2ug, u=0.7 friction coefficient, g=9.81
        braking_dist = (speed_ms ** 2) / (2 * 0.7 * 9.81)
        total_dist = reaction_dist + braking_dist
        
        # Minimum warning distance is 30m, max scaling up based on high speeds
        return max(30.0, min(total_dist, 250.0))
        
    def get_upcoming_hazard(self, current_lat, current_lon, current_speed_kmh, current_heading=0.0):
        """
        Checks if the vehicle is approaching any logged hazard too fast.
        """
        if current_lat == 0.0 or current_lon == 0.0:
            return None
            
        warning_radius = self.calculate_warning_dist(current_speed_kmh)
        
        upcoming = None
        min_dist = float('inf')
        
        for h in self.hazards:
            # Check heading first if it exists
            if 'heading' in h:
                diff = self._angular_difference(current_heading, h['heading'])
                if 60 < diff < 120:
                    continue # Ignore perpendicular intersections
            
            dist = self._haversine_distance(current_lat, current_lon, h['lat'], h['lon'])
            if dist <= warning_radius:
                # Only warn if current speed is greater than optimal speed AND it's the closest one
                if current_speed_kmh > h['optimal_speed']:
                    if dist < min_dist:
                        min_dist = dist
                        upcoming = dict(h) # Make copy to append dynamic data
                        upcoming['distance_m'] = int(dist)
                        
                        # Apply mirroring if opposite direction
                        if 'heading' in h:
                            diff = self._angular_difference(current_heading, h['heading'])
                            if diff >= 120:
                                # Mirror status
                                if "Left" in upcoming['status']:
                                    upcoming['status'] = upcoming['status'].replace("Left", "Right")
                                elif "Right" in upcoming['status']:
                                    upcoming['status'] = upcoming['status'].replace("Right", "Left")
                        
        return upcoming

    def get_recently_passed_hazard(self, current_lat, current_lon, radius_m=20.0):
        """
        Finds the closest pothole within a given radius.
        Used by the IMU window to find the pothole we just drove over.
        """
        if current_lat == 0.0 or current_lon == 0.0:
            return None
            
        closest_pothole = None
        min_dist = radius_m
        
        for h in self.hazards:
            if h.get('status') == 'Pothole':
                dist = self._haversine_distance(current_lat, current_lon, h['lat'], h['lon'])
                if dist <= min_dist:
                    min_dist = dist
                    closest_pothole = h
                    
        return closest_pothole
        
    def downgrade_pothole(self, lat, lon, speed_reduction=5):
        """Lowers the optimal speed of the correctly matched pothole due to heavy shock."""
        for h in self.hazards:
            if h.get('status') == 'Pothole' and h['lat'] == lat and h['lon'] == lon:
                new_speed = max(5, h['optimal_speed'] - speed_reduction)
                if h['optimal_speed'] != new_speed:
                    h['optimal_speed'] = new_speed
                    h['updated_at'] = datetime.now().isoformat()
                    self.save_hazards()
                    print(f"HazardManager: Pothole severity increased! New safe speed: {new_speed} km/h")
                return
