import json
import os
import sys

# Try importing folium, give helpful error if not installed
try:
    import folium
except ImportError:
    raise ImportError("Folium is missing. Please install it using: pip install folium")

def generate_map(db_path="curve_hazards.json", output_html="safeturn_map.html"):
    if not os.path.exists(db_path):
        print(f"No hazard database found at {db_path}.")
        print("Drive with SafeTurn+ first to log some curves!")
        return

    try:
        with open(db_path, "r") as f:
            hazards = json.load(f)
    except Exception as e:
        print(f"Error reading database: {e}")
        return

    if not hazards:
        print("Hazard database is empty. No map to generate.")
        return

    print(f"Loaded {len(hazards)} hazards from {db_path}.")
    
    # Filter valid coordinates to calculate average center
    valid_hazards = [h for h in hazards if h["lat"] != 0.0 and h["lon"] != 0.0]
    
    if not valid_hazards:
        print("Hazard database only contains null coordinates. No map to generate.")
        return
        
    avg_lat = sum(h["lat"] for h in valid_hazards) / len(valid_hazards)
    avg_lon = sum(h["lon"] for h in valid_hazards) / len(valid_hazards)

    # Create base map
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=14)
    
    # Add a marker for each hazard
    for h in valid_hazards:
        status = h.get("status", "Unknown")
        speed = h.get("optimal_speed", "Unknown")
        
        # Color code based on severity
        if "Pothole" in status:
            color = "black"
            icon = "exclamation-sign"
        elif "Sharp" in status:
            color = "red"
            icon = "warning-sign"
        elif "Curve" in status:
            color = "orange"
            icon = "warning-sign"
        elif "Mild" in status:
            color = "lightred"
            icon = "info-sign"
        else:
            color = "blue"
            icon = "info-sign"
            
        popup_html = f"""
        <div style="font-family: Arial; min-width: 200px;">
            <h4 style="margin-bottom: 5px;">⚠ Hazard Warning</h4>
            <b>Type:</b> {status}<br>
            <b>Suggested Speed:</b> <span style="color:{color}; font-weight:bold;">{speed} km/h</span><br>
            <hr style="margin: 5px 0;">
            <small>Lat: {h['lat']:.5f}, Lon: {h['lon']:.5f}</small>
        </div>
        """
        
        folium.Marker(
            location=[h["lat"], h["lon"]],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color=color, icon=icon),
            tooltip=f"{status}: {speed} km/h"
        ).add_to(m)

    # Save the map
    m.save(output_html)
    print(f"\n✅ Map successfully generated!")
    print(f"Open this file in your browser to view: {os.path.abspath(output_html)}")

if __name__ == "__main__":
    generate_map()
