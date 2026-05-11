# SafeTurn+ System Architecture & Explanation

Here is a detailed explanation of how the **SafeTurn+** system works, breaking down its architecture, core concepts, and the important functions and tools used in its implementation.

## 1. System Overview
**SafeTurn+** is a real-time Advanced Driver Assistance System (ADAS) that acts as a copilot for drivers. It mimics the functionality found in high-end, modern vehicles by projecting safe driving paths, curve warnings, and speed advisories onto a video feed (either live webcam or pre-recorded video). 

The defining characteristic of SafeTurn+ is **Sensor Fusion**. Instead of relying purely on computer vision (which can fail in poor lighting or unmarked roads), it fuses visual data with hardware telemetry—specifically an **MPU6050 IMU** (measuring acceleration and yaw rate) and a **Neo6M GPS** (measuring speed and location)—all routed through an **ESP32 Microcontroller**.

---

## 2. Core Concepts & Pipeline

The system operates in a continuous loop inside `main.py`, executing the following pipeline for every video frame:

### A. Preprocessing & Computer Vision
The system uses **OpenCV** to isolate road markings. Because roads can have varying lighting conditions and lane colors (e.g., white or yellow lines), it uses a multi-layered approach:
*   **Color Masking:** The frame is converted to the **HLS** (Hue, Lightness, Saturation) color space to easily isolate pure white and yellow pixels, masking out the rest of the environment.
*   **CLAHE & Adaptive Thresholding:** To handle harsh lighting (glare or deep shadows), the system applies CLAHE (Contrast Limited Adaptive Histogram Equalization) on a grayscale version of the image, followed by adaptive thresholding to pull out edges even when they are faint.
*   **Canny Edge & Hough Transform:** The system runs Canny edge detection and the Hough Line Transform to mathematically find straight line segments in a dynamically cropped Region of Interest (ROI).

### B. Sensor Fusion & Stabilization
Vision alone is jittery. SafeTurn+ uses **Kalman Filters** (housed in the `LaneTracker` class) to smooth the data. 
*   **Kalman Filtering:** It predicts where the lane *should* be based on previous frames and the vehicle's physics. If a frame drops or the lane markings disappear (e.g., a "blind" spot), the Kalman filter uses the vehicle's velocity and the IMU's yaw rate to accurately guess the lane position until markings reappear.
*   **Teleport / Sanity Checks:** If the vision algorithm detects a massive jump in the lane position that violates physics (a "teleport"), the system gracefully rejects the visual data and increases the Kalman filter's uncertainty covariance, relying more on hardware sensors.

### C. Hazard Prediction & Logging
The system doesn't just react to what it sees; it learns the road.
*   **Pothole Mapping:** If the ESP32's accelerometer detects a massive vertical shock (`accel_z > 0.6`), it logs that exact GPS coordinate as a pothole. 
*   **Predictive Warnings:** The `HazardManager` constantly checks the vehicle's current GPS location and heading. If the vehicle is approaching a previously mapped hazard (like a sharp curve or a pothole) at a high speed, the system issues a warning *before* the hazard is visible on camera.

### D. Dynamic AR HUD Rendering
The user interface is not just a static overlay. It uses **Cubic Bezier Curves** to render a 3D "whip" effect. 
*   Instead of drawing straight lines to the vanishing point, the system calculates 4 control points. This allows the AR lane to stay anchored straight in front of the car's hood, but smoothly and progressively bend into the distance according to the estimated curvature of the road.

---

## 3. Key Modules and Functions

Here are the critical components that make the system tick:

### `main.py` (The Orchestrator)
*   **`detect_lane_pixels(frame, gray)`**: Fuses HLS color masking (for white/yellow lines) with adaptive grayscale thresholding to return a robust binary map of where lane pixels exist.
*   **`find_lane_boundaries(binary_lane_img)`**: Uses a histogram sliding-window approach on the bottom half of the screen to find the statistical peaks of where the left and right lanes start.
*   **`draw_minimalist_hud(...)`**: The renderer. It overlays the telemetry (speed, target speed, warnings) and paints the 3D wireframe ladder over the road using the calculated Bezier curves.
*   **`generate_cubic_bezier_points(...)`**: The math function that takes an anchor point, a top point, and two control points to generate a smooth, curved array of pixels for the AR HUD to draw.

### `SensorBridge` (`sensor_bridge.py`)
Runs asynchronously to pull JSON telemetry packets from the ESP32 via Serial. It prevents the slow 10Hz serial hardware from blocking the fast 30FPS OpenCV video pipeline. It extracts `yaw`, `spd` (speed), `lat`, `lng`, and `accel_z`.

### `LaneTracker` (`tracker.py`)
The brain of the system. 
*   **`update(...)`**: Takes the raw pixel boundaries from `main.py`, fuses it with `yaw_rate` and `speed` from the sensors, passes it through the Left and Right Kalman Filters, and outputs smoothed lane boundaries and the horizon Vanishing Point (`vp_coord`).
*   **`get_curve_status(...)`**: Calculates how sharp a curve is based on lateral velocity and sensor data, returning human-readable statuses like `"Straight"`, `"Curve"`, or `"Sharp Curve"`.

### `GPSCurvatureEstimator` (`gps_utils.py`)
Uses the history of GPS points (`lat`, `lng`) to calculate the geometric curvature of the road the vehicle is driving on. This provides a macro-level confirmation of curves that the camera might struggle to see.

### `HazardManager` (`hazard_manager.py`)
*   **`add_hazard(...)`**: Logs a dangerous curve or pothole to a local database/file.
*   **`get_upcoming_hazard(...)`**: Calculates the distance to known hazards using the Haversine formula (distance between two GPS coordinates) and triggers the AR HUD overlay if the driver needs to brake.
