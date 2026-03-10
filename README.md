# SafeTurn+ 🚗⚡
**Advanced Driver Assistance System (ADAS) with Sensor Fusion**

SafeTurn+ is a sophisticated, real-time driver assistance system designed to mimic high-end ADAS functionality using accessible hardware. By fusing computer vision with hardware telemetry (IMU & GPS), SafeTurn+ dynamically projects safe driving paths and curve warnings directly onto your video feed.

## 🌟 Key Features

- **👁️ Live Computer Vision**: Utilizes OpenCV to track white and yellow road lanes in real-time.
- **🏎️ Sensor Fusion**: Integrates an MPU6050 (yaw rate) and Neo6M GPS (speed) via an ESP8266 NodeMCU to detect vehicle physics before the camera even sees a curve.
- **🧠 Kalman Filter Stabilization**: Drastically smooths vision-tracking jitter and recovers instantly from blank frames.
- **🚥 Speed-Adaptive HUD**:
  - **🔵 Straight**: Path is clear, high safe speed.
  - **🟠 Curve Warning**: Approach with caution.
  - **🔴 Sharp Curve/Active Maneuver**: DANGER, brake required immediately.
- **🏎️ Dynamic AR Hud**: Uses 4-point **Cubic Bezier curves** to render a progressive "whip" effect on the HUD, anchoring the trajectory to the vehicle while organically bending into distant horizons.
- **📺 Aspect Ratio Preservation**: Automatically center-crops and preserves native aspect ratios of video inputs and webcams to prevent geometric stretching during computer vision tracking.

## 🛠️ Architecture / Tech Stack

- **Software Core**: Python 3.12 🐍
- **Computer Vision**: OpenCV (`cv2`) & NumPy 🧮
- **Hardware Controller**: C++ / Arduino (Flashing ESP8266)
- **Dependency Management**: `uv`

## 🔌 Hardware Requirements

To utilize the full sensor fusion capabilities, you will need:
- **MCU**: NodeMCU ESP8266 (or ESP32)
- **IMU**: MPU6050 Accelerometer/Gyroscope
- **GPS**: Neo6M Module
- **Webcam**: Standard 720p/1080p USB Camera
- **Wiring**: I2C (D1/D2) for MPU, SoftwareSerial (D5/D6) for GPS.

*Note: The system can also run purely on the computer vision core by setting `USE_SENSORS = False` in `main.py`.*

## 🚀 Getting Started

### 1. Hardware Setup (Optional)
Upload the `hardware/5_imuandgps/imuandgps/imuandgps.ino` sketch to your ESP8266 via the Arduino IDE. Ensure your MCU is plugged into your PC via USB and that it is emitting 10Hz JSON packets: `{"yaw": 0.0, "lat": 0.0, "lng": 0.0, "spd": 0.0, "crash": false}`.

### 2. Software Setup
Using the wildly fast `uv` package manager:
```bash
# Clone the repository
git clone https://github.com/your-username/SafeTurn-Plus.git
cd SafeTurn-Plus

# Install dependencies via uv
uv add opencv-python numpy pyserial
```

### 3. Run the System
By default, the script will look for either `drive.mp4` or a live webcam (`VIDEO_PATH = None`).

```bash
uv run main.py
```

### 4. Testing the Sensor Bridge
To verify your serial telemetry is working before launching the heavy OpenCV UI:
```bash
uv run test_sensors.py
```

## 📈 Future Roadmap

- **Cloud Logging**: Log hazardous potholes mapped by the IMU accelerometer spikes tied to GPS coordinates.
- **Predictive Warnings**: Alert the driver *before* reaching a mapped hazard.
- **Lane Departure Warning (LDWS)**: Trigger audio alarms when lateral velocity crosses thresholds.
