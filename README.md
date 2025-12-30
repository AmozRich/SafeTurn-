# SafeTurn+ 🚗⚡
**Forza-Style Real-Time Hazard Warning System**

## Core Concept
SafeTurn+ is a driver assistance system that "gamifies" road safety. Just like the racing line in *Forza Horizon* changes color (Blue/Yellow/Red) to tell you if your speed is safe for a curve, SafeTurn+ uses a camera to watch the real road and tells you **exactly** how to drive it.

## Key Features
-   **👁️ Live Curve Detection**: Sees the road using a standard camera.
-   **🏎️ Speed-Adaptive Warnings**:
    -   **🔵 Blue**: Speed is safe.
    -   **🟠 Orange**: Caution, curve approaching or speed slightly high.
    -   **🔴 Red**: DANGER! Sharp curve or speed too high.
-   **🧭 Future Hazard Alerts** (Week 3): Remembers where potholes/sharp curves are (using GPS) and warns you *before* you see them next time.

## How It Works (The Pipeline)
1.  **See**: Camera captures the road.
2.  **Think**:
    -   Image processing finds lane lines.
    -   Math calculates the "Curve Radius" (how sharp is it?).
    -   System checks your speed (Simulated in W1, Real GPS in W2).
3.  **Act**:
    -   Overlays arrows on the screen in real-time.
    -   Logs hazards to the database for future warnings.

## 4-Week Roadmap
-   **Week 1**: **The "Vision"**. Building the camera system that detects lanes and overlay arrows.
-   **Week 2**: **The "Senses"**. Adding Accelerometer (bumps/potholes) and GPS (Real Speed).
-   **Week 3**: **The "Brain"**. Connecting to the Cloud to remember hazards and warn you in advance.
-   **Week 4**: **The "Polish"**. Making it run smoothly and look professional.

## Tech Stack
-   **Language**: Python 🐍
-   **Vision**: OpenCV (cv2) 📷
-   **Math**: NumPy 🧮
-   **Hardware**: Laptop + Arduino (for sensors)
