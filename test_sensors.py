import time
from sensor_bridge import SensorBridge

# Change this to your actual COM port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux/Mac)
PORT = 'COM10'
BAUD = 115200

def test_sensors():
    print(f"Starting Sensor Bridge on {PORT} at {BAUD} baud...")
    bridge = SensorBridge(port=PORT, baud=BAUD)
    
    bridge.start()
    
    try:
        print("Waiting for data. Press Ctrl+C to stop.")
        while True:
            data = bridge.get_latest_data()
            print(f"Data: {data}")
            time.sleep(0.50)  # Print 2 times a second
            
    except KeyboardInterrupt:
        print("\nStopping test...")
    finally:
        bridge.stop()
        print("Done.")

if __name__ == "__main__":
    test_sensors()
