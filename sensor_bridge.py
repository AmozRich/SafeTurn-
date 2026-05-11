import serial
import threading
import json
import time
from config import SENSOR_PORT, SENSOR_BAUD_RATE

class SensorBridge:
    def __init__(self, port=SENSOR_PORT, baud=SENSOR_BAUD_RATE):
        self.port = port
        self.baud = baud
        self.running = False
        self.latest_data = {
            "yaw": 0.0,
            "lat": 0.0,
            "lng": 0.0,
            "spd": 0.0,
            "accel_z": 1.0,
            "crash": False
        }
        self.offsets = {
            "yaw": 0.0,
            "accel_z": 1.0
        }
        self.lock = threading.Lock()
        self.thread = None
        self.serial_conn = None

    @property
    def is_connected(self):
        return self.serial_conn is not None and self.serial_conn.is_open

    def start(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        print(f"SensorBridge thread started. Awaiting connection on {self.port}")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.serial_conn:
            self.serial_conn.close()

    def _update_loop(self):
        while self.running:
            if not self.is_connected:
                try:
                    self.serial_conn = serial.Serial(self.port, self.baud, timeout=1)
                    print(f"SensorBridge connected on {self.port}")
                except Exception as e:
                    print(f"Bridge reconnect failed: {e}. Retrying in 2s...")
                    time.sleep(2)
                    continue

            try:
                if self.serial_conn.in_waiting:
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    if line:
                        try:
                            # Parse JSON
                            data = json.loads(line)
                            
                            # Thread-safe update
                            with self.lock:
                                self.latest_data = data
                                
                        except json.JSONDecodeError:
                            pass # Corrupt packet
            except Exception as e:
                print(f"Bridge Error: {e} - connection lost.")
                if self.serial_conn:
                    try:
                        self.serial_conn.close()
                    except:
                        pass
                self.serial_conn = None
                time.sleep(2)

    def calibrate(self, duration=2.0):
        print(f"Starting sensor calibration for {duration} seconds...")
        end_time = time.time() + duration
        yaw_samples = []
        az_samples = []
        while time.time() < end_time and self.running:
            with self.lock:
                yaw_samples.append(self.latest_data.get('yaw', 0.0))
                az_samples.append(self.latest_data.get('accel_z', 1.0))
            time.sleep(0.05)
            
        if yaw_samples:
            avg_yaw = sum(yaw_samples) / len(yaw_samples)
            avg_az = sum(az_samples) / len(az_samples) if az_samples else 1.0
            with self.lock:
                self.offsets['yaw'] = avg_yaw
                self.offsets['accel_z'] = avg_az
            print(f"Calibration complete: Yaw = {avg_yaw:.2f}, Z-Axis = {avg_az:.2f}G")
            return True
        return False

    def get_latest_data(self):
        with self.lock:
            data = self.latest_data.copy()
            data['yaw'] = data.get('yaw', 0.0) - self.offsets['yaw']
            data['accel_z'] = data.get('accel_z', 1.0) - self.offsets['accel_z']
            return data
