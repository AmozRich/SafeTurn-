import serial
import threading
import json
import time

class SensorBridge:
    def __init__(self, port='COM10', baud=115200):
        self.port = port
        self.baud = baud
        self.running = False
        self.latest_data = {
            "yaw": 0.0,
            "lat": 0.0,
            "lng": 0.0,
            "spd": 0.0,
            "crash": False
        }
        self.offsets = {
            "yaw": 0.0
        }
        self.lock = threading.Lock()
        self.thread = None
        self.serial_conn = None

    def start(self):
        try:
            self.serial_conn = serial.Serial(self.port, self.baud, timeout=1)
            self.running = True
            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()
            print(f"SensorBridge started on {self.port}")
        except Exception as e:
            print(f"Failed to start SensorBridge: {e}")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.serial_conn:
            self.serial_conn.close()

    def _update_loop(self):
        while self.running and self.serial_conn:
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
                print(f"Bridge Error: {e} - attempting reconnect...")
                self.serial_conn = None
                self.running = False
                time.sleep(2)
                self.start()
                break

    def calibrate(self, duration=2.0):
        print(f"Starting sensor calibration for {duration} seconds...")
        end_time = time.time() + duration
        yaw_samples = []
        while time.time() < end_time and self.running:
            with self.lock:
                yaw_samples.append(self.latest_data['yaw'])
            time.sleep(0.05)
            
        if yaw_samples:
            avg_yaw = sum(yaw_samples) / len(yaw_samples)
            with self.lock:
                self.offsets['yaw'] = avg_yaw
            print(f"Calibration complete: Yaw Offset = {avg_yaw}")
            return True
        return False

    def get_latest_data(self):
        with self.lock:
            data = self.latest_data.copy()
            data['yaw'] -= self.offsets['yaw']
            return data
