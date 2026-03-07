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
                time.sleep(2)
                self.start()
                break

    def get_latest_data(self):
        with self.lock:
            return self.latest_data.copy()
