import serial
import csv
import time
from datetime import datetime

# --- CONFIGURATION ---
SERIAL_PORT = 'COM10'  # Replace with the ESP32's COM port
BAUD_RATE = 115200    # Must match Serial.begin() in ESP32 sketch
# -------------------

def log_breadcrumbs():
    print(f"Connecting to ESP32 on {SERIAL_PORT}...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        print("Connected! Waiting for GPS data...")
        
        # Generate a unique filename based on the current date/time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"route_breadcrumbs_{timestamp}.csv"
        
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            header_written = False
            
            while True:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8').strip()
                    
                    # Ignore empty lines or status messages from the ESP32
                    if not line or line.startswith("#"):
                        if line: 
                            print(f"[STATUS] {line}")
                        continue
                        
                    # Write the header received from ESP32
                    if line.startswith("LATITUDE"):
                        if not header_written:
                            writer.writerow(line.split(","))
                            header_written = True
                            print(f"Started logging to {filename}. Press Ctrl+C to stop.")
                        continue
                    
                    # Log the actual data row
                    data = line.split(",")
                    if len(data) == 5:
                        writer.writerow(data)
                        print(f"Logged point: Lat={data[0]}, Lng={data[1]}, Spd={data[2]}, Hdg={data[3]}")
                        
    except serial.SerialException as e:
        print(f"Error connecting to serial port: {e}")
    except KeyboardInterrupt:
        print("\nLogging stopped by user.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed.")

if __name__ == "__main__":
    log_breadcrumbs()
