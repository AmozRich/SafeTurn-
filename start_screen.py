import tkinter as tk
from tkinter import messagebox
import time
import threading

class StartScreen:
    def __init__(self, bridge=None):
        self.bridge = bridge
        self.root = tk.Tk()
        self.root.title("SafeTurn+ Launcher")
        self.root.geometry("400x350")
        self.root.configure(bg="#2c3e50")
        
        # User configurations
        self.use_sensors = tk.BooleanVar(value=True if bridge else False)
        self.enable_webcam = tk.BooleanVar(value=False)
        self.start_mode = "None"
        
        self.setup_ui()

    def setup_ui(self):
        # Title
        title_label = tk.Label(
            self.root, 
            text="SafeTurn+", 
            font=("Helvetica", 24, "bold"), 
            bg="#2c3e50", 
            fg="#ecf0f1"
        )
        title_label.pack(pady=20)
        
        # Options Frame
        options_frame = tk.Frame(self.root, bg="#2c3e50")
        options_frame.pack(pady=10)
        
        # Sensor Toggle
        sensor_cb = tk.Checkbutton(
            options_frame, 
            text="Use Hardware Sensors", 
            variable=self.use_sensors,
            command=self.on_sensor_toggle,
            font=("Helvetica", 12),
            bg="#2c3e50",
            fg="#ecf0f1",
            selectcolor="#34495e",
            activebackground="#2c3e50",
            activeforeground="#ecf0f1"
        )
        sensor_cb.pack(anchor="w", pady=5)
        
        # Webcam Toggle and Index
        webcam_frame = tk.Frame(options_frame, bg="#2c3e50")
        webcam_frame.pack(anchor="w", pady=5)
        
        webcam_cb = tk.Checkbutton(
            webcam_frame, 
            text="Enable Live Webcam", 
            variable=self.enable_webcam,
            font=("Helvetica", 12),
            bg="#2c3e50",
            fg="#ecf0f1",
            selectcolor="#34495e",
            activebackground="#2c3e50",
            activeforeground="#ecf0f1"
        )
        webcam_cb.pack(side=tk.LEFT)
        
        self.webcam_index = tk.IntVar(value=1)
        tk.Label(webcam_frame, text=" Index:", bg="#2c3e50", fg="#ecf0f1", font=("Helvetica", 12)).pack(side=tk.LEFT, padx=(10, 2))
        webcam_spinbox = tk.Spinbox(
            webcam_frame, 
            from_=0, 
            to=5, 
            textvariable=self.webcam_index, 
            width=3,
            font=("Helvetica", 12)
        )
        webcam_spinbox.pack(side=tk.LEFT)
        
        # Calibration Button
        self.calib_btn = tk.Button(
            self.root, 
            text="Calibrate Sensors", 
            command=self.calibrate_sensors,
            font=("Helvetica", 12, "bold"),
            bg="#f39c12",
            fg="white",
            relief=tk.FLAT,
            width=20
        )
        self.calib_btn.pack(pady=15)
        
        # Start Button
        start_btn = tk.Button(
            self.root, 
            text="Start Curve Analysis", 
            command=self.start_analysis,
            font=("Helvetica", 14, "bold"),
            bg="#27ae60",
            fg="white",
            relief=tk.FLAT,
            width=20
        )
        start_btn.pack(pady=10)
        
        # Pothole Button
        pothole_btn = tk.Button(
            self.root, 
            text="Launch Pothole Scanner", 
            command=self.start_pothole,
            font=("Helvetica", 14, "bold"),
            bg="#d35400",
            fg="white",
            relief=tk.FLAT,
            width=20
        )
        pothole_btn.pack(pady=10)
        
        # Status Label
        self.status_label = tk.Label(
            self.root, 
            text="", 
            font=("Helvetica", 10), 
            bg="#2c3e50", 
            fg="#bdc3c7"
        )
        self.status_label.pack(side=tk.BOTTOM, pady=10)

    def on_sensor_toggle(self):
        if self.use_sensors.get() and self.bridge and not self.bridge.running:
            self.bridge.start()
            self.status_label.config(text="Connecting to sensors...")
        elif not self.use_sensors.get() and self.bridge and self.bridge.running:
            self.bridge.stop()
            self.status_label.config(text="Hardware sensors disabled.")

    def calibrate_sensors(self):
        if not self.use_sensors.get() or not self.bridge:
            messagebox.showwarning("Warning", "Hardware sensors are not enabled or connected.")
            return
            
        self.calib_btn.config(state=tk.DISABLED, text="Calibrating...")
        self.status_label.config(text="Please keep the vehicle stationary...")
        
        # Run calibration in separate thread to not freeze UI
        def calib_thread():
            try:
                self.bridge.calibrate(duration=2.0)
                self.root.after(0, lambda: self.status_label.config(text="Calibration successful!"))
                self.root.after(0, lambda: self.calib_btn.config(state=tk.NORMAL, text="Re-Calibrate"))
            except Exception as e:
                self.root.after(0, lambda: self.status_label.config(text=f"Calibration failed: {e}"))
                self.root.after(0, lambda: self.calib_btn.config(state=tk.NORMAL, text="Calibrate Sensors"))
                
        threading.Thread(target=calib_thread, daemon=True).start()

    def start_analysis(self):
        self.start_mode = "SafeTurn"
        self.root.destroy()

    def start_pothole(self):
        self.start_mode = "Pothole"
        self.root.destroy()
        
    def show(self):
        self.root.mainloop()
        return self.start_mode, self.use_sensors.get(), self.enable_webcam.get(), self.webcam_index.get()

if __name__ == "__main__":
    # Test UI
    screen = StartScreen()
    start_mode, use_sensors, use_webcam, webcam_index = screen.show()
    print(f"Start Mode: {start_mode}, Sensors: {use_sensors}, Webcam: {use_webcam}, Index: {webcam_index}")
