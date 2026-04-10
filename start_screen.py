import tkinter as tk
from tkinter import messagebox
import time
import threading
import os
import webbrowser

class StartScreen:
    def __init__(self, bridge=None):
        self.bridge = bridge
        self.root = tk.Tk()
        self.root.title("SafeTurn+ Launcher")
        self.root.geometry("400x560")
        
        # Modern Dark Theme Colors
        self.bg_color = "#1A202C" # Dark gray/blue back
        self.fg_color = "#E2E8F0" # Light gray text
        self.accent_bg = "#2D3748" # Slightly lighter gray for frames
        self.hover_color = "#4A5568"
        
        self.root.configure(bg=self.bg_color)
        
        # User configurations
        self.use_sensors = tk.BooleanVar(value=True if bridge else False)
        self.enable_webcam = tk.BooleanVar(value=False)
        self.start_mode = "None"
        
        self.setup_ui()

    def setup_ui(self):
        # Title Header
        header_frame = tk.Frame(self.root, bg=self.bg_color, pady=15)
        header_frame.pack(fill=tk.X)
        
        title_label = tk.Label(
            header_frame, 
            text="SafeTurn+", 
            font=("Segoe UI", 26, "bold"), 
            bg=self.bg_color, 
            fg="#4299E1" # Bright blue accent
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            header_frame, 
            text="Advanced Driver Assistance System", 
            font=("Segoe UI", 10), 
            bg=self.bg_color, 
            fg="#A0AEC0"
        )
        subtitle_label.pack()
        
        # Config Panel
        config_frame = tk.Frame(self.root, bg=self.accent_bg, padx=15, pady=15, bd=0, relief=tk.FLAT)
        config_frame.pack(fill=tk.X, padx=30, pady=10)
        
        tk.Label(config_frame, text="CONFIGURATION", font=("Segoe UI", 10, "bold"), bg=self.accent_bg, fg="#718096").pack(anchor="w", pady=(0, 5))
        
        # Sensor Toggle
        sensor_cb = tk.Checkbutton(
            config_frame, 
            text="Use Hardware Sensors", 
            variable=self.use_sensors,
            command=self.on_sensor_toggle,
            font=("Segoe UI", 11),
            bg=self.accent_bg,
            fg=self.fg_color,
            selectcolor=self.bg_color,
            activebackground=self.accent_bg,
            activeforeground=self.fg_color,
            bd=0
        )
        sensor_cb.pack(anchor="w", pady=2)
        
        # Webcam Toggle and Index
        webcam_frame = tk.Frame(config_frame, bg=self.accent_bg)
        webcam_frame.pack(anchor="w", pady=2)
        
        webcam_cb = tk.Checkbutton(
            webcam_frame, 
            text="Enable Live Webcam", 
            variable=self.enable_webcam,
            font=("Segoe UI", 11),
            bg=self.accent_bg,
            fg=self.fg_color,
            selectcolor=self.bg_color,
            activebackground=self.accent_bg,
            activeforeground=self.fg_color,
            bd=0
        )
        webcam_cb.pack(side=tk.LEFT)
        
        self.webcam_index = tk.IntVar(value=1)
        tk.Label(webcam_frame, text=" Port:", bg=self.accent_bg, fg="#A0AEC0", font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=(5, 2))
        webcam_spinbox = tk.Spinbox(
            webcam_frame, 
            from_=0, 
            to=5, 
            textvariable=self.webcam_index, 
            width=3,
            font=("Segoe UI", 10),
            bg=self.bg_color,
            fg=self.fg_color,
            buttonbackground=self.accent_bg,
            bd=0
        )
        webcam_spinbox.pack(side=tk.LEFT)
        
        # Action Buttons Panel
        action_frame = tk.Frame(self.root, bg=self.bg_color, pady=10)
        action_frame.pack(fill=tk.X, padx=30)
        
        def create_button(parent, text, cmd, bg_col, fg_col="white"):
            return tk.Button(
                parent,
                text=text,
                command=cmd,
                font=("Segoe UI", 11, "bold"),
                bg=bg_col,
                fg=fg_col,
                relief=tk.FLAT,
                borderwidth=0,
                pady=10,
                activebackground=self.hover_color,
                activeforeground=fg_col
            )
            
        self.calib_btn = create_button(action_frame, "⚙ Calibrate Sensors", self.calibrate_sensors, "#ECC94B", "black")
        self.calib_btn.pack(fill=tk.X, pady=5)
        
        start_btn = create_button(action_frame, "▶ Start ADAS Drive", self.start_analysis, "#48BB78")
        start_btn.pack(fill=tk.X, pady=5)
        
        map_btn = create_button(action_frame, "🗺 View Hazard Map", self.view_map, "#4299E1")
        map_btn.pack(fill=tk.X, pady=5)
        
        pothole_btn = create_button(action_frame, "🔍 Launch Pothole Scanner", self.start_pothole, "#ED8936")
        pothole_btn.pack(fill=tk.X, pady=5)
        
        # Status Label
        self.status_label = tk.Label(
            self.root, 
            text="Ready.", 
            font=("Segoe UI", 9), 
            bg=self.bg_color, 
            fg="#718096"
        )
        self.status_label.pack(side=tk.BOTTOM, pady=15)

    def view_map(self):
        self.status_label.config(text="Generating Hazard Map...")
        self.root.update()
        try:
            import generate_hazard_map
            generate_hazard_map.generate_map()
            
            map_path = os.path.abspath("safeturn_map.html")
            if os.path.exists(map_path):
                webbrowser.open(f"file:///{map_path}")
                self.status_label.config(text="Hazard Map opened in browser.")
            else:
                self.status_label.config(text="No Map Found. Start driving first!")
        except Exception as e:
            self.status_label.config(text=f"Error loading map: {str(e)}")

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
