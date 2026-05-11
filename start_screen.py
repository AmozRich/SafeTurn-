import tkinter as tk
from tkinter import messagebox
import threading
import time
import os
import webbrowser

# ─── DESIGN TOKENS ─────────────────────────────────────────────────────────────

COLORS = {
    'bg_deep':       '#0D1117',
    'bg_card':       '#161B22',
    'bg_elevated':   '#1C2333',
    'accent':        '#58A6FF',
    'accent_glow':   '#1F6FEB',
    'success':       '#3FB950',
    'warning':       '#D29922',
    'danger':        '#F85149',
    'text_primary':  '#E6EDF3',
    'text_muted':    '#8B949E',
    'border':        '#30363D',
    'toggle_on_bg':  '#238636',
    'toggle_off_bg': '#30363D',
    'toggle_knob':   '#FFFFFF',
    'knob_off':      '#8B949E',
    'orange':        '#ED8936',
}

FONT_TITLE      = ("Segoe UI", 26, "bold")
FONT_SUBTITLE   = ("Segoe UI", 10)
FONT_CARD_TITLE = ("Segoe UI", 13, "bold")
FONT_CARD_DESC  = ("Segoe UI", 9)
FONT_LABEL      = ("Segoe UI", 10)
FONT_LABEL_BOLD = ("Segoe UI", 10, "bold")
FONT_SMALL      = ("Segoe UI", 9)
FONT_BTN        = ("Segoe UI", 12, "bold")
FONT_BTN_SM     = ("Segoe UI", 10, "bold")
FONT_SECTION    = ("Segoe UI", 9, "bold")


# ─── HELPERS ───────────────────────────────────────────────────────────────────

def rounded_rect(canvas, x1, y1, x2, y2, r, **kw):
    """Draw a rounded rectangle on a tkinter Canvas using smooth polygon."""
    pts = [
        x1+r, y1, x2-r, y1, x2, y1, x2, y1+r,
        x2, y2-r, x2, y2, x2-r, y2, x1+r, y2,
        x1, y2, x1, y2-r, x1, y1+r, x1, y1,
    ]
    return canvas.create_polygon(pts, smooth=True, **kw)


def hdop_to_label(hdop):
    """Convert HDOP value to human-readable signal quality."""
    if hdop < 1:    return "Ideal"
    if hdop < 2:    return "Excellent"
    if hdop < 5:    return "Good"
    if hdop < 10:   return "Moderate"
    if hdop < 20:   return "Fair"
    return "Poor"


def hdop_to_color(hdop):
    if hdop < 5:    return COLORS['success']
    if hdop < 10:   return COLORS['warning']
    return COLORS['danger']


# ─── WIDGET: Toggle Switch ─────────────────────────────────────────────────────

class ToggleSwitch:
    W, H = 44, 24
    OFF_X, ON_X = 13, 31
    KNOB_R = 8

    def __init__(self, parent, variable, command=None, bg=None):
        self.var = variable
        self.cmd = command
        self._anim = False
        self.canvas = tk.Canvas(parent, width=self.W, height=self.H,
                                bg=bg or COLORS['bg_deep'], highlightthickness=0,
                                cursor="hand2")
        self.canvas.bind("<Button-1>", self._click)
        self._render(self.ON_X if self.var.get() else self.OFF_X)

    def _render(self, knob_x):
        c = self.canvas
        c.delete("all")
        on = knob_x > (self.W // 2)
        bg = COLORS['toggle_on_bg'] if on else COLORS['toggle_off_bg']
        kc = COLORS['toggle_knob'] if on else COLORS['knob_off']
        rounded_rect(c, 2, 2, self.W-2, self.H-2, 10, fill=bg, outline='')
        r = self.KNOB_R
        cy = self.H // 2
        c.create_oval(knob_x-r, cy-r, knob_x+r, cy+r, fill=kc, outline='')

    def _click(self, e=None):
        if self._anim: return
        self.var.set(not self.var.get())
        self._animate(self.var.get())
        if self.cmd: self.cmd()

    def _animate(self, to_on, step=0, steps=8):
        if step > steps:
            self._anim = False
            return
        self._anim = True
        t = step / steps
        t = 1 - (1-t)**2  # ease-out
        sx, ex = (self.OFF_X, self.ON_X) if to_on else (self.ON_X, self.OFF_X)
        self._render(int(sx + (ex-sx)*t))
        self.canvas.after(16, lambda: self._animate(to_on, step+1, steps))

    @property
    def widget(self):
        return self.canvas


# ─── WIDGET: Mode Card ─────────────────────────────────────────────────────────

class ModeCard:
    CW, CH = 255, 135

    def __init__(self, parent, title, desc, icon_char, on_select, bg=None):
        self.title = title
        self.desc = desc
        self.icon_char = icon_char
        self.on_select = on_select
        self.selected = False
        self.hovered = False

        self.canvas = tk.Canvas(parent, width=self.CW, height=self.CH,
                                bg=bg or COLORS['bg_deep'], highlightthickness=0,
                                cursor="hand2")
        self.canvas.bind("<Enter>", lambda e: self._set_hover(True))
        self.canvas.bind("<Leave>", lambda e: self._set_hover(False))
        self.canvas.bind("<Button-1>", lambda e: self.on_select(self))
        self._draw()

    def _set_hover(self, h):
        self.hovered = h
        self._draw()

    def set_selected(self, s):
        self.selected = s
        self._draw()

    def _draw(self):
        c = self.canvas
        c.delete("all")
        if self.selected:
            border, bg, tc = COLORS['accent'], COLORS['bg_elevated'], COLORS['text_primary']
        elif self.hovered:
            border, bg, tc = COLORS['accent'], '#1A2332', COLORS['text_primary']
        else:
            border, bg, tc = COLORS['border'], COLORS['bg_card'], COLORS['text_muted']

        # Border rect then inner fill
        rounded_rect(c, 2, 2, self.CW-2, self.CH-2, 12, fill=border, outline='')
        rounded_rect(c, 4, 4, self.CW-4, self.CH-4, 10, fill=bg, outline='')

        # Icon
        icon_color = COLORS['accent'] if (self.selected or self.hovered) else COLORS['text_muted']
        c.create_text(self.CW//2, 40, text=self.icon_char, font=("Segoe UI", 24),
                       fill=icon_color)
        # Title
        c.create_text(self.CW//2, 78, text=self.title, font=FONT_CARD_TITLE, fill=tc)
        # Description
        c.create_text(self.CW//2, 102, text=self.desc, font=FONT_CARD_DESC,
                       fill=COLORS['text_muted'])

        # Selection checkmark
        if self.selected:
            c.create_text(self.CW-18, 18, text="✓", font=("Segoe UI", 12, "bold"),
                           fill=COLORS['success'])

    @property
    def widget(self):
        return self.canvas


# ─── MAIN: StartScreen ─────────────────────────────────────────────────────────

class StartScreen:
    def __init__(self, bridge=None):
        self.bridge = bridge
        self.root = tk.Tk()
        self.root.title("SafeTurn+ Launcher")
        self.root.geometry("620x720")
        self.root.resizable(False, False)
        self.root.configure(bg=COLORS['bg_deep'])

        # Try to set dark title bar on Windows
        try:
            from ctypes import windll
            self.root.update()
            hwnd = windll.user32.GetParent(self.root.winfo_id())
            windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, byref := __import__('ctypes').byref(__import__('ctypes').c_int(2)), 4)
        except Exception:
            pass

        # State
        self.use_sensors = tk.BooleanVar(value=False)
        self.enable_webcam = tk.BooleanVar(value=False)
        self.webcam_index = tk.IntVar(value=1)
        self.start_mode = "None"
        self.selected_card = None
        self.calibrating = False

        self._build_ui()

        # Start sensor polling if bridge exists
        if self.bridge:
            self._poll_sensor_status()

    # ── UI Construction ─────────────────────────────────────────────────────

    def _build_ui(self):
        bg = COLORS['bg_deep']

        # ── HEADER ──
        hdr = tk.Frame(self.root, bg=bg)
        hdr.pack(fill=tk.X, pady=(25, 5))
        tk.Label(hdr, text="SafeTurn+", font=FONT_TITLE, bg=bg,
                 fg=COLORS['accent']).pack()
        tk.Label(hdr, text="Advanced Driver Assistance System", font=FONT_SUBTITLE,
                 bg=bg, fg=COLORS['text_muted']).pack()

        # ── Divider ──
        self._divider(self.root)

        # ── MODE CARDS ──
        tk.Label(self.root, text="SELECT MODE", font=FONT_SECTION, bg=bg,
                 fg=COLORS['text_muted']).pack(anchor="w", padx=38, pady=(10, 6))

        card_frame = tk.Frame(self.root, bg=bg)
        card_frame.pack(pady=(0, 5))

        self.card_adas = ModeCard(card_frame, "ADAS Drive",
                                   "Lane tracking & curve alerts", "⬡",
                                   self._on_card_select, bg=bg)
        self.card_adas.widget.pack(side=tk.LEFT, padx=(0, 10))

        self.card_pothole = ModeCard(card_frame, "Pothole Scanner",
                                      "Deep pothole detection", "◎",
                                      self._on_card_select, bg=bg)
        self.card_pothole.widget.pack(side=tk.LEFT)

        # ── Divider ──
        self._divider(self.root)

        # ── CONFIGURATION BAR ──
        tk.Label(self.root, text="CONFIGURATION", font=FONT_SECTION, bg=bg,
                 fg=COLORS['text_muted']).pack(anchor="w", padx=38, pady=(10, 6))

        cfg = tk.Frame(self.root, bg=COLORS['bg_card'], pady=12, padx=16)
        cfg.pack(fill=tk.X, padx=35, pady=(0, 5))

        # Sensor toggle row
        row1 = tk.Frame(cfg, bg=COLORS['bg_card'])
        row1.pack(fill=tk.X, pady=(0, 6))
        tk.Label(row1, text="Hardware Sensors", font=FONT_LABEL, bg=COLORS['bg_card'],
                 fg=COLORS['text_primary']).pack(side=tk.LEFT)
        self.toggle_sensor = ToggleSwitch(row1, self.use_sensors,
                                           command=self._on_sensor_toggle,
                                           bg=COLORS['bg_card'])
        self.toggle_sensor.widget.pack(side=tk.RIGHT)

        # Camera toggle row
        row2 = tk.Frame(cfg, bg=COLORS['bg_card'])
        row2.pack(fill=tk.X)
        tk.Label(row2, text="Live Camera", font=FONT_LABEL, bg=COLORS['bg_card'],
                 fg=COLORS['text_primary']).pack(side=tk.LEFT)

        # Webcam port (visible only when camera on)
        self.port_frame = tk.Frame(row2, bg=COLORS['bg_card'])
        tk.Label(self.port_frame, text="Port:", font=FONT_SMALL, bg=COLORS['bg_card'],
                 fg=COLORS['text_muted']).pack(side=tk.LEFT, padx=(0, 4))
        tk.Spinbox(self.port_frame, from_=0, to=5, textvariable=self.webcam_index,
                   width=3, font=FONT_SMALL, bg=COLORS['bg_deep'],
                   fg=COLORS['text_primary'], buttonbackground=COLORS['bg_card'],
                   bd=0, highlightthickness=0).pack(side=tk.LEFT)

        self.toggle_cam = ToggleSwitch(row2, self.enable_webcam,
                                        command=self._on_cam_toggle,
                                        bg=COLORS['bg_card'])
        self.toggle_cam.widget.pack(side=tk.RIGHT)

        # ── Divider ──
        self._divider(self.root)

        # ── SENSOR STATUS PANEL ──
        tk.Label(self.root, text="SENSOR STATUS", font=FONT_SECTION, bg=bg,
                 fg=COLORS['text_muted']).pack(anchor="w", padx=38, pady=(10, 6))

        self.sensor_panel = tk.Frame(self.root, bg=COLORS['bg_card'], padx=16, pady=12)
        self.sensor_panel.pack(fill=tk.X, padx=35, pady=(0, 5))

        # Status grid: IMU | GPS | Satellites | Signal
        status_grid = tk.Frame(self.sensor_panel, bg=COLORS['bg_card'])
        status_grid.pack(fill=tk.X, pady=(0, 8))

        self.imu_dot, self.imu_label = self._status_cell(status_grid, "IMU", 0)
        self.gps_dot, self.gps_label = self._status_cell(status_grid, "GPS", 1)
        self.sat_label_val = self._info_cell(status_grid, "Satellites", "—", 2)
        self.sig_label_val = self._info_cell(status_grid, "Signal", "—", 3)

        # Calibration area
        calib_frame = tk.Frame(self.sensor_panel, bg=COLORS['bg_deep'], padx=12, pady=10)
        calib_frame.pack(fill=tk.X, pady=(4, 0))

        calib_header = tk.Frame(calib_frame, bg=COLORS['bg_deep'])
        calib_header.pack(fill=tk.X, pady=(0, 6))
        tk.Label(calib_header, text="Calibration", font=FONT_LABEL_BOLD,
                 bg=COLORS['bg_deep'], fg=COLORS['text_primary']).pack(side=tk.LEFT)

        self.calib_status = tk.Label(calib_header, text="Not started", font=FONT_SMALL,
                                      bg=COLORS['bg_deep'], fg=COLORS['text_muted'])
        self.calib_status.pack(side=tk.RIGHT)

        # Progress bar canvas
        self.prog_canvas = tk.Canvas(calib_frame, width=510, height=18,
                                      bg=COLORS['bg_deep'], highlightthickness=0)
        self.prog_canvas.pack(fill=tk.X, pady=(0, 6))
        self._draw_progress(0)

        # Live readout
        self.calib_readout = tk.Label(calib_frame, text="Yaw: —   |   Z-Axis: —   |   Samples: 0",
                                       font=FONT_SMALL, bg=COLORS['bg_deep'],
                                       fg=COLORS['text_muted'])
        self.calib_readout.pack(anchor="w")

        # Calibrate button
        self.calib_btn = tk.Button(calib_frame, text="⚙  Calibrate",
                                    font=FONT_BTN_SM, bg=COLORS['warning'],
                                    fg="#000000", relief=tk.FLAT, bd=0, pady=4, padx=16,
                                    cursor="hand2", activebackground="#E5A825",
                                    command=self._start_calibration)
        self.calib_btn.pack(anchor="w", pady=(6, 0))
        self._bind_hover(self.calib_btn, COLORS['warning'], "#E5A825")

        # ── ACTION BUTTONS ──
        btn_frame = tk.Frame(self.root, bg=bg)
        btn_frame.pack(fill=tk.X, padx=35, pady=(15, 0))

        self.launch_btn = tk.Button(btn_frame, text="▶   L A U N C H",
                                     font=FONT_BTN, bg=COLORS['accent_glow'],
                                     fg="#FFFFFF", relief=tk.FLAT, bd=0, pady=12,
                                     cursor="hand2", activebackground=COLORS['accent'],
                                     command=self._launch, state=tk.DISABLED,
                                     disabledforeground="#555555")
        self.launch_btn.pack(fill=tk.X, pady=(0, 8))
        self._bind_hover(self.launch_btn, COLORS['accent_glow'], COLORS['accent'])

        map_btn = tk.Button(btn_frame, text="🗺  View Hazard Map",
                             font=FONT_BTN_SM, bg=COLORS['bg_card'],
                             fg=COLORS['text_muted'], relief=tk.FLAT, bd=0, pady=8,
                             cursor="hand2", activebackground=COLORS['bg_elevated'],
                             command=self._view_map)
        map_btn.pack(fill=tk.X)
        self._bind_hover(map_btn, COLORS['bg_card'], COLORS['bg_elevated'])

        # ── Footer ──
        tk.Label(self.root, text="SafeTurn+ v2.0  ·  Sensor Fusion ADAS",
                 font=("Segoe UI", 8), bg=bg, fg=COLORS['border']).pack(side=tk.BOTTOM, pady=10)

    # ── UI Helpers ──────────────────────────────────────────────────────────

    def _divider(self, parent):
        tk.Frame(parent, bg=COLORS['border'], height=1).pack(fill=tk.X, padx=35, pady=4)

    def _status_cell(self, parent, name, col):
        f = tk.Frame(parent, bg=COLORS['bg_card'])
        f.grid(row=0, column=col, padx=(0, 20), sticky="w")
        dot = tk.Canvas(f, width=10, height=10, bg=COLORS['bg_card'], highlightthickness=0)
        dot.pack(side=tk.LEFT, padx=(0, 5))
        dot.create_oval(1, 1, 9, 9, fill=COLORS['text_muted'], outline='', tags="dot")
        lbl = tk.Label(f, text=f"{name}: Offline", font=FONT_SMALL,
                        bg=COLORS['bg_card'], fg=COLORS['text_muted'])
        lbl.pack(side=tk.LEFT)
        return dot, lbl

    def _info_cell(self, parent, name, default, col):
        f = tk.Frame(parent, bg=COLORS['bg_card'])
        f.grid(row=0, column=col, padx=(0, 20), sticky="w")
        tk.Label(f, text=f"{name}: ", font=FONT_SMALL, bg=COLORS['bg_card'],
                 fg=COLORS['text_muted']).pack(side=tk.LEFT)
        val = tk.Label(f, text=default, font=FONT_SMALL, bg=COLORS['bg_card'],
                        fg=COLORS['text_primary'])
        val.pack(side=tk.LEFT)
        return val

    def _bind_hover(self, btn, normal_bg, hover_bg):
        """Add hover color change to a button."""
        btn.bind("<Enter>", lambda e: btn.configure(bg=hover_bg) if btn['state'] != 'disabled' else None)
        btn.bind("<Leave>", lambda e: btn.configure(bg=normal_bg) if btn['state'] != 'disabled' else None)

    def _draw_progress(self, pct):
        c = self.prog_canvas
        c.delete("all")
        w, h = 510, 18
        r = 8
        # Track
        rounded_rect(c, 0, 0, w, h, r, fill=COLORS['border'], outline='')
        # Fill
        if pct > 0:
            fw = max(2*r, int(w * pct / 100))
            color = COLORS['success'] if pct >= 100 else COLORS['accent']
            rounded_rect(c, 0, 0, fw, h, r, fill=color, outline='')
        # Percent text
        c.create_text(w//2, h//2, text=f"{pct}%", font=FONT_SMALL,
                       fill=COLORS['text_primary'])

    def _update_dot(self, dot_canvas, color):
        dot_canvas.delete("dot")
        dot_canvas.create_oval(1, 1, 9, 9, fill=color, outline='', tags="dot")

    # ── Card Selection ──────────────────────────────────────────────────────

    def _on_card_select(self, card):
        # Deselect both, select clicked
        self.card_adas.set_selected(card is self.card_adas)
        self.card_pothole.set_selected(card is self.card_pothole)
        self.selected_card = card

        # Enable launch
        self.launch_btn.configure(state=tk.NORMAL, bg=COLORS['accent_glow'])

    # ── Sensor Toggle ───────────────────────────────────────────────────────

    def _on_sensor_toggle(self):
        if self.use_sensors.get():
            if self.bridge:
                if not self.bridge.running:
                    self.bridge.start()
                self.imu_label.configure(text="IMU: Connecting...", fg=COLORS['warning'])
                self._update_dot(self.imu_dot, COLORS['warning'])
        else:
            if self.bridge and self.bridge.running:
                self.bridge.stop()
            self.imu_label.configure(text="IMU: Offline", fg=COLORS['text_muted'])
            self.gps_label.configure(text="GPS: Offline", fg=COLORS['text_muted'])
            self._update_dot(self.imu_dot, COLORS['text_muted'])
            self._update_dot(self.gps_dot, COLORS['text_muted'])
            self.sat_label_val.configure(text="—")
            self.sig_label_val.configure(text="—", fg=COLORS['text_primary'])

    def _on_cam_toggle(self):
        if self.enable_webcam.get():
            self.port_frame.pack(side=tk.RIGHT, padx=(0, 10))
        else:
            self.port_frame.pack_forget()

    # ── Sensor Polling ──────────────────────────────────────────────────────

    def _poll_sensor_status(self):
        if not self.use_sensors.get() or not self.bridge:
            self.root.after(1000, self._poll_sensor_status)
            return

        try:
            connected = self.bridge.is_connected
            data = self.bridge.get_latest_data() if connected else None

            if connected and data:
                # IMU status (if yaw != exactly 0, we're getting data)
                self.imu_label.configure(text="IMU: Connected", fg=COLORS['success'])
                self._update_dot(self.imu_dot, COLORS['success'])

                # GPS status
                lat = data.get('lat', 0.0)
                lng = data.get('lng', 0.0)
                if lat != 0.0 or lng != 0.0:
                    self.gps_label.configure(text="GPS: Fix", fg=COLORS['success'])
                    self._update_dot(self.gps_dot, COLORS['success'])
                else:
                    self.gps_label.configure(text="GPS: Acquiring...", fg=COLORS['warning'])
                    self._update_dot(self.gps_dot, COLORS['warning'])

                # Satellites
                sat = data.get('sat', 0)
                self.sat_label_val.configure(text=str(sat))

                # Signal quality
                hdop = data.get('hdop', 99.9)
                self.sig_label_val.configure(text=hdop_to_label(hdop),
                                              fg=hdop_to_color(hdop))
            elif connected:
                self.imu_label.configure(text="IMU: Connected", fg=COLORS['success'])
                self._update_dot(self.imu_dot, COLORS['success'])
            else:
                self.imu_label.configure(text="IMU: Connecting...", fg=COLORS['warning'])
                self._update_dot(self.imu_dot, COLORS['warning'])
        except Exception:
            pass

        self.root.after(500, self._poll_sensor_status)

    # ── Calibration ─────────────────────────────────────────────────────────

    def _start_calibration(self):
        if self.calibrating:
            return
        if not self.use_sensors.get() or not self.bridge:
            messagebox.showwarning("Sensors Offline",
                                    "Enable hardware sensors first.")
            return
        if not self.bridge.is_connected:
            messagebox.showwarning("Not Connected",
                                    "Waiting for sensor connection...")
            return

        self.calibrating = True
        self.calib_btn.configure(state=tk.DISABLED, text="Calibrating...")
        self.calib_status.configure(text="Keep vehicle stationary...", fg=COLORS['warning'])
        self._draw_progress(0)

        def progress_cb(pct, yaw, az, samples):
            # Schedule GUI update on main thread
            self.root.after(0, lambda: self._update_calib_gui(pct, yaw, az, samples))

        def run():
            try:
                ok = self.bridge.calibrate(duration=3.0, on_progress=progress_cb)
                def finish():
                    self.calibrating = False
                    if ok:
                        self.calib_status.configure(text="✓ Calibration complete",
                                                     fg=COLORS['success'])
                        self.calib_btn.configure(state=tk.NORMAL, text="⚙  Recalibrate")
                    else:
                        self.calib_status.configure(text="✗ Failed", fg=COLORS['danger'])
                        self.calib_btn.configure(state=tk.NORMAL, text="⚙  Calibrate")
                self.root.after(0, finish)
            except Exception as e:
                def err():
                    self.calibrating = False
                    self.calib_status.configure(text=f"Error: {e}", fg=COLORS['danger'])
                    self.calib_btn.configure(state=tk.NORMAL, text="⚙  Calibrate")
                self.root.after(0, err)

        threading.Thread(target=run, daemon=True).start()

    def _update_calib_gui(self, pct, yaw, az, samples):
        self._draw_progress(pct)
        self.calib_readout.configure(
            text=f"Yaw: {yaw:.2f}°/s   |   Z-Axis: {az:.2f}G   |   Samples: {samples}")

    # ── Actions ─────────────────────────────────────────────────────────────

    def _launch(self):
        if self.selected_card is self.card_adas:
            self.start_mode = "SafeTurn"
        elif self.selected_card is self.card_pothole:
            self.start_mode = "Pothole"
        else:
            return
        self.root.destroy()

    def _view_map(self):
        try:
            import generate_hazard_map
            generate_hazard_map.generate_map()
            map_path = os.path.abspath("safeturn_map.html")
            if os.path.exists(map_path):
                webbrowser.open(f"file:///{map_path}")
        except Exception as e:
            messagebox.showerror("Map Error", str(e))

    # ── Public Interface (unchanged signature) ──────────────────────────────

    def show(self):
        self.root.mainloop()
        return (self.start_mode, self.use_sensors.get(),
                self.enable_webcam.get(), self.webcam_index.get())


if __name__ == "__main__":
    screen = StartScreen()
    mode, sensors, webcam, idx = screen.show()
    print(f"Mode: {mode}, Sensors: {sensors}, Webcam: {webcam}, Index: {idx}")
