#!/usr/bin/env python3
import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2
import pandas as pd
import pyrealsense2 as rs
from sklearn.preprocessing import LabelEncoder
from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import splrep, splev
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import pickle
import joblib

# ---------------------- PIPELINE FUNCTIONS ----------------------
def remove_outliers(data):
    data = np.array(data)
    if len(data) < 5:
        return data
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 3*iqr, q3 + 3*iqr
    return np.clip(data, lower, upper)

def butterworth_lowpass(data, cutoff=0.7, fs=30, order=4):
    data = remove_outliers(np.array(data))
    if len(data) < max(10, order*3):
        return np.array(data, dtype=float)
    nyq = 0.5*fs
    b, a = butter(order, cutoff/nyq, btype='low')
    return filtfilt(b, a, data)

def bspline_fit(data, smooth_factor_ratio=0.02):
    data = remove_outliers(np.array(data))
    x = np.arange(len(data))
    if len(x) < 5:
        return data
    s = max(0.0, smooth_factor_ratio*len(data)*np.var(data))
    tck = splrep(x, data, s=s)
    return splev(x, tck)

def estimate_fs(timestamps):
    if len(timestamps) < 2:
        return 30.0
    diffs = np.diff(np.array(timestamps))
    diffs = diffs[diffs>0]
    return 1.0 / np.median(diffs) if len(diffs) else 30.0

def compute_features(peaks, timestamps):
    if len(peaks) < 2:
        return 0,0,0,0,0
    peak_times = np.array([timestamps[p] for p in peaks])
    intervals = np.diff(peak_times)
    bpm = 60.0 / np.mean(intervals)
    avg_interval = np.mean(intervals)
    std_interval = np.std(intervals)
    cycle_length = peak_times[-1] - peak_times[0]
    insp_exp_ratio = np.mean(intervals)/np.mean(intervals)  # placeholder
    return bpm, avg_interval, std_interval, cycle_length, insp_exp_ratio

def compute_relative_index(chest_data, abdomen_data):
    """
    Compute the relative index (chest-to-abdomen movement ratio).
    
    Returns:
        - relative_index: ratio of chest amplitude to abdomen amplitude
        - chest_amplitude: range of chest movement
        - abdomen_amplitude: range of abdomen movement
        - synchrony: correlation between chest and abdomen signals (0-1)
    """
    chest = np.array(chest_data)
    abdomen = np.array(abdomen_data)
    
    # Calculate movement amplitudes
    chest_amplitude = np.ptp(chest)  # peak-to-peak
    abdomen_amplitude = np.ptp(abdomen)
    
    # Relative index (ratio)
    if abdomen_amplitude > 0:
        relative_index = chest_amplitude / abdomen_amplitude
    else:
        relative_index = 0.0
    
    # Compute synchrony (correlation coefficient)
    if len(chest) > 1 and len(abdomen) > 1:
        correlation = np.corrcoef(chest, abdomen)[0, 1]
        synchrony = abs(correlation)  # 0-1 range
    else:
        synchrony = 0.0
    
    return relative_index, chest_amplitude, abdomen_amplitude, synchrony

def extract_features_and_predict(chest_raw, abdomen_raw, timestamps, xgb_model, label_encoder, butter_cutoff, bspline_factor):
    fs = estimate_fs(timestamps)
    
    # Apply Butterworth lowpass
    chest_filt = butterworth_lowpass(chest_raw, cutoff=butter_cutoff, fs=fs)
    abdomen_filt = butterworth_lowpass(abdomen_raw, cutoff=butter_cutoff, fs=fs)
    
    # Apply B-spline smoothing
    chest_bs = bspline_fit(chest_filt, smooth_factor_ratio=bspline_factor)
    abdomen_bs = bspline_fit(abdomen_filt, smooth_factor_ratio=bspline_factor)
    
    # Compute relative index
    rel_index, chest_amp, abdomen_amp, synchrony = compute_relative_index(chest_bs, abdomen_bs)
    
    # Peak detection
    chest_peaks, _ = find_peaks(chest_bs)
    abdomen_peaks, _ = find_peaks(abdomen_bs)
    
    # Compute features
    chest_features = compute_features(chest_peaks, timestamps)
    abdomen_features = compute_features(abdomen_peaks, timestamps)
    
    if xgb_model is None or label_encoder is None:
        raise ValueError("Pretrained model and label encoder must be provided.")
    
    # Use only chest features for prediction
    features = np.array(chest_features).reshape(1, -1)
    pred_encoded = xgb_model.predict(features)[0]
    disease = label_encoder.inverse_transform([pred_encoded])[0]
    
    return (disease, chest_bs, abdomen_bs, chest_peaks, abdomen_peaks, 
            chest_features, abdomen_features, rel_index, chest_amp, abdomen_amp, synchrony)

# ---------------------- GUI APPLICATION ----------------------
class DepthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chest & Abdomen Depth Analyzer with Disease Prediction")
        self.root.geometry("1350x1000")
        self.root.configure(bg="#1e1e1e")
        
        import pandas as pd
        label_classes = ['Eupnoea', 'Tachypnoea', 'Bradypnoea', 'Apnoea', 'Hyperpnoea', 
                        'Kussmauls', 'Cheyne-Stokes', 'Biots', 'Apneustic']
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(label_classes)
        print("✅ label encoder loaded successfully.")
        
        # Load pretrained XGBoost model
        self.xgb_model = joblib.load("xgboost.pkl")
        print("✅ Pretrained model loaded successfully.")
        
        # State
        self.file_path = None
        self.current_selection = "chest"
        self.chest_coord = None
        self.abdomen_coord = None
        self.depth_values_chest = []
        self.depth_values_abdomen = []
        self.timestamps_ms = []
        self.latest_frame = None
        self.pipeline = None
        self.playback = None
        
        self.butter_cutoff = tk.DoubleVar(value=0.7)
        self.bspline_factor = tk.DoubleVar(value=0.02)
        self.max_frames = tk.IntVar(value=150)
        
        # Top Controls
        topbar = tk.Frame(root, bg="#2e2e2e", pady=10)
        topbar.pack(fill="x")
        
        tk.Button(topbar, text="Open .bag", font=("Arial", 14, "bold"), bg="#4CAF50", 
                 fg="white", command=self.browse_file).pack(side="left", padx=12)
        
        self.status_label = tk.Label(topbar, text="Open a .bag file to begin.", 
                                     bg="#2e2e2e", fg="white", font=("Arial", 12, "bold"))
        self.status_label.pack(side="left", padx=12)
        
        # Scrollable Content
        container = tk.Frame(root, bg="#1e1e1e")
        container.pack(fill="both", expand=True)
        
        self.canvas = tk.Canvas(container, bg="#1e1e1e", highlightthickness=0)
        vscroll = tk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        
        self.scroll_frame = tk.Frame(self.canvas, bg="#1e1e1e")
        self.scroll_frame.bind("<Configure>", 
                              lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0,0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=vscroll.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")
        
        # Video Card
        self.video_card = tk.Frame(self.scroll_frame, bg="#2a2a2a")
        self.video_card.pack(pady=20, padx=20)
        self.video_card.configure(width=800, height=450)
        self.video_card.pack_propagate(False)
        self.add_shadow(self.video_card)
        
        self.video_label = tk.Label(self.video_card, bg="#000000", bd=0)
        self.video_label.pack(expand=True, fill="both")
        self.video_label.bind("<Button-1>", self.on_frame_click)
        
        # Sliders Card
        self.slider_card = tk.Frame(self.scroll_frame, bg="#2a2a2a")
        self.slider_card.pack(pady=15, padx=20, fill="x")
        self.slider_card.configure(height=200)
        self.add_shadow(self.slider_card)
        
        slider_frame = tk.Frame(self.slider_card, bg="#2a2a2a")
        slider_frame.pack(expand=True, fill="both", padx=15, pady=10)
        
        # Max Frames Slider
        tk.Label(slider_frame, text="Max Frames to Process (30 fps)", bg="#2a2a2a", 
                fg="white", font=("Arial", 11, "bold")).pack()
        frames_slider = tk.Scale(slider_frame, from_=90, to=600, resolution=30, 
                                orient="horizontal", variable=self.max_frames, 
                                bg="#3e3e3e", fg="white", highlightbackground="#2a2a2a")
        frames_slider.pack(fill="x", padx=5, pady=5)
        
        self.frames_info = tk.Label(slider_frame, text="150 frames ≈ 5 seconds", 
                                    bg="#2a2a2a", fg="#aaaaaa", font=("Arial", 9))
        self.frames_info.pack()
        frames_slider.config(command=self.update_frame_info)
        
        tk.Label(slider_frame, text="Butterworth Cutoff (Hz)", bg="#2a2a2a", 
                fg="white", font=("Arial", 11, "bold")).pack(pady=(10,0))
        tk.Scale(slider_frame, from_=0.1, to=5.0, resolution=0.1, orient="horizontal", 
                variable=self.butter_cutoff, bg="#3e3e3e", fg="white", 
                highlightbackground="#2a2a2a", 
                command=lambda val: self.update_plots()).pack(fill="x", padx=5, pady=5)
        
        tk.Label(slider_frame, text="B-spline Smooth Factor", bg="#2a2a2a", 
                fg="white", font=("Arial", 11, "bold")).pack()
        tk.Scale(slider_frame, from_=0.0, to=0.1, resolution=0.005, orient="horizontal", 
                variable=self.bspline_factor, bg="#3e3e3e", fg="white", 
                highlightbackground="#2a2a2a", 
                command=lambda val: self.update_plots()).pack(fill="x", padx=5, pady=5)
        
        # Metrics Card
        self.metrics_card = tk.Frame(self.scroll_frame, bg="#2a2a2a")
        self.metrics_card.pack(pady=10, padx=20, fill="x")
        self.add_shadow(self.metrics_card)
        
        self.metrics_label = tk.Label(self.metrics_card, text="Metrics will appear here", 
                                      font=("Arial", 12, "bold"), bg="#2a2a2a", fg="white")
        self.metrics_label.pack(padx=15, pady=10)
        
        # Plots Card
        self.plots_card = tk.Frame(self.scroll_frame, bg="#2a2a2a")
        self.plots_card.pack(pady=20, padx=20, fill="both", expand=True)
        self.add_shadow(self.plots_card)
        
        self.plots_holder = tk.Frame(self.plots_card, bg="#2a2a2a")
        self.plots_holder.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Reset Button
        tk.Button(self.scroll_frame, text="Reset", font=("Arial", 12, "bold"), 
                 bg="#f44336", fg="white", command=self.reset_all).pack(pady=15)
    
    def add_shadow(self, frame):
        frame.config(highlightbackground="#000000", highlightthickness=1)
        frame.config(bd=0, relief="flat")
    
    def update_frame_info(self, val):
        frames = int(float(val))
        seconds = frames / 30.0
        self.frames_info.config(text=f"{frames} frames ≈ {seconds:.1f} seconds")
    
    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("RealSense bag", "*.bag")])
        if not path:
            return
        self.reset_all(clear_file=False)
        self.file_path = path
        self.status_label.config(text="File selected. Click on CHEST in the frame below.")
        self.current_selection = "chest"
        self.capture_first_frame()
    
    def reset_all(self, clear_file=True):
        if self.pipeline:
            self.pipeline.stop()
        for child in self.plots_holder.winfo_children():
            child.destroy()
        if clear_file:
            self.file_path = None
        self.current_selection = "chest"
        self.chest_coord = None
        self.abdomen_coord = None
        self.depth_values_chest = []
        self.depth_values_abdomen = []
        self.timestamps_ms = []
        self.status_label.config(text="Open a .bag file to begin.")
        self.metrics_label.config(text="Metrics will appear here")
    
    def capture_first_frame(self):
        if not self.file_path:
            return
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(self.file_path, repeat_playback=False)
        config.enable_stream(rs.stream.depth, rs.format.z16, 30)
        self.profile = self.pipeline.start(config)
        self.playback = self.profile.get_device().as_playback()
        self.playback.set_real_time(False)
        
        frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        depth = frames.get_depth_frame()
        depth_image = np.asanyarray(depth.get_data())
        colorized = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), 
                                     cv2.COLORMAP_JET)
        self.latest_frame = colorized
        
        img_rgb = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.frame_image = ImageTk.PhotoImage(img_pil)
        self.video_label.config(image=self.frame_image)
    
    def on_frame_click(self, event):
        if self.latest_frame is None:
            return
        x, y = event.x, event.y
        frame_h, frame_w = self.latest_frame.shape[:2]
        label_w, label_h = self.video_label.winfo_width(), self.video_label.winfo_height()
        scale_x, scale_y = frame_w / label_w, frame_h / label_h
        mapped_x, mapped_y = int(x*scale_x), int(y*scale_y)
        
        if self.current_selection=="chest":
            self.chest_coord = (mapped_x,mapped_y)
            self.status_label.config(text=f"Chest @ {self.chest_coord}. Click ABDOMEN.")
            self.current_selection="abdomen"
        elif self.current_selection=="abdomen":
            self.abdomen_coord = (mapped_x,mapped_y)
            self.status_label.config(text=f"Abdomen @ {self.abdomen_coord}. Processing depth...")
            self.extract_depth_values()
            self.update_plots()
    
    def extract_depth_values(self):
        self.depth_values_chest = []
        self.depth_values_abdomen = []
        self.timestamps_ms = []
        
        self.pipeline.stop()
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(self.file_path, repeat_playback=False)
        config.enable_stream(rs.stream.depth, rs.format.z16, 30)
        self.profile = self.pipeline.start(config)
        self.playback = self.profile.get_device().as_playback()
        self.playback.set_real_time(False)
        
        frame_count = 0
        MAX_FRAMES = self.max_frames.get()
        
        try:
            while frame_count < MAX_FRAMES:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                depth = frames.get_depth_frame()
                if not depth:
                    break
                
                cx, cy = self.chest_coord
                ax, ay = self.abdomen_coord
                
                self.depth_values_chest.append(depth.get_distance(cx, cy))
                self.depth_values_abdomen.append(depth.get_distance(ax, ay))
                self.timestamps_ms.append(depth.get_timestamp()/1000.0)
                
                frame_count += 1
                
                if frame_count % 30 == 0:
                    self.status_label.config(text=f"Processing... {frame_count}/{MAX_FRAMES} frames")
                    self.root.update()
        except RuntimeError:
            pass
        
        self.pipeline.stop()
        self.status_label.config(text="Processing complete!")
    
    def update_plots(self):
        for child in self.plots_holder.winfo_children():
            child.destroy()
        
        if not self.depth_values_chest or not self.depth_values_abdomen:
            return
        
        try:
            result = extract_features_and_predict(
                self.depth_values_chest,
                self.depth_values_abdomen,
                self.timestamps_ms,
                xgb_model=self.xgb_model,
                label_encoder=self.label_encoder,
                butter_cutoff=self.butter_cutoff.get(),
                bspline_factor=self.bspline_factor.get()
            )
            
            (disease, chest_bs, abdomen_bs, chest_peaks, abdomen_peaks, 
             chest_feat, abdomen_feat, rel_index, chest_amp, abdomen_amp, synchrony) = result
            
        except Exception as e:
            self.metrics_label.config(text=f"Error: {e}")
            return
        
        # Update metrics with relative index
        self.metrics_label.config(
            text=f"Predicted Disease: {disease}\n"
                 f"Relative Index (Chest/Abdomen): {rel_index:.3f}\n"
                 f"Chest Amplitude: {chest_amp:.4f}m | Abdomen Amplitude: {abdomen_amp:.4f}m\n"
                 f"Breathing Synchrony: {synchrony:.3f} (0=asynchronous, 1=synchronous)\n"
                 f"Chest Features (BPM, Avg, Std, Cycle, Insp/Exp): {chest_feat}\n"
                 f"Abdomen Features (BPM, Avg, Std, Cycle, Insp/Exp): {abdomen_feat}"
        )
        
        # Plot signals
        fig = Figure(figsize=(10,8), dpi=100)
        axs=[[fig.add_subplot(2,2,1), fig.add_subplot(2,2,2)],
             [fig.add_subplot(2,2,3), fig.add_subplot(2,2,4)]]
        
        fig.suptitle(
            f"Depth Signals | Relative Index: {rel_index:.3f} | Synchrony: {synchrony:.3f}\n"
            f"(Cutoff={self.butter_cutoff.get():.2f}Hz, B-spline={self.bspline_factor.get():.3f})",
            fontsize=11, color="white"
        )
        
        def plot_signal(ax, data, peaks, label):
            ax.plot(data, label=label, color='cyan')
            ax.plot(peaks, np.array(data)[peaks], "ro")
            ax.set_title(label)
            ax.set_xlabel("Frame")
            ax.set_ylabel("Depth (m)")
            ax.set_facecolor("#2e2e2e")
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.title.set_color("white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.legend(facecolor="#2a2a2a", edgecolor="white", labelcolor="white")
        
        plot_signal(axs[0][0], self.depth_values_chest, chest_peaks, "Chest - Raw")
        plot_signal(axs[0][1], self.depth_values_abdomen, abdomen_peaks, "Abdomen - Raw")
        plot_signal(axs[1][0], chest_bs, chest_peaks, "Chest - Smoothed")
        plot_signal(axs[1][1], abdomen_bs, abdomen_peaks, "Abdomen - Smoothed")
        
        fig.patch.set_facecolor("#2a2a2a")
        fig.tight_layout(rect=[0,0.03,1,0.96])
        
        canvas = FigureCanvasTkAgg(fig, master=self.plots_holder)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

# ---------------------- MAIN ----------------------
if __name__=="__main__":
    root = tk.Tk()
    app = DepthApp(root)
    root.mainloop()
