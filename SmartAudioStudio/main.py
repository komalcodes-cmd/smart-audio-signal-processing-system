
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg
)
from scipy.signal import butter, sosfiltfilt, sosfilt


class SmartAudioStudio:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Audio Signal Processing Studio")
        self.root.geometry("1050x720")
        self.root.minsize(850, 600)
        self.root.configure(bg="#edf2f7")

        self.audio = None
        self.processed_audio = None
        self.sample_rate = None
        self.file_name = None

        self.cutoff = tk.IntVar(value=3000)
        self.status = tk.StringVar(
            value="Welcome! Open an audio file to begin."
        )

        self.build_interface()

    def build_interface(self):
        # Header
        header = tk.Frame(self.root, bg="#183153", pady=18)
        header.pack(fill="x")

        tk.Label(
            header,
            text="SMART AUDIO SIGNAL PROCESSING STUDIO",
            font=("Arial", 18, "bold"),
            bg="#183153",
            fg="white"
        ).pack()

        tk.Label(
            header,
            text="Audio Visualization • Digital Filtering • Playback",
            font=("Arial", 10),
            bg="#183153",
            fg="#dbeafe"
        ).pack(pady=(5, 0))

        # Main controls
        controls = ttk.Frame(self.root, padding=12)
        controls.pack(fill="x")

        ttk.Button(
            controls,
            text="Open Audio",
            command=self.open_audio
        ).pack(side="left", padx=4)

        ttk.Button(
            controls,
            text="Play Original",
            command=self.play_original
        ).pack(side="left", padx=4)

        ttk.Button(
            controls,
            text="Play Processed",
            command=self.play_processed
        ).pack(side="left", padx=4)

        ttk.Button(
            controls,
            text="Stop Audio",
            command=self.stop_audio
        ).pack(side="left", padx=4)

        ttk.Button(
            controls,
            text="Save Processed",
            command=self.save_audio
        ).pack(side="left", padx=4)

        # Filter controls
        filter_box = ttk.LabelFrame(
            self.root,
            text="Digital Signal Processing",
            padding=12
        )
        filter_box.pack(fill="x", padx=16, pady=5)

        ttk.Label(
            filter_box,
            text="Low-pass cutoff frequency (Hz):"
        ).pack(side="left", padx=5)

        self.cutoff_label = ttk.Label(
            filter_box, text="3000 Hz"
        )
        self.cutoff_label.pack(side="right", padx=10)

        self.cutoff_scale = ttk.Scale(
            filter_box,
            from_=100,
            to=10000,
            orient="horizontal",
            variable=self.cutoff,
            command=self.update_cutoff_label
        )
        self.cutoff_scale.pack(
            side="left", fill="x", expand=True, padx=12
        )

        ttk.Button(
            filter_box,
            text="Apply Filter",
            command=self.apply_filter
        ).pack(side="right", padx=5)

        # File information
        info_box = ttk.LabelFrame(
            self.root,
            text="Audio Information",
            padding=10
        )
        info_box.pack(fill="x", padx=16, pady=5)

        self.info_label = ttk.Label(
            info_box,
            text="No audio file selected.",
            wraplength=950
        )
        self.info_label.pack(anchor="w")

        # Graph area
        graph_box = ttk.LabelFrame(
            self.root,
            text="Signal Waveforms",
            padding=8
        )
        graph_box.pack(
            fill="both", expand=True, padx=16, pady=8
        )

        self.figure, self.axes = plt.subplots(
            2, 1, figsize=(9, 5)
        )
        self.figure.subplots_adjust(
            hspace=0.55, left=0.10, right=0.97,
            top=0.94, bottom=0.10
        )

        self.canvas = FigureCanvasTkAgg(
            self.figure, master=graph_box
        )
        self.canvas.get_tk_widget().pack(
            fill="both", expand=True
        )

        self.draw_empty_graphs()

        # Status bar
        status_bar = tk.Label(
            self.root,
            textvariable=self.status,
            anchor="w",
            bg="#dbeafe",
            fg="#183153",
            padx=12,
            pady=8
        )
        status_bar.pack(fill="x", side="bottom")

    def draw_empty_graphs(self):
        self.axes[0].set_title("Original Audio")
        self.axes[0].set_xlabel("Time (seconds)")
        self.axes[0].set_ylabel("Amplitude")
        self.axes[0].grid(True, alpha=0.3)

        self.axes[1].set_title("Processed Audio")
        self.axes[1].set_xlabel("Time (seconds)")
        self.axes[1].set_ylabel("Amplitude")
        self.axes[1].grid(True, alpha=0.3)

        self.canvas.draw()

    def update_cutoff_label(self, value):
        self.cutoff_label.config(
            text=f"{int(float(value))} Hz"
        )

    def open_audio(self):
        path = filedialog.askopenfilename(
            title="Select an audio file",
            filetypes=[
                ("Audio files", "*.wav *.flac *.ogg"),
                ("WAV files", "*.wav"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        try:
            data, sample_rate = sf.read(
                path, dtype="float32"
            )

            if data.size == 0:
                raise ValueError("The selected file is empty.")

            if not np.all(np.isfinite(data)):
                raise ValueError(
                    "The audio contains invalid sample values."
                )

            # Convert stereo audio to mono for simple analysis.
            if data.ndim == 2:
                data = np.mean(data, axis=1)

            if data.ndim != 1:
                raise ValueError("Unsupported audio format.")

            self.audio = data
            self.processed_audio = None
            self.sample_rate = int(sample_rate)
            self.file_name = path

            duration = len(data) / self.sample_rate

            self.info_label.config(
                text=(
                    f"File: {path}\n"
                    f"Sample rate: {self.sample_rate} Hz | "
                    f"Duration: {duration:.2f} seconds | "
                    f"Channels analyzed: Mono"
                )
            )

            # Keep cutoff below Nyquist frequency.
            maximum = max(100, int(self.sample_rate / 2) - 1)
            self.cutoff_scale.config(to=maximum)

            current = min(
                int(self.cutoff.get()), maximum
            )
            current = max(1, current)
            self.cutoff.set(current)
            self.update_cutoff_label(current)

            self.plot_signals()
            self.status.set("Audio loaded successfully.")

        except Exception as error:
            messagebox.showerror(
                "Audio Loading Error", str(error)
            )

    def plot_signals(self):
        if self.audio is None:
            return

        # Limit plotted samples for faster rendering.
        step = max(1, len(self.audio) // 50000)

        original = self.audio[::step]
        times_original = (
            np.arange(len(original)) * step
            / self.sample_rate
        )

        self.axes[0].clear()
        self.axes[0].plot(
            times_original, original,
            linewidth=0.6, color="#2563eb"
        )
        self.axes[0].set_title("Original Audio Waveform")
        self.axes[0].set_xlabel("Time (seconds)")
        self.axes[0].set_ylabel("Amplitude")
        self.axes[0].grid(True, alpha=0.3)

        self.axes[1].clear()

        if self.processed_audio is None:
            self.axes[1].text(
                0.5, 0.5,
                "Apply the filter to view processed audio",
                ha="center", va="center",
                transform=self.axes[1].transAxes
            )
        else:
            processed = self.processed_audio[::step]
            times_processed = (
                np.arange(len(processed)) * step
                / self.sample_rate
            )

            self.axes[1].plot(
                times_processed, processed,
                linewidth=0.6, color="#16a34a"
            )
            self.axes[1].set_title(
                "Processed Audio Waveform"
            )

        self.axes[1].set_xlabel("Time (seconds)")
        self.axes[1].set_ylabel("Amplitude")
        self.axes[1].grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def apply_filter(self):
        if self.audio is None:
            messagebox.showwarning(
                "No Audio", "Please open an audio file first."
            )
            return

        nyquist = self.sample_rate / 2
        cutoff = float(self.cutoff.get())

        if not 0 < cutoff < nyquist:
            messagebox.showerror(
                "Invalid Frequency",
                f"Cutoff must be between 0 and {nyquist:.1f} Hz."
            )
            return

        try:
            # Fourth-order Butterworth low-pass filter.
            sos = butter(
                4, cutoff, btype="lowpass",
                fs=self.sample_rate, output="sos"
            )

            if len(self.audio) > 30:
                result = sosfiltfilt(sos, self.audio)
            else:
                result = sosfilt(sos, self.audio)

            self.processed_audio = result.astype(
                np.float32
            )

            self.plot_signals()
            self.status.set(
                f"Filter applied. Cutoff: {cutoff:.0f} Hz"
            )

        except Exception as error:
            messagebox.showerror(
                "Processing Error", str(error)
            )

    def play_original(self):
        if self.audio is None:
            messagebox.showwarning(
                "No Audio", "Open an audio file first."
            )
            return

        try:
            sd.play(self.audio, self.sample_rate)
            self.status.set("Playing original audio.")
        except Exception as error:
            messagebox.showerror(
                "Playback Error", str(error)
            )

    def play_processed(self):
        if self.processed_audio is None:
            messagebox.showwarning(
                "No Processed Audio",
                "Apply the filter first."
            )
            return

        try:
            sd.play(
                self.processed_audio, self.sample_rate
            )
            self.status.set("Playing processed audio.")
        except Exception as error:
            messagebox.showerror(
                "Playback Error", str(error)
            )

    def stop_audio(self):
        try:
            sd.stop()
            self.status.set("Audio playback stopped.")
        except Exception as error:
            messagebox.showerror(
                "Audio Error", str(error)
            )

    def save_audio(self):
        if self.processed_audio is None:
            messagebox.showwarning(
                "No Processed Audio",
                "Apply the filter before saving."
            )
            return

        path = filedialog.asksaveasfilename(
            title="Save processed audio",
            defaultextension=".wav",
           
filetypes=[
    ("Audio files", "*.wav *.flac *.ogg *.mp3"),
    ("MP3 files", "*.mp3"),
    ("WAV files", "*.wav"),
    ("All files", "*.*")
]

        )

        if not path:
            return

        try:
            sf.write(
                path,
                self.processed_audio,
                self.sample_rate,
                subtype="PCM_16"
            )

            self.status.set("Processed audio saved.")
            messagebox.showinfo(
                "Success",
                f"Processed audio saved successfully:\n{path}"
            )

        except Exception as error:
            messagebox.showerror(
                "Save Error", str(error)
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = SmartAudioStudio(root)
    root.mainloop()
