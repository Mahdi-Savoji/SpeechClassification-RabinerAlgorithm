import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import librosa
import librosa.display
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time

# ==================== Default Settings ====================
DEFAULT_DURATION = 5
DEFAULT_FRAME_SIZE = 2048
DEFAULT_ENERGY_THRESHOLD = 0.01
DEFAULT_ZCR_THRESHOLD_UNVOICED_NON_SPEECH = 0.4  # Threshold for unvoiced vs non-speech
DEFAULT_ZCR_THRESHOLD_UNVOICED_VOICED = 0.15    # Threshold for unvoiced vs voiced
SR = 16000  # Sampling rate

# ==================== Functions ====================
def record_audio(duration, sr):
    """ Record audio from microphone with live timer """
    global is_recording
    is_recording = True
    status_label.config(text="Recording...", fg="blue")
    record_button.config(state="disabled")
    root.update()

    audio = np.zeros(int(duration * sr))
    recording = sd.InputStream(samplerate=sr, channels=1)
    recording.start()
    start_time = time.time()

    idx = 0
    blocksize = 1024

    while time.time() - start_time < duration:
        block, _ = recording.read(blocksize)
        end_idx = idx + len(block.flatten())
        if end_idx > len(audio):
            end_idx = len(audio)
        audio[idx:end_idx] = block.flatten()[:end_idx-idx]
        idx = end_idx
        elapsed = time.time() - start_time
        timer_label.config(text=f"Recording: {elapsed:.1f}s")
        root.update()

    recording.stop()
    timer_label.config(text="")
    status_label.config(text="Recording finished. Processing...", fg="green")
    record_button.config(state="normal")
    is_recording = False
    return audio

def analyze_audio(audio, sr, frame_size, energy_threshold, zcr_threshold_unvoiced_non_speech, zcr_threshold_unvoiced_voiced):
    """ Analyze the recorded audio """
    hop_length = frame_size // 2

    energies = np.array([np.sum(audio[i:i+frame_size]**2) for i in range(0, len(audio) - frame_size, hop_length)])
    zcrs = np.array([np.mean(librosa.zero_crossings(audio[i:i+frame_size], pad=False)) for i in range(0, len(audio) - frame_size, hop_length)])

    labels = []
    for e, z in zip(energies, zcrs):
        if e < energy_threshold:
            if z > zcr_threshold_unvoiced_non_speech:
                labels.append('unvoiced')
            else:
                labels.append('non-speech')
        else:
            if z > zcr_threshold_unvoiced_voiced:
                labels.append('unvoiced')
            else:
                labels.append('voiced')
    return labels

def plot_results(frame, audio, labels, sr, frame_size):
    """ Plot inside the tkinter window """
    hop_length = frame_size // 2
    duration = len(audio) / sr
    time_axis = np.linspace(0, duration, len(audio))

    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    # Plot waveform
    axs[0].plot(time_axis, audio, color='black')
    axs[0].set_title("Speech Signal - Time Domain", fontsize=14)
    axs[0].set_xlabel("Time (seconds)", fontsize=12)
    axs[0].set_ylabel("Amplitude", fontsize=12)

    if labels is not None:
        frame_times = np.linspace(0, duration, len(labels))
        colors = {'voiced': 'green', 'unvoiced': 'orange', 'non-speech': 'gray'}
        for i, label in enumerate(labels):
            axs[0].axvspan(frame_times[i], frame_times[i] + (hop_length / sr), color=colors[label], alpha=0.3)

        # Add legend
        legend_patches = [
            mpatches.Patch(color='green', label='Voiced'),
            mpatches.Patch(color='orange', label='Unvoiced'),
            mpatches.Patch(color='gray', label='Non-Speech')
        ]
        axs[0].legend(handles=legend_patches, loc='upper right', fontsize=10)

    # Plot spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axs[1])
    fig.colorbar(img, ax=axs[1], format="%+2.0f dB")
    axs[1].set_title("Spectrogram", fontsize=14)

    plt.tight_layout()

    # Draw inside tkinter
    global canvas
    if canvas:
        canvas.get_tk_widget().destroy()
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def start_recording_thread():
    """ Start recording in a new thread """
    t = threading.Thread(target=start_recording)
    t.start()

def start_recording():
    """ Main function when Record button is clicked """
    try:
        duration = float(duration_entry.get())
        frame_size = int(frame_size_entry.get())
        energy_threshold = float(energy_thresh_entry.get())
        zcr_threshold_unvoiced_non_speech = float(zcr_thresh_unvoiced_non_speech_entry.get())
        zcr_threshold_unvoiced_voiced = float(zcr_thresh_unvoiced_voiced_entry.get())
    except ValueError:
        status_label.config(text="Invalid settings! Please check your inputs.", fg="red")
        return

    audio = record_audio(duration, SR)
    labels = analyze_audio(audio, SR, frame_size, energy_threshold, zcr_threshold_unvoiced_non_speech, zcr_threshold_unvoiced_voiced)
    plot_results(plot_frame, audio, labels, SR, frame_size)
    status_label.config(text="Analysis Completed.", fg="green")

# ==================== Reset to Recommended Settings ====================
def reset_to_default_settings():
    """ Reset settings to recommended values """
    duration_entry.delete(0, tk.END)
    duration_entry.insert(0, str(DEFAULT_DURATION))
    frame_size_entry.delete(0, tk.END)
    frame_size_entry.insert(0, str(DEFAULT_FRAME_SIZE))
    energy_thresh_entry.delete(0, tk.END)
    energy_thresh_entry.insert(0, str(DEFAULT_ENERGY_THRESHOLD))
    zcr_thresh_unvoiced_non_speech_entry.delete(0, tk.END)
    zcr_thresh_unvoiced_non_speech_entry.insert(0, str(DEFAULT_ZCR_THRESHOLD_UNVOICED_NON_SPEECH))
    zcr_thresh_unvoiced_voiced_entry.delete(0, tk.END)
    zcr_thresh_unvoiced_voiced_entry.insert(0, str(DEFAULT_ZCR_THRESHOLD_UNVOICED_VOICED))

# ==================== GUI ====================
root = tk.Tk()
root.title("Speech Processing Application - Rabiner Algorithm")
root.geometry("1300x900")
root.configure(bg="#f0f0f0")

# Frames
main_frame = tk.Frame(root, bg="#f0f0f0")
main_frame.pack(pady=10, fill=tk.BOTH, expand=True)

settings_frame = tk.LabelFrame(main_frame, text="Settings", bg="white", padx=20, pady=10, font=("Arial", 14, "bold"))
settings_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=10)

plot_frame = tk.LabelFrame(main_frame, text="Visualization", bg="white", padx=20, pady=20, font=("Arial", 14, "bold"))
plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=10)

# Settings Widgets
tk.Label(settings_frame, text="Record Duration (s):", bg="white", font=("Arial", 12)).grid(row=0, column=0, padx=5, pady=5, sticky='e')
duration_entry = tk.Entry(settings_frame, width=10, font=("Arial", 12))
duration_entry.grid(row=0, column=1, padx=5, pady=5)
duration_entry.insert(0, str(DEFAULT_DURATION))

tk.Label(settings_frame, text="Frame Size (samples):", bg="white", font=("Arial", 12)).grid(row=0, column=2, padx=5, pady=5, sticky='e')
frame_size_entry = tk.Entry(settings_frame, width=10, font=("Arial", 12))
frame_size_entry.grid(row=0, column=3, padx=5, pady=5)
frame_size_entry.insert(0, str(DEFAULT_FRAME_SIZE))

tk.Label(settings_frame, text="Energy Threshold:", bg="white", font=("Arial", 12)).grid(row=1, column=0, padx=5, pady=5, sticky='e')
energy_thresh_entry = tk.Entry(settings_frame, width=10, font=("Arial", 12))
energy_thresh_entry.grid(row=1, column=1, padx=5, pady=5)
energy_thresh_entry.insert(0, str(DEFAULT_ENERGY_THRESHOLD))

# Add two new ZCR threshold entries
tk.Label(settings_frame, text="ZCR Threshold (Unvoiced vs Non-Speech):", bg="white", font=("Arial", 12)).grid(row=1, column=2, padx=5, pady=5, sticky='e')
zcr_thresh_unvoiced_non_speech_entry = tk.Entry(settings_frame, width=10, font=("Arial", 12))
zcr_thresh_unvoiced_non_speech_entry.grid(row=1, column=3, padx=5, pady=5)
zcr_thresh_unvoiced_non_speech_entry.insert(0, str(DEFAULT_ZCR_THRESHOLD_UNVOICED_NON_SPEECH))

tk.Label(settings_frame, text="ZCR Threshold (Unvoiced vs Voiced):", bg="white", font=("Arial", 12)).grid(row=2, column=2, padx=5, pady=5, sticky='e')
zcr_thresh_unvoiced_voiced_entry = tk.Entry(settings_frame, width=10, font=("Arial", 12))
zcr_thresh_unvoiced_voiced_entry.grid(row=2, column=3, padx=5, pady=5)
zcr_thresh_unvoiced_voiced_entry.insert(0, str(DEFAULT_ZCR_THRESHOLD_UNVOICED_VOICED))

# Button Frame for record and reset buttons
button_frame = tk.Frame(settings_frame, bg="white")
button_frame.grid(row=3, column=0, columnspan=4, pady=15, padx=5, sticky="ew")

# Record Button
record_button = tk.Button(button_frame, text="Record and Analyze", command=start_recording_thread, bg="#4CAF50", fg="white", font=("Arial", 14, "bold"))
record_button.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10)

# Reset Button
reset_button = tk.Button(button_frame, text="Reset to Recommended", command=reset_to_default_settings, bg="#FF9800", fg="white", font=("Arial", 14, "bold"))
reset_button.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10)

# Timer Label
timer_label = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="#f0f0f0", fg="blue")
timer_label.pack()

# Status Label
status_label = tk.Label(root, text="Press 'Record and Analyze' to start.", font=("Arial", 14), bg="#f0f0f0", fg="black")
status_label.pack()

# Global Canvas Holder
canvas = None
plot_results(plot_frame, np.zeros(SR*DEFAULT_DURATION), None, SR, DEFAULT_FRAME_SIZE)  # Empty initial plot

# Start GUI
root.mainloop()
