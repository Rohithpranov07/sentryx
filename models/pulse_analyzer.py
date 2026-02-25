"""
models/pulse_analyzer.py
Calculates Remote Photoplethysmography (rPPG) and Temporal biological hooks.
These metrics are fundamentally resistant to spatial compression artifacts
because they extract aggregate frequency domain signals over time (across frames).
"""
import numpy as np

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class PulseAnalyzer:
    def __init__(self, target_fps=30.0):
        self.fps = target_fps
        self.min_pulse_hz = 0.7  # 42 BPM
        self.max_pulse_hz = 2.5  # 150 BPM

    def analyze_temporal_pulse(self, frames_green_channel_avgs: np.ndarray) -> dict:
        """
        Analyzes a sequence of average ROI color values (Green channel) to extract
        the human heartbeat. Returns physiological compliance scores.
        """
        if not SCIPY_AVAILABLE or len(frames_green_channel_avgs) < int(self.fps * 1.5):
            # Not enough data for FFT / temporal analysis, or missing scipy
            return {"pulse_snr": 0.0, "is_biologically_plausible": False, "confidence_modifier": 0.0}

        # 1. Detrend the signal (remove low frequency lighting shifts)
        detrended = signal.detrend(frames_green_channel_avgs)

        # 2. Bandpass filter for human heart rate range
        nyquist = self.fps / 2.0
        b, a = signal.butter(3, [self.min_pulse_hz / nyquist, self.max_pulse_hz / nyquist], btype='bandpass')
        filtered = signal.filtfilt(b, a, detrended)

        # 3. Calculate Signal-to-Noise Ratio (SNR) in the frequency domain
        freqs, psd = signal.welch(filtered, fs=self.fps, nperseg=len(filtered))
        
        # Find peak power within the human heart rate band
        valid_idx = np.where((freqs >= self.min_pulse_hz) & (freqs <= self.max_pulse_hz))[0]
        if len(valid_idx) == 0:
            return {"pulse_snr": 0.0, "is_biologically_plausible": False, "confidence_modifier": 0.2}

        peak_idx = valid_idx[np.argmax(psd[valid_idx])]
        peak_power = psd[peak_idx]
        total_power = np.sum(psd)
        
        snr = float(peak_power / total_power) if total_power > 0 else 0.0
        
        # Deepfakes (especially frame-by-frame temporal diffusion models)
        # scramble temporal color consistency, resulting in near-zero pulse SNR.
        
        is_plausible = snr > 0.15
        
        # If it lacks a human heartbeat, it pushes the fake confidence UP.
        modifier = 0.0 if is_plausible else 0.3

        return {
            "pulse_snr": snr,
            "estimated_bpm": float(freqs[peak_idx] * 60),
            "is_biologically_plausible": is_plausible,
            "confidence_modifier": modifier
        }

pulse_analyzer = PulseAnalyzer()
