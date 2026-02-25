"""
models/fft_analyzer.py
Extracts and analyzes frequency domain patterns from images using Fast Fourier Transform.
Used to detect spectral artifacts left behind by specific upsampling mechanics 
in diffusion models and GANs.
"""
import numpy as np
import cv2
from PIL import Image

class FFTAnalyzer:
    def extract_spectrum_features(self, image: Image.Image) -> dict:
        """
        Converts the image to grayscale, computes the 2D FFT, 
        and extracts key statistical features from the magnitude spectrum.
        """
        image_np = np.array(image.convert("L"))
        
        # Compute 2D FFT
        f = np.fft.fft2(image_np)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        h, w = magnitude_spectrum.shape
        cy, cx = h // 2, w // 2
        
        # Feature 1: High-Frequency Energy (edges of the spectrum)
        # Generators like Midjourney often have unusually clean/dense high-freq bands
        r_high = int(0.7 * min(cx, cy))
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        mask_high = x*x + y*y >= r_high*r_high
        high_freq_energy = np.mean(magnitude_spectrum[mask_high])
        
        # Feature 2: Mid-Frequency Variance (checkerboard artifacts from convolution)
        r_mid_in = int(0.3 * min(cx, cy))
        r_mid_out = int(0.7 * min(cx, cy))
        mask_mid = (x*x + y*y >= r_mid_in*r_mid_in) & (x*x + y*y <= r_mid_out*r_mid_out)
        mid_freq_var = np.var(magnitude_spectrum[mask_mid])
        
        # Feature 3: Azimuthal integration (radially averaged power spectrum approximation)
        # Not a full 1D PSD, but a simpler indicator of energy drop-off
        center_energy = np.mean(magnitude_spectrum[cy-5:cy+5, cx-5:cx+5])
        
        return {
            "high_freq_energy": float(high_freq_energy),
            "mid_freq_var": float(mid_freq_var),
            "center_energy": float(center_energy),
            "spectrum_ratio": float(high_freq_energy / (center_energy + 1e-5))
        }

fft_analyzer = FFTAnalyzer()
