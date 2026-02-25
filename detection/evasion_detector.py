"""
detection/evasion_detector.py
Identifies adversarial attempts to bypass forensic systems, such as:
- Metadata stripping
- Screenshot bypass
- Adversarial noise injection
- Recompression obfuscation
"""
from PIL import Image, ExifTags
import numpy as np
import cv2


class EvasionDetector:
    def __init__(self):
        # Common social media and screen resolutions to flag screenshots
        self.screenshot_dims = {(1170, 2532), (1080, 2340), (1284, 2778), (1290, 2796), (1080, 1920)}
    
    def detect_metadata_stripping(self, image: Image.Image) -> float:
        """
        Original camera photos contain EXIF data.
        Attackers routinely strip this using ExifTool or WhatsApp compression.
        """
        try:
            exif = image.getexif()
            if not exif:
                return 1.0  # Completely stripped
            if len(exif.keys()) < 3:
                return 0.8  # Heavily stripped (Suspicious)
            return 0.0
        except Exception:
            return 1.0

    def detect_screenshot_artifacts(self, image: Image.Image) -> float:
        """
        Checks if the image resolution perfectly matches known device screens,
        a common way to launder a deepfake (play it on screen, screenshot it).
        """
        w, h = image.size
        # Also check inverted for landscape
        if (w, h) in self.screenshot_dims or (h, w) in self.screenshot_dims:
            return 0.9  # High likelihood of screenshot
        
        # Check aspect ratios typical of mobile screenshots (e.g. 19.5:9)
        aspect = max(w, h) / min(w, h)
        if 2.1 < aspect < 2.2:
            return 0.7  # Suspicious aspect ratio
            
        return 0.0

    def detect_adversarial_noise(self, image_np: np.ndarray) -> float:
        """
        Looking for high-frequency, unstructured adversarial perturbations
        designed to confuse CNNs (FGSM, PGD attacks).
        """
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        
        # FFT to analyze high frequency components
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        # In natural images, high frequencies decay rapidly.
        # Adversarial noise often injects strange energy into the high-freq spectrum.
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        # Mask out low and mid frequencies (radius = 0.8 * min(cx, cy))
        r = int(0.8 * min(cx, cy))
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        mask = x*x + y*y >= r*r
        
        high_freq_energy = np.mean(magnitude_spectrum[mask])
        
        # Modern smartphones apply heavy computational sharpening, raising this baseline
        # Only flag if it's astronomically high (classic FGSM/PGD attacks hit >> 300)
        if high_freq_energy > 250.0:
            return 0.95  # Detected aggressive noise attack
        elif high_freq_energy > 180.0:
            return 0.60
            
        return 0.0

    def analyze(self, image: Image.Image) -> dict:
        """
        Run the full evasion detection sweep.
        """
        image_np = np.array(image.convert("RGB"))[:, :, ::-1]
        
        meta_score = self.detect_metadata_stripping(image)
        screen_score = self.detect_screenshot_artifacts(image)
        adv_score = self.detect_adversarial_noise(image_np)
        
        signals = []
        if meta_score > 0.8:
            signals.append("Evasion: EXIF metadata completely stripped.")
        if screen_score > 0.6:
            signals.append("Evasion: Image dimensions strongly indicate a mobile screenshot bypass.")
        if adv_score > 0.8:
            signals.append("Evasion: Aggressive high-frequency adversarial noise detected.")
            
        # Overall evasion risk. High meta_score isn't definitive proof of fake natively
        # (WhatsApp strips EXIF too), but combined with other factors it acts as a multiplier.
        evasion_risk = max(meta_score * 0.4, screen_score * 0.7, adv_score * 0.9)
        
        return {
            "evasion_risk": evasion_risk,
            "metadata_stripped": meta_score > 0.8,
            "screenshot_detected": screen_score > 0.6,
            "adversarial_noise": adv_score > 0.8,
            "signals": signals
        }

evasion_detector = EvasionDetector()
