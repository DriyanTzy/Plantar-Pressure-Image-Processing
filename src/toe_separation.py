"""
=============================================================================
Module: toe_separation.py
Pipeline Step 5 — Pemisahan Jari Kaki (Toe Separation)
=============================================================================
Fungsi utama:
  - compute_width_profile() : Hitung profil lebar per baris.
  - find_toe_boundary()     : Deteksi batas antara jari dan bola kaki.
  - separate_toes()         : Pisahkan area jari dari telapak kaki.
=============================================================================
"""

import cv2
import numpy as np
from scipy.signal import find_peaks


def compute_width_profile(mask: np.ndarray) -> np.ndarray:
    """
    Menghitung jumlah piksel aktif per baris (profil lebar).
    
    Args:
        mask: Binary mask kaki (255 = foreground).
        
    Returns:
        Array 1D berisi lebar kaki di setiap baris.
    """
    widths = np.sum(mask > 0, axis=1).astype(np.float64)
    return widths


def smooth_profile(widths: np.ndarray, window: int = 25) -> np.ndarray:
    """
    Smoothing profil lebar menggunakan moving average.
    
    Args:
        widths: Array profil lebar.
        window: Ukuran window moving average.
        
    Returns:
        Array profil lebar yang sudah di-smooth.
    """
    if len(widths) < window:
        window = max(3, len(widths) // 3)
        if window % 2 == 0:
            window += 1
    
    kernel = np.ones(window) / window
    smoothed = np.convolve(widths, kernel, mode='same')
    return smoothed


def find_toe_boundary(mask: np.ndarray, fallback_ratio: float = 0.30,
                       smoothing_window: int = 25) -> int:
    """
    Mendeteksi batas antara area jari (forefoot) dan bola kaki.
    
    Metode:
    1. Hitung profil lebar dan smooth.
    2. Cari lembah (valley) di bagian atas (30% atas) profil — 
       ini menandakan lekukan antara jari dan ball of foot.
    3. Fallback: Jika tidak ditemukan lembah, gunakan rasio 
       persentase tinggi konstan dari ujung atas.
    
    Args:
        mask: Binary mask kaki (jari di atas).
        fallback_ratio: Rasio fallback dari atas jika valley tidak ditemukan.
        smoothing_window: Ukuran window untuk smoothing.
        
    Returns:
        Indeks baris yang memisahkan jari dari telapak.
    """
    widths = compute_width_profile(mask)
    smoothed = smooth_profile(widths, smoothing_window)
    
    h = mask.shape[0]
    
    # Cari valley di 35% atas gambar (area transisi jari → ball)
    search_end = int(h * 0.35)
    search_region = smoothed[:search_end]
    
    if len(search_region) < 10:
        return int(h * fallback_ratio)
    
    # Invert untuk mencari valleys sebagai peaks
    inverted = -search_region
    
    # Cari peaks (= valleys pada profil asli)
    # prominence tinggi agar hanya mendeteksi lekukan signifikan
    peaks, properties = find_peaks(inverted, 
                                    prominence=np.max(smoothed) * 0.05,
                                    distance=int(h * 0.05))
    
    if len(peaks) > 0:
        # Ambil valley terakhir (paling dekat ke ball of foot)
        toe_boundary = peaks[-1]
        return int(toe_boundary)
    
    # Fallback: gunakan rasio konstan
    return int(h * fallback_ratio)


def separate_toes(mask: np.ndarray, **kwargs) -> tuple:
    """
    Memisahkan area jari dari telapak kaki.
    
    Args:
        mask: Binary mask kaki (jari di atas).
        **kwargs: Argumen tambahan untuk find_toe_boundary().
        
    Returns:
        Tuple (toe_boundary_row, foot_without_toes_mask):
            - toe_boundary_row: Indeks baris pemisah.
            - foot_without_toes_mask: Mask telapak kaki tanpa jari.
    """
    boundary = find_toe_boundary(mask, **kwargs)
    
    # Mask telapak tanpa jari = area dari boundary ke bawah
    foot_no_toes = mask.copy()
    foot_no_toes[:boundary, :] = 0
    
    return boundary, foot_no_toes
