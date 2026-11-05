"""
=============================================================================
Module: toe_separation.py
Pipeline Step 5 — Pemisahan Jari Kaki (Toe Separation)
=============================================================================
Fungsi utama:
  - compute_width_profile() : Hitung profil lebar per baris.
  - find_toe_boundary()     : Deteksi batas antara jari dan bola kaki (forefoot).
  - separate_toes()         : Pisahkan area jari dari telapak kaki.
=============================================================================
"""

import cv2
import numpy as np


def compute_width_profile(mask: np.ndarray) -> np.ndarray:
    """
    Menghitung jumlah piksel aktif per baris (profil lebar horizontal).
    
    Args:
        mask: Binary mask kaki (255 = foreground).
        
    Returns:
        Array 1D berisi lebar kaki di setiap baris.
    """
    widths = np.sum(mask > 0, axis=1).astype(np.float64)
    return widths


def smooth_profile(widths: np.ndarray, window: int = 15) -> np.ndarray:
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
    
    kernel = np.ones(window) / float(window)
    smoothed = np.convolve(widths, kernel, mode='same')
    return smoothed


def find_toe_boundary(mask: np.ndarray, max_toe_ratio: float = 0.15) -> int:
    """
    Mendeteksi batas pemisah antara jari kaki dan bola kaki (metatarsal head).
    
    Prinsip Anatomis & Cavanagh:
    1. Bola kaki (forefoot / ball of foot) adalah bagian TERLEBAR di sepertiga atas kaki.
    2. Garis pemisah jari HARUS berada DI ATAS puncak bola kaki (peak_y).
    3. Jika terdapat lekukan / lembah (valley) nyata di atas bola kaki:
       gunakan titik lembah tersebut sebagai batas jari.
    4. Jika TIDAK ada lembah (jari sudah terlepas dari kontak sensor atau tidak terekam):
       batas jari adalah puncak teratas (top) agar bantalan bola kaki TIDAK terpotong!
       
    Args:
        mask: Binary mask kaki tegak lurus (jari di atas, tumit di bawah).
        max_toe_ratio: Batas maksimal proporsi panjang jari dari total panjang kaki.
        
    Returns:
        Indeks baris pemisah jari dari telapak kaki.
    """
    rows_with_pixels = np.where(np.any(mask > 0, axis=1))[0]
    if len(rows_with_pixels) == 0:
        return 0
        
    top = rows_with_pixels[0]
    bottom = rows_with_pixels[-1]
    foot_length = bottom - top
    
    if foot_length < 30:
        return top
        
    # Periksa apakah jari merupakan kontur terpisah di bagian paling atas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 1:
        boxes = [cv2.boundingRect(c) for c in contours]
        areas = [cv2.contourArea(c) for c in contours]
        max_area = max(areas)
        
        # Urutkan kontur dari atas ke bawah
        sorted_indices = sorted(range(len(contours)), key=lambda i: boxes[i][1])
        top_idx = sorted_indices[0]
        
        # Jika kontur teratas adalah jari terpisah (luas relatif kecil dan terpisah)
        if areas[top_idx] < 0.25 * max_area and len(sorted_indices) > 1:
            next_idx = sorted_indices[1]
            # Cek jika kontur teratas berhenti sebelum/berdekatan dengan kontur berikutnya
            if boxes[top_idx][1] + boxes[top_idx][3] <= boxes[next_idx][1] + 10:
                return int(boxes[next_idx][1])

    # Analisis profil lebar vertikal
    widths = compute_width_profile(mask)
    smoothed = smooth_profile(widths, window=11)
    
    # Cari puncak bola kaki (titik terlebar di 35% atas telapak kaki)
    upper_limit = top + max(10, int(foot_length * 0.35))
    peak_in_upper = top + np.argmax(smoothed[top:upper_limit])
    
    # Jari hanya boleh dipotong DI ATAS puncak bola kaki
    if peak_in_upper - top > 10:
        above_peak = smoothed[top:peak_in_upper]
        valley_rel = np.argmin(above_peak)
        valley_y = top + valley_rel
        
        # Validasi bahwa lembah ini nyata (lebar di lembah < 75% lebar puncak bola kaki)
        # dan tidak melebihi 15% panjang kaki total
        is_significant_dip = above_peak[valley_rel] < 0.75 * smoothed[peak_in_upper]
        is_within_toe_range = (valley_y - top) <= int(foot_length * max_toe_ratio)
        
        if is_significant_dip and is_within_toe_range and valley_y > top + 3:
            return int(valley_y)
            
    # Jika tidak ada lembah anatomis, berarti citra sudah tidak memiliki jari (hanya bola kaki).
    # JANGAN memotong sembarangan agar Area A (Forefoot) tetap utuh!
    return int(top)


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
    
    foot_no_toes = mask.copy()
    if boundary > 0:
        foot_no_toes[:boundary, :] = 0
        
    return boundary, foot_no_toes
