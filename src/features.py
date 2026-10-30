"""
=============================================================================
Module: features.py
Advanced Morphological & Pressure Feature Extraction
=============================================================================
Mengekstrak fitur-fitur biomekanik klinis tingkat lanjut dari citra plantar:
  - Cavanagh Arch Index (AI = B / (A + B + C))
  - Modified Chippaux-Smirak Index (CSI = W_midfoot / W_forefoot)
  - Staheli Index (SI = W_midfoot / W_heel)
  - Rasio Tekanan Forefoot / Rearfoot
  - Koordinat Pusat Tekanan (Center of Pressure - COP)
=============================================================================
"""

import cv2
import numpy as np


def extract_advanced_features(gray_img: np.ndarray, 
                              norm_mask: np.ndarray, 
                              toe_boundary: int, 
                              regions: dict) -> dict:
    """
    Ekstraksi kumpulan fitur komprehensif (morfologi + intensitas tekanan).
    
    Args:
        gray_img: Citra grayscale ternormalisasi (tingkat intensitas tekanan).
        norm_mask: Binary mask telapak kaki tegak lurus.
        toe_boundary: Garis pembatas jari kaki.
        regions: Dict region A, B, C dari arch_index.py.
        
    Returns:
        dict kumpulan fitur numerik bernilai tinggi untuk klasifikasi klinis.
    """
    # 1. Hitung Lebar per Region (Chippaux-Smirak & Staheli)
    A_mask = regions['A_mask']
    B_mask = regions['B_mask']
    C_mask = regions['C_mask']
    
    w_A = np.max(np.sum(A_mask > 0, axis=1)) if np.any(A_mask > 0) else 1.0
    w_B = np.min(np.sum(B_mask > 0, axis=1)[np.sum(B_mask > 0, axis=1) > 0]) if np.any(B_mask > 0) else 0.0
    w_C = np.max(np.sum(C_mask > 0, axis=1)) if np.any(C_mask > 0) else 1.0
    
    # Chippaux-Smirak Index: Lebar minimum midfoot / Lebar maksimum forefoot
    csi = float(w_B) / float(w_A) if w_A > 0 else 0.0
    
    # Staheli Index: Lebar minimum midfoot / Lebar maksimum heel
    si = float(w_B) / float(w_C) if w_C > 0 else 0.0
    
    # 2. Rasio Luas Kontak
    tot_area = max(1, regions['A_area'] + regions['B_area'] + regions['C_area'])
    area_ratio_A = float(regions['A_area']) / tot_area
    area_ratio_B = float(regions['B_area']) / tot_area
    area_ratio_C = float(regions['C_area']) / tot_area
    
    # 3. Intensitas Tekanan Rata-Rata per Region
    # Pada citra termal grayscale: intensitas piksel proporsional terhadap besaran tekanan
    press_A = float(np.mean(gray_img[A_mask > 0])) if np.any(A_mask > 0) else 0.0
    press_B = float(np.mean(gray_img[B_mask > 0])) if np.any(B_mask > 0) else 0.0
    press_C = float(np.mean(gray_img[C_mask > 0])) if np.any(C_mask > 0) else 0.0
    
    # Rasio Tekanan Midfoot terhadap Total (Pressure Index)
    tot_press = max(1.0, press_A + press_B + press_C)
    press_ratio_B = press_B / tot_press
    
    # 4. Forefoot-to-Rearfoot Pressure Ratio
    fore_to_rear_ratio = press_A / max(1.0, press_C)
    
    return {
        'csi': round(csi, 4),
        'si': round(si, 4),
        'area_ratio_A': round(area_ratio_A, 4),
        'area_ratio_B': round(area_ratio_B, 4),
        'area_ratio_C': round(area_ratio_C, 4),
        'mean_press_A': round(press_A, 2),
        'mean_press_B': round(press_B, 2),
        'mean_press_C': round(press_C, 2),
        'press_ratio_B': round(press_ratio_B, 4),
        'fore_to_rear_ratio': round(fore_to_rear_ratio, 4),
    }
