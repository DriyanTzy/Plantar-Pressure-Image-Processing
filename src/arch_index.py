"""
=============================================================================
Module: arch_index.py
Pipeline Step 6 — Ekstraksi Fitur: Cavanagh Arch Index
=============================================================================
Fungsi utama:
  - divide_foot_regions() : Bagi telapak kaki menjadi 3 region (A, B, C).
  - compute_arch_index()  : Hitung Arch Index = B / (A + B + C).
=============================================================================
"""

import cv2
import numpy as np


def divide_foot_regions(mask_no_toes: np.ndarray) -> dict:
    """
    Membagi panjang total telapak kaki (tanpa jari) menjadi 3 bagian 
    sama besar secara vertikal:
      A (Forefoot / Ball of Foot) — bagian atas
      B (Midfoot)                 — bagian tengah
      C (Heel / Rearfoot)         — bagian bawah
    
    Args:
        mask_no_toes: Binary mask telapak kaki TANPA jari 
                      (output dari toe_separation).
        
    Returns:
        dict dengan keys:
            'A_mask'    : mask region A
            'B_mask'    : mask region B  
            'C_mask'    : mask region C
            'A_area'    : jumlah piksel foreground di region A
            'B_area'    : jumlah piksel foreground di region B
            'C_area'    : jumlah piksel foreground di region C
            'boundaries': tuple (top, split1, split2, bottom)
    """
    h, w = mask_no_toes.shape[:2]
    
    # Cari batas atas dan bawah area kaki (non-zero rows)
    rows_with_pixels = np.where(np.any(mask_no_toes > 0, axis=1))[0]
    
    if len(rows_with_pixels) == 0:
        empty_mask = np.zeros_like(mask_no_toes)
        return {
            'A_mask': empty_mask, 'B_mask': empty_mask, 'C_mask': empty_mask,
            'A_area': 0, 'B_area': 0, 'C_area': 0,
            'boundaries': (0, 0, 0, 0)
        }
    
    top = rows_with_pixels[0]
    bottom = rows_with_pixels[-1]
    foot_length = bottom - top
    
    # Bagi menjadi 3 bagian sama besar
    third = foot_length // 3
    split1 = top + third           # Batas A-B
    split2 = top + 2 * third       # Batas B-C
    
    # Buat mask untuk setiap region
    A_mask = np.zeros_like(mask_no_toes)
    B_mask = np.zeros_like(mask_no_toes)
    C_mask = np.zeros_like(mask_no_toes)
    
    A_mask[top:split1, :] = mask_no_toes[top:split1, :]
    B_mask[split1:split2, :] = mask_no_toes[split1:split2, :]
    C_mask[split2:bottom+1, :] = mask_no_toes[split2:bottom+1, :]
    
    # Hitung area (jumlah piksel non-zero)
    A_area = cv2.countNonZero(A_mask)
    B_area = cv2.countNonZero(B_mask)
    C_area = cv2.countNonZero(C_mask)
    
    return {
        'A_mask': A_mask,
        'B_mask': B_mask,
        'C_mask': C_mask,
        'A_area': A_area,
        'B_area': B_area,
        'C_area': C_area,
        'boundaries': (top, split1, split2, bottom)
    }


def compute_arch_index(regions: dict) -> float:
    """
    Menghitung Cavanagh Arch Index:
    
        AI = B / (A + B + C)
    
    Dimana:
        A = area forefoot (ball of foot)
        B = area midfoot
        C = area heel (rearfoot)
    
    Args:
        regions: dict output dari divide_foot_regions().
        
    Returns:
        Nilai Arch Index (float). Returns -1.0 jika total area = 0.
    """
    A = regions['A_area']
    B = regions['B_area']
    C = regions['C_area']
    
    total = A + B + C
    
    if total == 0:
        return -1.0
    
    arch_index = B / total
    return round(arch_index, 4)
