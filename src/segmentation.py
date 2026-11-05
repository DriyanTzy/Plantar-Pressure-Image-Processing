"""
=============================================================================
Module: segmentation.py
Pipeline Step 3 — Segmentasi
=============================================================================
Fungsi utama:
  - otsu_threshold()            : Binarisasi Otsu foreground/background.
  - morphological_refinement()  : Closing & opening lembut untuk reduksi noise.
  - filter_foot_components()    : Pertahankan semua komponen kaki >= 10% komponen
                                  terbesar (mencegah tumit/heel terbuang pada high arch).
  - largest_connected_component(): Isolasi kontur terbesar (backward compatibility).
  - segment_foot()              : Pipeline segmentasi terpadu.
=============================================================================
"""

import cv2
import numpy as np


def otsu_threshold(gray: np.ndarray) -> np.ndarray:
    """
    Otsu's Thresholding untuk memisahkan area kaki dari background.
    
    Args:
        gray: Gambar grayscale yang sudah dibersihkan/denoised.
        
    Returns:
        Binary mask (0/255).
    """
    _, binary = cv2.threshold(gray, 0, 255, 
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def morphological_refinement(binary: np.ndarray, 
                             close_kernel_size: tuple = (5, 5),
                             open_kernel_size: tuple = (5, 5)) -> np.ndarray:
    """
    Pembersihan morfologi lembut tanpa melebarkan / menjembatani lengkungan midfoot.
    
    1. Closing dengan kernel kecil (5x5): menutup lubang-lubang kecil internal.
    2. Opening dengan kernel kecil (5x5): membersihkan noise bintik kecil.
    
    Args:
        binary: Binary mask input.
        close_kernel_size: Ukuran kernel closing (w, h).
        open_kernel_size: Ukuran kernel opening (w, h).
        
    Returns:
        Binary mask yang sudah di-refine.
    """
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_kernel_size)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
    
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, open_kernel_size)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)
    
    return opened


def filter_foot_components(binary: np.ndarray, 
                           min_area_ratio: float = 0.10,
                           min_area: float = 150.0) -> np.ndarray:
    """
    Mempertahankan semua komponen telapak kaki yang signifikan (luas >= 10%
    dari komponen terbesar).
    
    PENTING UNTUK HIGH ARCH:
    Pada high arch, kontak midfoot terputus sehingga forefoot dan heel menjadi
    dua komponen terpisah. Jangan hanya mengambil komponen terbesar, karena
    tumit akan terbuang dan membuat pengukuran Arch Index salah total.
    
    Args:
        binary: Binary mask input.
        min_area_ratio: Rasio minimal luas terhadap komponen terbesar (default: 0.10).
        min_area: Luas piksel absolut minimum untuk mengabaikan noise.
        
    Returns:
        Binary mask berisi seluruh komponen kaki yang valid.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binary
        
    areas = [cv2.contourArea(c) for c in contours]
    max_area = max(areas)
    threshold_area = max(min_area, min_area_ratio * max_area)
    
    mask = np.zeros_like(binary)
    kept_contours = [c for c, a in zip(contours, areas) if a >= threshold_area]
    
    if kept_contours:
        cv2.drawContours(mask, kept_contours, -1, 255, -1)
    else:
        # Fallback jika semua di bawah threshold
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, -1)
        
    return mask


def largest_connected_component(binary: np.ndarray) -> np.ndarray:
    """
    Ambil hanya kontur/komponen terbesar (backward compatibility).
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binary
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    return mask


def segment_foot(gray_or_enhanced: np.ndarray, 
                 close_kernel: tuple = (5, 5), 
                 open_kernel: tuple = (5, 5),
                 min_area_ratio: float = 0.10) -> np.ndarray:
    """
    Pipeline segmentasi lengkap:
    Otsu Threshold -> Morphological Refinement -> Multi-Component Foot Filter.
    
    Args:
        gray_or_enhanced: Gambar grayscale (filtered).
        close_kernel: Ukuran kernel morphological closing (default: 5x5).
        open_kernel: Ukuran kernel morphological opening (default: 5x5).
        min_area_ratio: Ambang batas kontur sekunder (heel pada high arch).
        
    Returns:
        Binary mask kaki yang sudah bersih dan utuh.
    """
    binary = otsu_threshold(gray_or_enhanced)
    refined = morphological_refinement(binary, close_kernel, open_kernel)
    foot_mask = filter_foot_components(refined, min_area_ratio=min_area_ratio)
    
    return foot_mask
