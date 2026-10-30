"""
=============================================================================
Module: segmentation.py
Pipeline Step 3 — Segmentasi
=============================================================================
Fungsi utama:
  - otsu_threshold()            : Binarisasi Otsu foreground/background.
  - morphological_refinement()  : Closing (menghubungkan midfoot) & opening.
  - largest_connected_component(): Isolasi kontur telapak kaki utama.
  - segment_foot()              : Pipeline segmentasi terpadu.
=============================================================================
"""

import cv2
import numpy as np


def otsu_threshold(enhanced: np.ndarray) -> np.ndarray:
    """
    Otsu's Thresholding untuk memisahkan area kaki dari background.
    
    Args:
        enhanced: Gambar grayscale yang sudah di-enhance (output CLAHE).
        
    Returns:
        Binary mask (0/255).
    """
    _, binary = cv2.threshold(enhanced, 0, 255, 
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def morphological_refinement(binary: np.ndarray, 
                             close_kernel_size: tuple = (15, 21),
                             open_kernel_size: tuple = (5, 5)) -> np.ndarray:
    """
    Penanganan midfoot yang terputus (kritis untuk high arch).
    
    1. Morphological Closing dengan kernel elips agak besar secara vertikal: 
       Menyambungkan komponen midfoot yang terputus agar tidak terbuang saat
       ekstraksi komponen terbesar.
    2. Morphological Opening:
       Membersihkan sisa noise kecil setelah closing.
    
    Args:
        binary: Binary mask input.
        close_kernel_size: Ukuran kernel closing (w, h).
        open_kernel_size: Ukuran kernel opening (w, h).
        
    Returns:
        Binary mask yang sudah di-refine.
    """
    # Closing: menyambungkan komponen isthmus yang terputus
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_kernel_size)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
    
    # Opening: bersihkan noise bintik
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, open_kernel_size)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)
    
    return opened


def largest_connected_component(binary: np.ndarray) -> np.ndarray:
    """
    Ambil hanya kontur/komponen terbesar (bentuk kaki utuh).
    Abaikan artefak kecil di sekitarnya.
    
    Args:
        binary: Binary mask input.
        
    Returns:
        Binary mask hanya berisi komponen terbesar.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return binary
    
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    
    return mask


def segment_foot(enhanced: np.ndarray, 
                 close_kernel: tuple = (15, 21), 
                 open_kernel: tuple = (5, 5)) -> np.ndarray:
    """
    Pipeline segmentasi lengkap:
    Otsu Threshold -> Morphological Refinement -> Largest Component.
    
    Args:
        enhanced: Gambar CLAHE-enhanced.
        close_kernel: Ukuran kernel morphological closing.
        open_kernel: Ukuran kernel morphological opening.
        
    Returns:
        Binary mask kaki yang sudah bersih.
    """
    binary = otsu_threshold(enhanced)
    refined = morphological_refinement(binary, close_kernel, open_kernel)
    foot_mask = largest_connected_component(refined)
    
    return foot_mask
