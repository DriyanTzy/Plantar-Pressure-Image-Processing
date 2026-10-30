"""
=============================================================================
Module: orientation.py
Pipeline Step 4 — Normalisasi Orientasi
=============================================================================
Fungsi utama:
  - compute_orientation() : Hitung sudut kemiringan kaki via PCA.
  - rotate_and_crop()     : Rotasi dan crop ke bounding box kaki.
  - standardize_side()    : Standardisasi kaki kiri/kanan (flip horizontal).
  - normalize_orientation(): Pipeline lengkap normalisasi orientasi.
=============================================================================
"""

import cv2
import numpy as np


def compute_orientation(mask: np.ndarray) -> tuple:
    """
    Menghitung sudut kemiringan sumbu utama kaki dari posisi vertikal
    menggunakan PCA pada koordinat piksel mask foreground.
    
    Args:
        mask: Binary mask kaki (255 = foreground).
        
    Returns:
        Tuple (angle_degrees, center_x, center_y):
            - angle_degrees: Sudut rotasi yang dibutuhkan untuk meluruskan kaki
              ke arah vertikal (90 derajat / tegak lurus).
            - center_x, center_y: Titik pusat massa kaki.
    """
    pts = cv2.findNonZero(mask)
    if pts is None or len(pts) < 10:
        h, w = mask.shape[:2]
        return 0.0, float(w // 2), float(h // 2)
    
    pts = pts.reshape(-1, 2).astype(np.float32)  # (N, 2) dengan format [x, y]
    mean, eigenvectors = cv2.PCACompute2(pts, mean=None)[:2]
    
    cx, cy = mean[0]
    vx, vy = eigenvectors[0]  # Sumbu utama variansi terbesar
    
    # Pastikan vektor mengarah ke bawah (vy positif) agar arah referensi konsisten
    if vy < 0:
        vx, vy = -vx, -vy
        
    # Hitung deviasi sudut dari sumbu vertikal (vektor [0, 1])
    # arctan2(vx, vy) menghasilkan sudut deviasi terhadap garis vertikal
    angle_rad = np.arctan2(vx, vy)
    angle_degrees = float(np.degrees(angle_rad))
    
    return angle_degrees, float(cx), float(cy)


def rotate_and_crop(image: np.ndarray, mask: np.ndarray, 
                    angle: float, center: tuple) -> tuple:
    """
    Merotasi citra dan mask agar kaki tegak lurus vertikal,
    lalu crop ke bounding box kaki dengan sedikit padding.
    
    Args:
        image: Citra input (grayscale atau RGB).
        mask: Binary mask kaki.
        angle: Sudut deviasi vertikal (dari compute_orientation).
        center: Titik pusat rotasi (cx, cy).
        
    Returns:
        Tuple (cropped_image, cropped_mask).
    """
    h, w = mask.shape[:2]
    
    # Matriks rotasi: rotasi sebesar -angle untuk meniadakan tilt deviasi
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    
    rotated_img = cv2.warpAffine(image, M, (w, h), 
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rotated_mask = cv2.warpAffine(mask, M, (w, h), 
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Binarisasi ulang mask setelah rotasi (mencegah blur pada batas akibat interpolasi)
    _, rotated_mask = cv2.threshold(rotated_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Crop ke bounding box kaki
    coords = cv2.findNonZero(rotated_mask)
    if coords is None:
        return rotated_img, rotated_mask
    
    x, y, bw, bh = cv2.boundingRect(coords)
    
    pad = 8
    x_start = max(0, x - pad)
    y_start = max(0, y - pad)
    x_end = min(w, x + bw + pad)
    y_end = min(h, y + bh + pad)
    
    cropped_img = rotated_img[y_start:y_end, x_start:x_end]
    cropped_mask = rotated_mask[y_start:y_end, x_start:x_end]
    
    return cropped_img, cropped_mask


def standardize_side(mask: np.ndarray, image: np.ndarray, 
                     side: str = 'L') -> tuple:
    """
    Standardisasi orientasi bilateral (kiri vs kanan).
    Konvensi standar:
      - Kaki kiri (L): Medial arch berada di sebelah kanan (tidak di-flip).
      - Kaki kanan (R): Di-flip secara horizontal agar konform dengan orientasi kaki kiri.
    
    Args:
        mask: Binary mask kaki.
        image: Citra kaki (grayscale atau RGB).
        side: 'L' (Left) atau 'R' (Right).
        
    Returns:
        Tuple (image, mask) yang sudah distandardisasi.
    """
    if str(side).upper() == 'R':
        mask = cv2.flip(mask, 1)
        image = cv2.flip(image, 1)
    
    return image, mask


def normalize_orientation(image: np.ndarray, mask: np.ndarray, 
                          side: str = 'L') -> tuple:
    """
    Pipeline normalisasi orientasi lengkap:
      1. Estimasi orientasi sumbu utama via PCA
      2. Rotasi tegak lurus vertikal & crop bounding box
      3. Standardisasi sisi kaki (horizontal flip jika R)
    
    Args:
        image: Citra grayscale/enhanced.
        mask: Binary mask kaki.
        side: 'L' atau 'R'.
        
    Returns:
        Tuple (normalized_image, normalized_mask).
    """
    angle, cx, cy = compute_orientation(mask)
    rot_img, rot_mask = rotate_and_crop(image, mask, angle, (cx, cy))
    final_img, final_mask = standardize_side(rot_mask, rot_img, side)
    
    return final_img, final_mask
