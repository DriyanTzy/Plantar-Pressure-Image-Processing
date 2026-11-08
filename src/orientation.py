"""
=============================================================================
Module: orientation.py
Pipeline Step 4 — Normalisasi Orientasi Adaptif & Invariant
=============================================================================
Fitur:
  - compute_orientation()          : Deteksi kemiringan sumbu utama via PCA.
  - rotate_and_crop()              : Rotasi lurus vertikal & crop bounding box.
  - detect_and_correct_upside_down(): Deteksi telapak kaki terbalik (jari di bawah)
                                     dan rotasi 180° agar jari selalu di atas.
  - detect_foot_side()             : Deteksi otomatis kaki Kiri (L) vs Kanan (R)
                                     berdasarkan lekukan arkus medial.
  - standardize_side()             : Standardisasi bilateral (flip horizontal).
  - normalize_orientation()        : Pipeline lengkap normalisasi orientasi.
=============================================================================
"""

import cv2
import numpy as np


def compute_orientation(mask: np.ndarray) -> tuple:
    """
    Menghitung sudut kemiringan sumbu utama kaki dari posisi vertikal
    menggunakan PCA pada koordinat piksel mask foreground.
    """
    pts = cv2.findNonZero(mask)
    if pts is None or len(pts) < 10:
        h, w = mask.shape[:2]
        return 0.0, float(w // 2), float(h // 2)
    
    pts = pts.reshape(-1, 2).astype(np.float32)
    mean, eigenvectors = cv2.PCACompute2(pts, mean=None)[:2]
    
    cx, cy = mean[0]
    vx, vy = eigenvectors[0]
    
    if vy < 0:
        vx, vy = -vx, -vy
        
    angle_rad = np.arctan2(vx, vy)
    angle_degrees = float(np.degrees(angle_rad))
    
    return angle_degrees, float(cx), float(cy)


def rotate_and_crop(image: np.ndarray, mask: np.ndarray, 
                    angle: float, center: tuple) -> tuple:
    """
    Merotasi citra dan mask agar tegak lurus vertikal,
    lalu memotong ke bounding box telapak kaki.
    """
    h, w = mask.shape[:2]
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    
    rotated_img = cv2.warpAffine(image, M, (w, h), 
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rotated_mask = cv2.warpAffine(mask, M, (w, h), 
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    _, rotated_mask = cv2.threshold(rotated_mask, 127, 255, cv2.THRESH_BINARY)
    
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


def detect_and_correct_upside_down(mask: np.ndarray, image: np.ndarray) -> tuple:
    """
    Mendeteksi apakah telapak kaki terbalik (jari di bawah, tumit di atas)
    dan merotasikannya 180° agar jari selalu berada di bagian atas.
    
    Biomekanik:
      - Forefoot (bola metatarsal + jari) secara anatomis LEBIH LEBAR (1.3x - 1.8x)
        dibandingkan tumit (heel / calcaneus).
      - Jika lebar maksimum di 35% area bawah jauh lebih besar dari area atas,
        citra dipastikan terbalik.
    """
    bh = mask.shape[0]
    if bh < 20:
        return image, mask, False
        
    top_region = mask[:int(bh * 0.35), :]
    bot_region = mask[int(bh * 0.65):, :]
    
    top_w = np.max(np.sum(top_region > 0, axis=1)) if np.any(top_region > 0) else 1
    bot_w = np.max(np.sum(bot_region > 0, axis=1)) if np.any(bot_region > 0) else 1
    
    # Jika bagian atas lebih sempit dari bagian bawah dengan margin jelas
    is_upside_down = (top_w < bot_w * 0.88)
    
    if is_upside_down:
        mask = cv2.rotate(mask, cv2.ROTATE_180)
        image = cv2.rotate(image, cv2.ROTATE_180)
        
    return image, mask, is_upside_down


def detect_foot_side(mask: np.ndarray) -> str:
    """
    Mendeteksi secara otomatis apakah telapak kaki merupakan kaki Kiri (L) atau Kanan (R)
    berdasarkan letak cekungan arkus medial (medial arch concavity).
    
    Anatomi:
      - Kaki Kiri (L): Bagian luar (lateral) lurus/pejal di sisi kiri,
                       cekungan arkus medial berada di sisi KANAN.
                       -> Area piksel sisi kiri > sisi kanan di zona midfoot.
      - Kaki Kanan (R): Bagian luar lurus/pejal di sisi kanan,
                        cekungan arkus medial berada di sisi KIRI.
                        -> Area piksel sisi kanan > sisi kiri di zona midfoot.
    """
    h, w = mask.shape[:2]
    mid_start = int(h * 0.35)
    mid_end = int(h * 0.65)
    
    mid_crop = mask[mid_start:mid_end, :]
    mid_w = mid_crop.shape[1]
    
    left_half = mid_crop[:, :mid_w // 2]
    right_half = mid_crop[:, mid_w // 2:]
    
    left_area = np.sum(left_half > 0)
    right_area = np.sum(right_half > 0)
    
    if left_area >= right_area:
        return 'L'
    else:
        return 'R'


def standardize_side(mask: np.ndarray, image: np.ndarray, 
                     side: str = None) -> tuple:
    """
    Standardisasi orientasi bilateral:
      - Kaki Kiri (L): Tidak di-flip.
      - Kaki Kanan (R): Di-flip horizontal agar konform dengan standar anatomi kaki kiri.
      - Jika side None atau 'auto', gunakan deteksi otomatis berbasis arkus.
    """
    if side is None or str(side).lower() in ['none', 'auto', 'unknown']:
        detected_side = detect_foot_side(mask)
    else:
        detected_side = str(side).upper()
        
    if detected_side == 'R':
        mask = cv2.flip(mask, 1)
        image = cv2.flip(image, 1)
        
    return image, mask, detected_side


def normalize_orientation(image: np.ndarray, mask: np.ndarray, 
                          side: str = None) -> tuple:
    """
    Pipeline normalisasi orientasi lengkap:
      1. Estimasi orientasi sumbu utama via PCA & rotasi vertikal
      2. Crop bounding box
      3. Deteksi & koreksi telapak kaki terbalik (Upside-Down / Flip Vertikal)
      4. Deteksi otomatis sisi kaki (Kiri vs Kanan) & standardisasi horizontal
    """
    # 1. PCA alignment & bounding box crop
    angle, cx, cy = compute_orientation(mask)
    rot_img, rot_mask = rotate_and_crop(image, mask, angle, (cx, cy))
    
    # 2. Koreksi jika telapak kaki terbalik (atas <-> bawah)
    upright_img, upright_mask, _ = detect_and_correct_upside_down(rot_mask, rot_img)
    
    # 3. Standardisasi sisi (kiri <-> kanan)
    final_img, final_mask, detected_side = standardize_side(upright_mask, upright_img, side)
    
    return final_img, final_mask, detected_side
