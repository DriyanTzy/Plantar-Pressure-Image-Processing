"""
=============================================================================
Module: preprocessing.py
Pipeline Steps 1 & 2 — Akuisisi, Inspeksi Data, dan Preprocessing
Mendukung:
  1. Citra Tekanan Termal / Jet Colormap (Dataset Sensor Digital)
  2. Citra Hitam Putih / Monokrom (Scanned Podogram, Ink Footprint, Siluet B&W)
  3. Format Latar Terang (Kertas Putih) maupun Latar Gelap (Mat Hitam)
=============================================================================
"""

import os
import cv2
import numpy as np


def detect_and_normalize_modality(img: np.ndarray, target_height: int = 640) -> dict:
    """
    Mendeteksi modalitas citra (Warna vs Hitam Putih) dan polaritas latar belakang
    (Latar Terang/Kertas Putih vs Latar Gelap/Mat Hitam), serta standardisasi skala.
    
    Args:
        img: Citra input BGR atau Grayscale (dari cv2.imread).
        target_height: Standar tinggi resolusi kerja (default 640px).
        
    Returns:
        dict berisi citra yang telah dinormalisasi polaritasnya dan metadata modalitas.
    """
    orig_h, orig_w = img.shape[:2]
    
    # 1. Standardisasi Skala Resolusi Kerja (agar kernel morfologi bekerja konsisten)
    if orig_h != target_height and orig_h > 0:
        scale = target_height / float(orig_h)
        target_w = max(100, int(orig_w * scale))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(img, (target_w, target_height), interpolation=interp)
    else:
        scale = 1.0
        resized = img.copy()
        target_w = orig_w
        
    # 2. Deteksi Warna (Thermal Colormap) vs Monokrom / Hitam-Putih
    if len(resized.shape) == 3 and resized.shape[2] == 3:
        b, g, r = cv2.split(resized)
        color_diff = float(np.mean(np.abs(r.astype(float) - g.astype(float)) + 
                                   np.abs(g.astype(float) - b.astype(float))))
        is_color = color_diff > 12.0
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        is_color = False
        gray = resized.copy() if len(resized.shape) == 2 else cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
    # 3. Deteksi Polaritas Background (Terang vs Gelap)
    # Analisis 5% piksel perimeter bingkai tepi terluar
    h, w = gray.shape[:2]
    pad_h = max(2, int(h * 0.05))
    pad_w = max(2, int(w * 0.05))
    border_pixels = np.concatenate([
        gray[:pad_h, :].flatten(),
        gray[-pad_h:, :].flatten(),
        gray[:, :pad_w].flatten(),
        gray[:, -pad_w:].flatten(),
    ])
    mean_border = float(np.mean(border_pixels))
    
    # Jika rata-rata border > 127: Latar belakang adalah KERTAS PUTIH / TERANG
    # Telapak kaki adalah area TINTA GELAP (0) -> Harus di-invert
    is_inverted = mean_border > 127.0
    
    if is_inverted:
        gray_norm = 255 - gray
    else:
        gray_norm = gray.copy()
        
    # 4. Deteksi apakah citra murni siluet biner (Hitam-Putih keras tanpa gradasi)
    non_zero = gray_norm[gray_norm > 20]
    if len(non_zero) > 0:
        mid_tones = np.sum((non_zero > 35) & (non_zero < 220))
        is_binary = float(mid_tones / len(non_zero)) < 0.15
    else:
        is_binary = False
        
    # Keterangan deskriptif untuk laporan
    if is_color:
        modality_str = "Peta Tekanan Termal / Jet Colormap (Warna)"
    elif is_binary and is_inverted:
        modality_str = "Jejak Tinta / Podogram Kertas Putih (Inverted B&W)"
    elif is_binary:
        modality_str = "Siluet Biner Telapak Kaki (Hitam Putih)"
    elif is_inverted:
        modality_str = "Scan Dokumen / Foto Monokrom Latar Terang"
    else:
        modality_str = "Citra Intensitas Tekanan Grayscale"
        
    return {
        'resized_bgr': resized if len(resized.shape) == 3 else cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR),
        'gray_norm': gray_norm,
        'is_color': is_color,
        'is_inverted': is_inverted,
        'is_binary': is_binary,
        'modality_str': modality_str,
        'scale': scale,
        'orig_size': (orig_w, orig_h),
        'target_size': (target_w, target_height),
    }


def load_and_inspect(image_path: str) -> dict:
    """
    Membaca citra dari path lokal, mendeteksi modalitas (Warna/B&W)
    dan menormalisasi polaritas latar belakang.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File gambar tidak ditemukan: {image_path}")
        
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Gagal memuat gambar (format tidak didukung atau file korup): {image_path}")
        
    modality_info = detect_and_normalize_modality(img, target_height=640)
    
    return {
        'original': img,
        'resized_bgr': modality_info['resized_bgr'],
        'gray': modality_info['gray_norm'],
        'height': modality_info['target_size'][1],
        'width': modality_info['target_size'][0],
        'orig_width': modality_info['orig_size'][0],
        'orig_height': modality_info['orig_size'][1],
        'channels': 3 if len(img.shape) == 3 else 1,
        'path': image_path,
        'filename': os.path.basename(image_path),
        'is_color': modality_info['is_color'],
        'is_inverted': modality_info['is_inverted'],
        'is_binary': modality_info['is_binary'],
        'modality_str': modality_info['modality_str'],
        'scale': modality_info['scale'],
    }


def clean_background(gray: np.ndarray, is_binary: bool = False, min_contour_area: int = 150) -> np.ndarray:
    """
    Membersihkan latar belakang dan membuang artefak bintik-bintik/noise kecil
    di luar area utama kaki.
    
    Bekerja adaptif untuk:
      - Citra sensor dengan background hitam murni (0)
      - Citra hasil scan kertas dengan sedikit noise tekstur kertas
    """
    if is_binary:
        # Jika siluet biner, threshold langsung di intensitas rendah
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    else:
        # Gunakan Otsu awal untuk memisahkan noise background dari foreground
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    clean_mask = np.zeros_like(gray)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_contour_area:
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)
            
    # Terapkan mask ke gambar grayscale
    cleaned = cv2.bitwise_and(gray, gray, mask=clean_mask)
    return cleaned


def apply_median_filter(gray: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Median filter untuk meredam noise bintik tanpa mengaburkan tepi kontur."""
    return cv2.medianBlur(gray, kernel_size)


def apply_clahe(gray: np.ndarray, is_binary: bool = False,
                clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Krusial untuk memperjelas area midfoot bertekanan rendah.
    Jika citra sudah merupakan siluet biner murni, CLAHE dilewati agar tidak menimbulkan gradien palsu.
    """
    if is_binary:
        return gray
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def preprocess(image_path: str, median_kernel: int = 5,
               clahe_clip: float = 2.0, clahe_tile: tuple = (8, 8),
               min_contour_area: int = 150) -> dict:
    """
    Pipeline preprocessing lengkap:
      Load & Modality Normalization -> Clean Background -> Median Filter -> CLAHE
    """
    # Step 1: Load, Inspeksi, dan Deteksi Modalitas (Warna vs B&W, Latar Terang vs Gelap)
    info = load_and_inspect(image_path)
    
    # Step 2a: Pembersihan noise background adaptif
    cleaned = clean_background(info['gray'], is_binary=info['is_binary'], min_contour_area=min_contour_area)
    
    # Step 2b: Median Filter
    filtered = apply_median_filter(cleaned, median_kernel)
    
    # Step 2c: CLAHE
    enhanced = apply_clahe(filtered, is_binary=info['is_binary'], clip_limit=clahe_clip, tile_grid_size=clahe_tile)
    
    return {
        'info': info,
        'cleaned': cleaned,
        'filtered': filtered,
        'enhanced': enhanced,
    }
