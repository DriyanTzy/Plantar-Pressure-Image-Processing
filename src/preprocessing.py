"""
=============================================================================
Module: preprocessing.py
Pipeline Steps 1 & 2 — Akuisisi, Inspeksi Data, dan Preprocessing
=============================================================================
Fungsi utama:
  - load_and_inspect() : Membaca gambar, mengecek resolusi, menampilkan info.
  - clean_background() : Membersihkan artefak/noise kecil di luar area kaki.
  - preprocess()       : Konversi grayscale → Median Filter → CLAHE.
=============================================================================
"""

import cv2
import numpy as np


def load_and_inspect(image_path: str) -> dict:
    """
    Membaca gambar dari path, melakukan inspeksi awal.
    
    Returns:
        dict dengan keys:
            'original'  : gambar BGR asli
            'gray'      : gambar grayscale
            'height'    : tinggi gambar
            'width'     : lebar gambar
            'channels'  : jumlah channel
            'path'      : path file
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")
    
    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) == 3 else 1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return {
        'original': img,
        'gray': gray,
        'height': h,
        'width': w,
        'channels': c,
        'path': image_path
    }


def clean_background(gray: np.ndarray, min_contour_area: int = 500) -> np.ndarray:
    """
    Membersihkan latar belakang dan noise bintik-bintik kecil.
    
    Karena background sudah hitam murni (0):
    1. Threshold awal untuk mengisolasi piksel non-hitam (foreground).
    2. Filter kontur minimum untuk membuang artefak kecil (misal titik merah
       dari marker yang tersisa setelah konversi grayscale).
    3. Masking ulang terhadap gambar grayscale asli.
    
    Args:
        gray: Gambar grayscale input.
        min_contour_area: Luas minimum kontur yang dianggap valid (bukan noise).
        
    Returns:
        Gambar grayscale yang sudah dibersihkan dari noise kecil.
    """
    # Threshold awal: piksel > 5 dianggap foreground
    _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    
    # Cari semua kontur
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Buat mask hanya dari kontur yang cukup besar
    clean_mask = np.zeros_like(gray)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_contour_area:
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)
    
    # Terapkan mask ke gambar grayscale
    cleaned = cv2.bitwise_and(gray, gray, mask=clean_mask)
    
    return cleaned


def apply_median_filter(gray: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Median filter untuk meredam noise bintik pada citra tekanan
    tanpa mengaburkan batas tepi (edges).
    
    Args:
        gray: Gambar grayscale.
        kernel_size: Ukuran kernel median filter (harus ganjil).
        
    Returns:
        Gambar yang sudah difilter.
    """
    return cv2.medianBlur(gray, kernel_size)


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, 
                tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Krusial untuk memperjelas area midfoot yang sering memiliki 
    tekanan rendah (warna biru gelap/hitam pada citra asli) agar 
    tidak hilang saat disegmentasi.
    
    Args:
        gray: Gambar grayscale.
        clip_limit: Batas clipping untuk kontras.
        tile_grid_size: Ukuran tile grid.
        
    Returns:
        Gambar yang sudah di-enhance kontrasnya.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def preprocess(image_path: str, median_kernel: int = 5,
               clahe_clip: float = 2.0, clahe_tile: tuple = (8, 8),
               min_contour_area: int = 500) -> dict:
    """
    Pipeline preprocessing lengkap: Load → Clean → Median → CLAHE.
    
    Args:
        image_path: Path ke file gambar.
        median_kernel: Ukuran kernel median filter.
        clahe_clip: CLAHE clip limit.
        clahe_tile: CLAHE tile grid size.
        min_contour_area: Luas minimum kontur valid.
        
    Returns:
        dict berisi semua tahap intermediate:
            'info'      : dict info gambar dari load_and_inspect
            'cleaned'   : gambar setelah pembersihan background
            'filtered'  : gambar setelah median filter  
            'enhanced'  : gambar setelah CLAHE (hasil akhir preprocessing)
    """
    # Step 1: Load & Inspect
    info = load_and_inspect(image_path)
    
    # Step 2a: Clean background noise
    cleaned = clean_background(info['gray'], min_contour_area)
    
    # Step 2b: Median Filter
    filtered = apply_median_filter(cleaned, median_kernel)
    
    # Step 2c: CLAHE
    enhanced = apply_clahe(filtered, clahe_clip, clahe_tile)
    
    return {
        'info': info,
        'cleaned': cleaned,
        'filtered': filtered,
        'enhanced': enhanced
    }
