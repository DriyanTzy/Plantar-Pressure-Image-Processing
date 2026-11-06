"""
=============================================================================
Script: augment_dataset.py
Tujuan: Melipatgandakan dataset telapak kaki 5x lipat (896 -> 4.480 citra)
        untuk kebutuhan pelatihan Deep Learning (CNN/PyTorch).
=============================================================================
Teknik Augmentasi per Citra:
  1. Horizontal Flip (Mirror L <-> R) -> Anatomi kaki kiri <-> kanan valid
  2. Rotasi Rot-CW (+6 derajat)       -> Variasi sudut pijakan kaki
  3. Rotasi Rot-CCW (-6 derajat)      -> Variasi sudut pijakan kaki
  4. Zoom / Scale (1.04x) + Shift     -> Variasi proporsi ukuran & posisi
=============================================================================
"""

import os
import glob
import cv2
import numpy as np
from pathlib import Path


def get_background_color(img: np.ndarray) -> tuple:
    """Mendeteksi apakah background citra hitam atau putih berdasarkan batas tepi."""
    h, w = img.shape[:2]
    pad_h = max(2, int(h * 0.04))
    pad_w = max(2, int(w * 0.04))
    
    border_pixels = np.concatenate([
        img[:pad_h, :, :].reshape(-1, 3),
        img[-pad_h:, :, :].reshape(-1, 3),
        img[:, :pad_w, :].reshape(-1, 3),
        img[:, -pad_w:, :].reshape(-1, 3),
    ])
    mean_color = np.mean(border_pixels, axis=0)
    if np.mean(mean_color) > 127:
        return (255, 255, 255)
    else:
        return (0, 0, 0)


def augment_image(img: np.ndarray, bg_color: tuple) -> list:
    """
    Menghasilkan 4 varian augmentasi berkualitas tinggi
    yang secara fisik realistis untuk citra telapak kaki.
    """
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    
    # Varian 1: Horizontal Flip (Mirror Kiri <-> Kanan)
    v1_fliph = cv2.flip(img, 1)
    
    # Varian 2: Rotasi +6 derajat (Clockwise)
    M_cw = cv2.getRotationMatrix2D(center, 6, 1.0)
    v2_rot_cw = cv2.warpAffine(img, M_cw, (w, h), borderValue=bg_color)
    
    # Varian 3: Rotasi -6 derajat (Counter-Clockwise)
    M_ccw = cv2.getRotationMatrix2D(center, -6, 1.0)
    v3_rot_ccw = cv2.warpAffine(img, M_ccw, (w, h), borderValue=bg_color)
    
    # Varian 4: Scale 1.04x + sedikit translasi vertikal (+4 px)
    M_zoom = cv2.getRotationMatrix2D(center, 0, 1.04)
    M_zoom[1, 2] += 4.0
    v4_zoom = cv2.warpAffine(img, M_zoom, (w, h), borderValue=bg_color)
    
    return [
        ('aug1_fliph', v1_fliph),
        ('aug2_rotcw', v2_rot_cw),
        ('aug3_rotccw', v3_rot_ccw),
        ('aug4_zoom', v4_zoom),
    ]


def make_aug_filename(orig_name: str, aug_tag: str) -> str:
    """
    Membuat nama file augmentasi.
    Khusus horizontal flip, tukar tag _L_ menjadi _R_ (dan sebaliknya)
    agar metadata sisi kaki tetap konsisten.
    """
    stem = Path(orig_name).stem
    ext = Path(orig_name).suffix
    
    if aug_tag == 'aug1_fliph':
        if '_L_' in stem:
            new_stem = stem.replace('_L_', '_R_')
        elif '_R_' in stem:
            new_stem = stem.replace('_R_', '_L_')
        else:
            new_stem = stem
        return f"{new_stem}_{aug_tag}{ext}"
    else:
        return f"{stem}_{aug_tag}{ext}"


def run_augmentation(data_dir: str):
    classes = ['H', 'N', 'L']
    total_original = 0
    total_generated = 0
    
    print("=" * 65)
    print("   PROGRAM AUGMENTASI DATASET TELAPAK KAKI (5X MULTIPLIER)")
    print("=" * 65)
    print(f"Direktori Target: {data_dir}\n")
    
    for c in classes:
        folder = os.path.join(data_dir, c)
        if not os.path.isdir(folder):
            print(f"[SKIP] Folder tidak ditemukan: {folder}")
            continue
            
        # Ambil hanya file asli (bukan hasil augmentasi sebelumnya jika ada)
        all_files = [f for f in glob.glob(os.path.join(folder, '*.png')) if '_aug' not in os.path.basename(f)]
        all_files.sort()
        count_c = len(all_files)
        total_original += count_c
        
        print(f"-> Memproses Kelas [{c}]: {count_c} citra asli -> Target: {count_c * 5} citra...")
        
        generated_c = 0
        for i, file_path in enumerate(all_files):
            orig_name = os.path.basename(file_path)
            img = cv2.imread(file_path)
            if img is None:
                continue
                
            bg_color = get_background_color(img)
            aug_list = augment_image(img, bg_color)
            
            for aug_tag, aug_img in aug_list:
                new_fname = make_aug_filename(orig_name, aug_tag)
                new_path = os.path.join(folder, new_fname)
                cv2.imwrite(new_path, aug_img, [cv2.IMWRITE_PNG_COMPRESSION, 4])
                generated_c += 1
                total_generated += 1
                
            if (i + 1) % 50 == 0 or (i + 1) == count_c:
                print(f"   [{i + 1:>3}/{count_c}] {(i + 1)/count_c*100:>5.1f}% selesai ({generated_c} citra baru dibuat)")
                
        final_count = len(glob.glob(os.path.join(folder, '*.png')))
        print(f"   [OK] Kelas [{c}] sekarang berisi total {final_count} citra.\n")
        
    print("=" * 65)
    print("   AUGMENTASI DATASET SELESAI!")
    print(f"   Total Citra Asli       : {total_original}")
    print(f"   Total Citra Baru Dibuat: {total_generated}")
    print(f"   Total Keseluruhan      : {total_original + total_generated} citra (Tepat 5x lipat)")
    print("=" * 65)


if __name__ == '__main__':
    target_data_dir = r'C:\Project\Image Processing\foot_data'
    run_augmentation(target_data_dir)
