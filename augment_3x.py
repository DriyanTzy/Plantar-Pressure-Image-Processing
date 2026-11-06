"""
=============================================================================
Script: augment_3x.py
Tujuan: Melipatgandakan dataset CURRENT (4.480 citra) menjadi 3x lipat
        yaitu 13.440 citra total.
        Caranya: setiap file yang ada (termasuk aug sebelumnya) digenerate
        2 varian BARU dengan tag unik aug5_ dan aug6_.
=============================================================================
Teknik Augmentasi Baru (tidak bentrok dengan aug1-aug4):
  aug5_rot3cw  : Rotasi +3 derajat (searah jarum jam, kecil)
  aug6_rot3ccw : Rotasi -3 derajat (berlawanan jarum jam, kecil)
=============================================================================
"""

import os
import glob
import cv2
import numpy as np
from pathlib import Path


def get_background_color(img: np.ndarray) -> tuple:
    """Mendeteksi warna background (hitam atau putih) dari tepi gambar."""
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
    return (255, 255, 255) if np.mean(mean_color) > 127 else (0, 0, 0)


def augment_image_new(img: np.ndarray, bg_color: tuple) -> list:
    """
    Menghasilkan 2 varian augmentasi baru:
      - aug5_rot3cw  : Rotasi +3 derajat
      - aug6_rot3ccw : Rotasi -3 derajat
    """
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)

    # Varian 5: Rotasi +3 derajat (Clockwise, kecil)
    M_cw3 = cv2.getRotationMatrix2D(center, 3, 1.0)
    v5_rot3cw = cv2.warpAffine(img, M_cw3, (w, h), borderValue=bg_color)

    # Varian 6: Rotasi -3 derajat (Counter-Clockwise, kecil)
    M_ccw3 = cv2.getRotationMatrix2D(center, -3, 1.0)
    v6_rot3ccw = cv2.warpAffine(img, M_ccw3, (w, h), borderValue=bg_color)

    return [
        ('aug5_rot3cw',  v5_rot3cw),
        ('aug6_rot3ccw', v6_rot3ccw),
    ]


def make_aug_filename(orig_name: str, aug_tag: str) -> str:
    """Membuat nama file augmentasi baru dengan tag unik."""
    stem = Path(orig_name).stem
    ext  = Path(orig_name).suffix
    return f"{stem}_{aug_tag}{ext}"


def run_augmentation_3x(data_dir: str):
    classes = ['H', 'N', 'L']
    total_source    = 0
    total_generated = 0

    print("=" * 65)
    print("   PROGRAM AUGMENTASI DATASET 3X (aug5 & aug6)")
    print("=" * 65)
    print(f"Direktori Target : {data_dir}")
    print(f"Sumber           : SEMUA file .png yang ada (termasuk aug1-4)")
    print(f"Target baru      : 2 varian per file (aug5_rot3cw, aug6_rot3ccw)\n")

    for c in classes:
        folder = os.path.join(data_dir, c)
        if not os.path.isdir(folder):
            print(f"[SKIP] Folder tidak ditemukan: {folder}")
            continue

        # Ambil semua PNG yang BELUM punya tag aug5/aug6
        # (supaya idempotent — aman dijalankan ulang tanpa duplikasi)
        all_files = sorted([
            f for f in glob.glob(os.path.join(folder, '*.png'))
            if '_aug5_' not in os.path.basename(f)
            and '_aug6_' not in os.path.basename(f)
        ])
        count_c = len(all_files)
        total_source += count_c

        print(f"-> Memproses Kelas [{c}]: {count_c} file sumber -> Target baru: {count_c * 2} citra...")

        generated_c = 0
        for i, file_path in enumerate(all_files):
            orig_name = os.path.basename(file_path)
            img = cv2.imread(file_path)
            if img is None:
                print(f"   [WARN] Gagal baca: {orig_name}")
                continue

            bg_color = get_background_color(img)
            aug_list = augment_image_new(img, bg_color)

            for aug_tag, aug_img in aug_list:
                new_fname = make_aug_filename(orig_name, aug_tag)
                new_path  = os.path.join(folder, new_fname)
                cv2.imwrite(new_path, aug_img, [cv2.IMWRITE_PNG_COMPRESSION, 4])
                generated_c  += 1
                total_generated += 1

            if (i + 1) % 100 == 0 or (i + 1) == count_c:
                pct = (i + 1) / count_c * 100
                print(f"   [{i + 1:>4}/{count_c}] {pct:>5.1f}% selesai  ({generated_c} citra baru dibuat)")

        final_count = len(glob.glob(os.path.join(folder, '*.png')))
        print(f"   [OK] Kelas [{c}] sekarang berisi total {final_count} citra.\n")

    print("=" * 65)
    print("   AUGMENTASI 3X SELESAI!")
    print(f"   Total File Sumber      : {total_source}")
    print(f"   Total Citra Baru Dibuat: {total_generated}")
    print(f"   Total Keseluruhan      : {total_source + total_generated} citra")
    print("=" * 65)


if __name__ == '__main__':
    target_data_dir = r'C:\Project\Image Processing\foot_data'
    run_augmentation_3x(target_data_dir)
