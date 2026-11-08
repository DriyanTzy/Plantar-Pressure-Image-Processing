"""
=============================================================================
Script: augment_dataset_3x.py
Tujuan: Melipatgandakan dataset telapak kaki menjadi TEPAT 3x lipat
        (896 base -> 2.688 citra total) dengan variasi morfologis bermakna:
        1. Base Original (1x)
        2. Horizontal Flip (Flip Kanan <-> Kiri / Mirror Anatomi Bilateral)
        3. Vertical Flip (Flip Atas <-> Bawah / Upside Down)
=============================================================================
"""

import os
import glob
import cv2
import numpy as np
from pathlib import Path


def generate_3x_variations(img: np.ndarray) -> list:
    """
    Menghasilkan 2 varian baru dari 1 citra asli:
      1. Horizontal Flip (Kanan <-> Kiri)
      2. Vertical Flip (Atas <-> Bawah / Inversi Gravitasi Pijakan)
    """
    # Varian 1: Flip Horizontal (Kiri <-> Kanan)
    v_fliph = cv2.flip(img, 1)
    
    # Varian 2: Flip Vertikal (Atas <-> Bawah)
    v_flipv = cv2.flip(img, 0)
    
    return [
        ('aug_fliph', v_fliph),
        ('aug_flipv', v_flipv),
    ]


def make_aug_filename(orig_name: str, aug_tag: str) -> str:
    """
    Membuat nama file augmentasi dengan label sisi yang akurat.
    Khusus horizontal flip, jika nama memiliki _L_ maka diubah menjadi _R_
    (dan sebaliknya) karena membalik kaki kiri menghasilkan kaki kanan.
    """
    stem = Path(orig_name).stem
    ext = Path(orig_name).suffix
    
    if aug_tag == 'aug_fliph':
        if '_foot_L_' in stem:
            new_stem = stem.replace('_foot_L_', '_foot_R_')
        elif '_foot_R_' in stem:
            new_stem = stem.replace('_foot_R_', '_foot_L_')
        elif '_L_' in stem:
            new_stem = stem.replace('_L_', '_R_')
        elif '_R_' in stem:
            new_stem = stem.replace('_R_', '_L_')
        else:
            new_stem = stem
        return f"{new_stem}_{aug_tag}{ext}"
    else:
        return f"{stem}_{aug_tag}{ext}"


def run_3x_augmentation(data_dir: str):
    classes = ['H', 'N', 'L']
    total_base = 0
    total_new = 0
    
    print("=" * 65)
    print("   PROGRAM AUGMENTASI DATASET 3X LIPAT (BERVARIASI & NON-OVERFITTING)")
    print("=" * 65)
    print(f"Direktori Target : {data_dir}")
    print("Variasi yang dibuat per citra:")
    print("  1. Flip Horizontal (Kiri <-> Kanan / Mirror Bilateral)")
    print("  2. Flip Vertikal   (Atas <-> Bawah / Upside Down)\n")
    
    for c in classes:
        folder = os.path.join(data_dir, c)
        if not os.path.isdir(folder):
            print(f"[SKIP] Folder tidak ditemukan: {folder}")
            continue
            
        # Ambil hanya citra dasar (bukan hasil aug)
        base_files = sorted([f for f in glob.glob(os.path.join(folder, '*.png')) if '_aug' not in os.path.basename(f)])
        count_c = len(base_files)
        total_base += count_c
        
        print(f"-> Memproses Kelas [{c}]: {count_c} citra asli -> Target: {count_c * 3} citra...")
        
        gen_c = 0
        for i, fpath in enumerate(base_files):
            fname = os.path.basename(fpath)
            img = cv2.imread(fpath)
            if img is None:
                continue
                
            variations = generate_3x_variations(img)
            for aug_tag, aug_img in variations:
                new_fname = make_aug_filename(fname, aug_tag)
                new_path = os.path.join(folder, new_fname)
                cv2.imwrite(new_path, aug_img, [cv2.IMWRITE_PNG_COMPRESSION, 4])
                gen_c += 1
                total_new += 1
                
            if (i + 1) % 50 == 0 or (i + 1) == count_c:
                pct = (i + 1) / count_c * 100
                print(f"   [{i + 1:>3}/{count_c}] {pct:>5.1f}% selesai ({gen_c} citra baru dibuat)")
                
        final_count = len(glob.glob(os.path.join(folder, '*.png')))
        print(f"   [OK] Kelas [{c}] sekarang berisi total {final_count} citra.\n")
        
    print("=" * 65)
    print("   AUGMENTASI 3X BERHASIL!")
    print(f"   Total Citra Asli (Base) : {total_base}")
    print(f"   Total Citra Baru Dibuat : {total_new}")
    print(f"   Total Keseluruhan       : {total_base + total_new} citra (Tepat 3x lipat)")
    print("=" * 65)


if __name__ == '__main__':
    target_dir = r'C:\Project\Image Processing\foot_data'
    run_3x_augmentation(target_dir)
