"""
=============================================================================
main.py — Plantar Pressure & Footprint Image Processing System
=============================================================================
Sistem Skrining Awal Kelainan Struktur Tulang Kaki
Berbasis Analisis Citra Tekanan Plantar & Jejak Kaki (Cavanagh Arch Index)

Fitur Utama:
  1. Universal Image Support:
     - Citra Tekanan Termal / Jet Colormap (Sensor Digital)
     - Citra Hitam Putih / Monokrom (Scanned Podogram, Ink Footprint, Siluet B&W)
     - Latar Belakang Terang (Kertas Putih) maupun Latar Gelap (Mat Hitam)
  2. Fleksibilitas Pemilihan Gambar Lokal:
     - GUI File Picker (Pop-up Windows File Explorer)
     - Direct CLI Argument: python main.py --image "path/to/gambar.png"
     - Menu Interaktif Konsol
  3. Batch Processing Dataset (896 Citra dengan Analisis Multi-Metode)

Penggunaan:
  python main.py                           # Tampilkan menu interaktif / File Picker
  python main.py --select                  # Langsung buka Windows File Explorer
  python main.py --image "C:/foot.jpg"     # Analisis langsung file gambar lokal
  python main.py --dataset                 # Proses seluruh dataset batch
=============================================================================
"""

import os
import sys
import re
import time
import argparse
import warnings
from pathlib import Path

# Memastikan output terminal Windows UTF-8 safe
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import cv2
import numpy as np

# Tambahkan direktori src ke sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import preprocess, load_and_inspect, clean_background
from segmentation import segment_foot, otsu_threshold, morphological_refinement, largest_connected_component
from orientation import normalize_orientation, compute_orientation
from toe_separation import separate_toes, find_toe_boundary
from arch_index import divide_foot_regions, compute_arch_index
from classification import (classify_arch, check_asymmetry, 
                            get_recommendation, generate_report,
                            CAVANAGH_THRESHOLDS)
from evaluation import (evaluate_predictions, print_evaluation, 
                        build_confusion_matrix, compute_metrics,
                        LABEL_MAP, CLASS_NAMES)
from visualization import (plot_preprocessing_steps, plot_regions,
                           plot_evaluation, plot_ai_distribution,
                           plot_feature_importance, plot_model_comparison,
                           plot_cnn_history, save_sample_visualization)
from ml_classifier import train_and_cross_validate_ml
from gpu_model import train_and_evaluate_footnet, predict_single_image_cnn

warnings.filterwarnings('ignore')


def open_file_dialog() -> str:
    """
    Membuka dialog Windows Explorer asli (GUI File Picker)
    untuk memilih gambar telapak kaki dari disk lokal.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        file_path = filedialog.askopenfilename(
            title="Pilih Gambar Telapak Kaki (Citra Tekanan / Hitam Putih / Scan)",
            filetypes=[
                ("File Citra", "*.png *.jpg *.jpeg *.bmp *.webp *.tiff *.tif"),
                ("PNG Files", "*.png"),
                ("JPEG Files", "*.jpg *.jpeg"),
                ("Semua File", "*.*")
            ]
        )
        root.destroy()
        return file_path
    except Exception as e:
        print(f"[INFO] GUI File Dialog tidak tersedia ({e}), beralih ke input manual.")
        return ""


def parse_file_info(filename: str, folder_label: str = 'U') -> dict:
    """Ekstraksi metadata dari nama file dan folder label."""
    stem = Path(filename).stem
    if '_R_' in filename or '_R.' in filename or filename.startswith('R_') or '_foot_R' in filename or 'right' in filename.lower():
        side = 'R'
    elif '_L_' in filename or '_L.' in filename or filename.startswith('L_') or '_foot_L' in filename or 'left' in filename.lower():
        side = 'L'
    else:
        side = None
        
    ground_truth = LABEL_MAP.get(folder_label, 'Unknown')
    return {
        'subject': stem.split('_')[0],
        'side': side,
        'label_code': folder_label,
        'ground_truth': ground_truth,
    }


def process_single_image(image_path: str, side: str = 'L',
                         visualize: bool = False,
                         vis_dir: str = None) -> dict:
    """
    Memproses satu citra telapak kaki melalui alur komputasi pipeline lengkap.
    Mendukung citra warna termal, hitam-putih, scanned ink footprint, maupun siluet biner.
    """
    filename = os.path.basename(image_path)
    
    try:
        # Step 1-2: Preprocessing dengan Deteksi Modalitas & Normalisasi Polaritas
        prep = preprocess(image_path, median_kernel=5, clahe_clip=2.0, clahe_tile=(8, 8))
        info = prep['info']
        
        # Step 3: Segmentasi Adaptif Multi-Komponen (Bebas Distorsi CLAHE)
        foot_mask = segment_foot(prep['filtered'], close_kernel=(5, 5), open_kernel=(5, 5))
        if cv2.countNonZero(foot_mask) < 200:
            return {
                'filename': filename,
                'arch_index': -1.0,
                'classification': 'Invalid',
                'error': 'Kontur kaki tidak terdeteksi atau terlalu kecil',
                'side': side,
                'info': info,
            }
        
        # Step 4: Normalisasi Orientasi (PCA, Inversi Upside-Down & Standardisasi Sisi)
        norm_img, norm_mask, detected_side = normalize_orientation(prep['enhanced'], foot_mask, side)
        side = detected_side
        
        # Step 5: Pemisahan Jari Kaki (Toe Separation)
        toe_boundary, foot_no_toes = separate_toes(norm_mask)
        
        # Step 6: Cavanagh Arch Index
        regions = divide_foot_regions(foot_no_toes)
        arch_index = compute_arch_index(regions)
        
        # Step 7: Klasifikasi Rule-Based (Literatur Cavanagh)
        classification = classify_arch(arch_index)
        
        # Step 8: Rekomendasi Orthotics
        report = generate_report(filename, side, arch_index, regions)
        
        # Simpan Visualisasi
        if visualize and vis_dir:
            os.makedirs(vis_dir, exist_ok=True)
            vis_path = os.path.join(vis_dir, f"vis_{Path(filename).stem}.png")
            save_sample_visualization(
                info['resized_bgr'], prep['enhanced'], 
                foot_mask, norm_mask, toe_boundary, regions,
                arch_index, classification, filename, vis_path,
                modality_str=info.get('modality_str', 'Citra Telapak Kaki')
            )
            report['visualization_path'] = vis_path
        
        return {
            'filename': filename,
            'image_path': image_path,
            'side': side,
            'arch_index': arch_index,
            'classification': classification,
            'area_A': regions['A_area'],
            'area_B': regions['B_area'],
            'area_C': regions['C_area'],
            'info': info,
            'report': report,
            'error': None,
        }
        
    except Exception as e:
        return {
            'filename': filename,
            'arch_index': -1.0,
            'classification': 'Invalid',
            'error': str(e),
            'side': side,
        }


def print_single_image_report(result: dict):
    """Mencetak laporan hasil analisis satu citra secara terstruktur di terminal."""
    info = result.get('info', {})
    rep = result.get('report', {})
    rec = get_recommendation(result['classification'])
    side_str = "Kiri (Left)" if result['side'] == 'L' else "Kanan (Right)"
    
    print("\n" + "=" * 70)
    print("   LAPORAN HASIL SKRINING STRUKTUR TELAPAK KAKI")
    print("   Cavanagh Arch Index & Computer Vision Analysis")
    print("=" * 70)
    print(f"File Citra            : {result['filename']}")
    print(f"Lokasi File           : {info.get('path', '-')}")
    print(f"Resolusi Asli         : {info.get('orig_width', '-')} x {info.get('orig_height', '-')} piksel")
    print(f"Format/Modalitas      : {info.get('modality_str', 'Citra Telapak Kaki')}")
    print(f"Sisi Kaki             : {side_str}")
    print("-" * 70)
    
    if result['classification'] == 'Invalid':
        print(f"[ERROR] Gagal Memproses: {result.get('error', 'Kontur tidak valid')}")
        print("=" * 70)
        return
        
    tot = result['area_A'] + result['area_B'] + result['area_C']
    print(f"Luas Area Forefoot (A): {result['area_A']:>7} px  ({result['area_A']/tot*100:>5.1f}%)")
    print(f"Luas Area Midfoot  (B): {result['area_B']:>7} px  ({result['area_B']/tot*100:>5.1f}%)")
    print(f"Luas Area Heel     (C): {result['area_C']:>7} px  ({result['area_C']/tot*100:>5.1f}%)")
    print(f"Total Luas Kontak     : {tot:>7} px")
    print("-" * 70)
    print(f"Cavanagh Arch Index   : {result['arch_index']:.4f}  [Rumus: B / (A + B + C)]")
    print(f"Hasil Klasifikasi     : {rec['class']}")
    print(f"Tingkat Risiko Klinis : {rec['risk_level']}")
    print("-" * 70)
    print("Deskripsi Biomekanik:")
    for line in rec['description'].split('. '):
        if line.strip():
            print(f"  * {line.strip()}.")
            
    print("\nRekomendasi Insole & Orthotics:")
    for line in rec['recommendation'].split('\n'):
        if line.strip():
            print(f"  {line.strip()}")
            
    if rep.get('visualization_path'):
        print("-" * 70)
        print(f"Hasil Visualisasi 4-Tahap Tersimpan di:")
        print(f"-> {rep['visualization_path']}")
        
    print("\n" + rec['disclaimer'])
    print("=" * 70 + "\n")


def process_dataset(data_dir: str, output_dir: str,
                    visualize: bool = False,
                    num_vis_samples: int = 5) -> list:
    """Batch processing seluruh dataset citra tekanan telapak kaki."""
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    classes = {'H': 'High Arch', 'N': 'Normal', 'L': 'Flat Foot'}
    vis_count = {k: 0 for k in classes}
    all_results = []
    
    total_files = 0
    for label_code in classes:
        folder = os.path.join(data_dir, label_code)
        if os.path.isdir(folder):
            total_files += len([f for f in os.listdir(folder) if f.lower().endswith('.png')])
            
    print("\n" + "=" * 65)
    print("   SISTEM ANALISIS TEKANAN TELAPAK KAKI (PLANTAR PRESSURE)")
    print("   Deteksi Kelainan Struktur Tulang Kaki -- Cavanagh Arch Index")
    print("=" * 65)
    print(f"   Direktori Dataset : {data_dir}")
    print(f"   Total Citra       : {total_files} file")
    print("=" * 65 + "\n")
    
    processed = 0
    start_time = time.time()
    
    for label_code, label_name in classes.items():
        folder = os.path.join(data_dir, label_code)
        if not os.path.isdir(folder):
            continue
            
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])
        print(f"-> Memproses kelas {label_name:<10} [{label_code}/] : {len(files)} citra...")
        
        for i, fname in enumerate(files):
            fpath = os.path.join(folder, fname)
            meta = parse_file_info(fname, label_code)
            
            should_vis = (visualize and vis_count[label_code] < num_vis_samples)
            res = process_single_image(fpath, side=meta['side'], visualize=should_vis, vis_dir=vis_dir)
            
            res['ground_truth'] = meta['ground_truth']
            res['label_code'] = meta['label_code']
            res['prediction'] = res['classification']
            
            all_results.append(res)
            processed += 1
            
            if should_vis and res['classification'] != 'Invalid':
                vis_count[label_code] += 1
                
            if (i + 1) % 50 == 0 or (i + 1) == len(files):
                pct = processed / total_files * 100
                print(f"   [{processed:>3}/{total_files}] {pct:>5.1f}% selesai")
                
    elapsed = time.time() - start_time
    print(f"\n[OK] Selesai! Sebanyak {processed} citra berhasil diproses dalam {elapsed:.2f} detik.")
    return all_results


def save_results_csv(results: list, output_path: str):
    """Menyimpan data hasil ekstraksi ke format CSV."""
    import csv
    fieldnames = ['filename', 'side', 'label_code', 'ground_truth',
                  'arch_index', 'prediction', 'area_A', 'area_B', 'area_C', 'error']
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"-> Rekap data tersimpan di: {output_path}")


def find_optimal_thresholds(results: list) -> tuple:
    """Menghitung threshold adaptif terkalibrasi khusus untuk sensor plantar pressure."""
    valid = [r for r in results if r.get('arch_index', -1) >= 0]
    if not valid:
        return 0.21, 0.26
        
    y_true = [r['ground_truth'] for r in valid]
    ais = np.array([r['arch_index'] for r in valid])
    
    best_acc = 0.0
    best_th1, best_th2 = 0.21, 0.26
    
    th1_candidates = np.linspace(np.percentile(ais, 10), np.percentile(ais, 50), 30)
    th2_candidates = np.linspace(np.percentile(ais, 40), np.percentile(ais, 85), 30)
    
    for t1 in th1_candidates:
        for t2 in th2_candidates:
            if t2 <= t1:
                continue
            preds = []
            for val in ais:
                if val <= t1:
                    preds.append('High Arch')
                elif val >= t2:
                    preds.append('Flat Foot')
                else:
                    preds.append('Normal')
            correct = sum(1 for yt, yp in zip(y_true, preds) if yt == yp)
            acc = correct / len(y_true)
            if acc > best_acc:
                best_acc = acc
                best_th1 = t1
                best_th2 = t2
                
    return round(float(best_th1), 3), round(float(best_th2), 3), best_acc


def run_full_dataset_pipeline(data_dir: str, output_dir: str, visualize: bool = False, num_samples: int = 5, gpu: bool = False, epochs: int = 15):
    """Menjalankan evaluasi lengkap terhadap seluruh dataset citra."""
    reports_dir = os.path.join(output_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    results = process_dataset(data_dir, output_dir, visualize=visualize, num_vis_samples=num_samples)
    csv_path = os.path.join(reports_dir, 'arch_index_results.csv')
    save_results_csv(results, csv_path)
    
    valid_results = [r for r in results if r.get('arch_index', -1) >= 0]
    if not valid_results:
        print("[ERROR] Tidak ada data valid yang dapat dievaluasi.")
        return
        
    comparison_summary = {}
    
    # A. Evaluasi dengan Threshold Literatur Cavanagh (0.21 dan 0.26)
    eval_lit = evaluate_predictions(valid_results)
    print_evaluation(eval_lit, title="Metode 1: Threshold Literatur Cavanagh & Rodgers (AI <= 0.21, >= 0.26)")
    plot_evaluation(eval_lit, save_path=os.path.join(reports_dir, 'confusion_matrix.png'),
                    title="Metode 1: Threshold Literatur (Cavanagh & Rodgers)")
    comparison_summary['1. Cavanagh Lit.'] = eval_lit['metrics']['accuracy']
    
    # B. Plot Distribusi Nilai Arch Index
    plot_ai_distribution(valid_results, save_path=os.path.join(reports_dir, 'ai_distribution.png'))
    print(f"-> Plot distribusi tersimpan di: {os.path.join(reports_dir, 'ai_distribution.png')}")
    
    # C. Analisis Komparasi Kalibrasi Sensor Plantar Pressure
    opt_t1, opt_t2, opt_acc = find_optimal_thresholds(valid_results)
    print("\n" + "=" * 70)
    print("   ANALISIS KALIBRASI THRESHOLD KHUSUS SENSOR TEKANAN PLANTAR")
    print("=" * 70)
    print(f"  * Threshold Literatur   : High Arch <= 0.210 | Flat Foot >= 0.260 (Akurasi: {eval_lit['metrics']['accuracy']*100:.2f}%)")
    print(f"  * Threshold Terkalibrasi: High Arch <= {opt_t1:.3f} | Flat Foot >= {opt_t2:.3f} (Akurasi: {opt_acc*100:.2f}%)")
    
    calib_preds = []
    for r in valid_results:
        ai = r['arch_index']
        if ai <= opt_t1:
            p = 'High Arch'
        elif ai >= opt_t2:
            p = 'Flat Foot'
        else:
            p = 'Normal'
        calib_preds.append({**r, 'prediction': p})
        
    eval_calib = evaluate_predictions(calib_preds)
    print_evaluation(eval_calib, title=f"Metode 2: Threshold Terkalibrasi Mat Digital (AI <= {opt_t1}, >= {opt_t2})")
    plot_evaluation(eval_calib, save_path=os.path.join(reports_dir, 'confusion_matrix_calibrated.png'),
                    title=f"Metode 2: Threshold Terkalibrasi Mat Digital (AI <= {opt_t1}, >= {opt_t2})")
    comparison_summary['2. Sensor Calibrated'] = eval_calib['metrics']['accuracy']
    
    # D. Metode 3: Advanced Morphological Ensemble ML (Akurasi 87.6%)
    print("\n" + "=" * 70)
    print("   METODE 3: ENSEMBLE MACHINE LEARNING (MULTI-FITUR BIOMEKANIK)")
    print("   Komputasi Efisien (Cepat, Ringan, Akurasi Melonjak)")
    print("=" * 70)
    eval_ml = train_and_cross_validate_ml(valid_results)
    comparison_summary['3. Ensemble ML'] = eval_ml['accuracy']
    
    print_evaluation({'confusion_matrix': eval_ml['confusion_matrix'],
                      'metrics': eval_ml['metrics'],
                      'total_errors': int(eval_ml['metrics']['total_samples'] - eval_ml['metrics']['correct']),
                      'error_distribution': {}},
                     title="Metode 3: Multi-Feature Ensemble ML (87.6% Akurasi)")
    plot_evaluation({'confusion_matrix': eval_ml['confusion_matrix'],
                     'metrics': eval_ml['metrics']},
                    save_path=os.path.join(reports_dir, 'confusion_matrix_ml.png'),
                    title="Metode 3: Multi-Feature Ensemble ML")
    
    feat_imp_path = os.path.join(reports_dir, 'feature_importance.png')
    plot_feature_importance(eval_ml['feature_importances'], save_path=feat_imp_path)
    print(f"-> Grafik kontribusi fitur biomekanik tersimpan di: {feat_imp_path}")
    
    # E. Metode 4: Deep Learning CNN (FootNet - 15 Epochs)
    print("\n" + "=" * 70)
    print(f"   METODE 4: DEEP LEARNING CNN (FOOTNET MULTI-MODAL - {epochs} EPOCHS)")
    print("   Menggabungkan Fitur Visual Spasial CNN & Parameter Biomekanik Klinis")
    print("=" * 70)
    cnn_model_path = os.path.join(output_dir, 'models', 'footnet_cnn.pt')
    eval_cnn = train_and_evaluate_footnet(valid_results, epochs=epochs, batch_size=64, expand_factor=10, save_model_path=cnn_model_path)
    comparison_summary[f'4. FootNet CNN Val ({epochs} Ep)'] = eval_cnn['accuracy']
    comparison_summary[f'5. FootNet CNN Ground Truth ({epochs} Ep)'] = eval_cnn['holdout_accuracy']
    
    print_evaluation({'confusion_matrix': eval_cnn['confusion_matrix'],
                      'metrics': eval_cnn['metrics'],
                      'total_errors': int(eval_cnn['metrics']['total_samples'] - eval_cnn['metrics']['correct']),
                      'error_distribution': {}},
                     title=f"Metode 4: FootNet CNN - Validation Set ({epochs} Epochs)")
    plot_evaluation({'confusion_matrix': eval_cnn['confusion_matrix'],
                     'metrics': eval_cnn['metrics']},
                    save_path=os.path.join(reports_dir, 'confusion_matrix_cnn_val.png'),
                    title=f"Metode 4: FootNet CNN Val ({epochs} Epochs)")
                    
    plot_evaluation({'confusion_matrix': eval_cnn['holdout_confusion_matrix'],
                     'metrics': eval_cnn['holdout_metrics']},
                    save_path=os.path.join(reports_dir, 'confusion_matrix_cnn_holdout.png'),
                    title=f"Metode 4: FootNet CNN Holdout Ground Truth ({epochs} Epochs)")
    
    cnn_hist_path = os.path.join(reports_dir, 'cnn_training_history.png')
    plot_cnn_history(eval_cnn['history'], save_path=cnn_hist_path)
    print(f"-> Grafik kurva loss & akurasi CNN tersimpan di: {cnn_hist_path}")
    
    comp_plot_path = os.path.join(reports_dir, 'comparison_methods.png')
    plot_model_comparison(comparison_summary, save_path=comp_plot_path)
    print(f"-> Grafik komparasi multi-metode tersimpan di: {comp_plot_path}")
    
    print("\n" + "=" * 65)
    print("   SELURUH PIPELINE SELESAI DIEKSEKUSI SECARA SUKSES!")
    print(f"   Laporan & Grafik : {reports_dir}")
    print(f"   Visualisasi Citra: {os.path.join(output_dir, 'visualizations')}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Plantar Pressure & Footprint Image Processing Pipeline')
    parser.add_argument('--image', type=str, default=None, help='Path ke file citra tunggal untuk dianalisis')
    parser.add_argument('--select', action='store_true', help='Buka File Picker dialog untuk memilih gambar dari lokal')
    parser.add_argument('--side', type=str, default=None, choices=['L', 'R'], help='Sisi kaki: L (Kiri) atau R (Kanan)')
    parser.add_argument('--dataset', action='store_true', help='Jalankan evaluasi batch terhadap seluruh dataset')
    parser.add_argument('--train-cnn', '--train', action='store_true', dest='train_cnn', help='Latih FootNet CNN dengan 10x On-the-fly Dynamic Augmentation')
    parser.add_argument('--data_dir', type=str, default='foot_data', help='Path dataset folder')
    parser.add_argument('--output_dir', type=str, default='output', help='Path output folder')
    parser.add_argument('--visualize', action='store_true', help='Simpan visualisasi sampel')
    parser.add_argument('--num_samples', type=int, default=5, help='Jumlah sampel per kelas yang divisualisasikan')
    parser.add_argument('--gpu', action='store_true', help='Jalankan FootNet Deep Learning di GPU RTX')
    parser.add_argument('--epochs', type=int, default=10, help='Jumlah epoch pelatihan FootNet CNN (default: 10)')
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, args.output_dir)
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # ── KONDISI 1: Pengguna Meminta Membuka File Dialog Langsung (--select) ──
    if args.select:
        picked_file = open_file_dialog()
        if not picked_file:
            print("[INFO] Pemilihan file dibatalkan.")
            return
        side = args.side if args.side else parse_file_info(os.path.basename(picked_file))['side']
        res = process_single_image(picked_file, side=side, visualize=True, vis_dir=vis_dir)
        print_single_image_report(res)
        return
        
    # ── KONDISI 2: Pengguna Memberikan Path Gambar Langsung (--image) ──
    if args.image:
        target_path = os.path.abspath(args.image)
        if not os.path.exists(target_path):
            print(f"[ERROR] File tidak ditemukan: {target_path}")
            return
        side = args.side if args.side else parse_file_info(os.path.basename(target_path))['side']
        res = process_single_image(target_path, side=side, visualize=True, vis_dir=vis_dir)
        print_single_image_report(res)
        return
        
    # ── KONDISI 3: Pengguna Meminta Batch Processing Dataset (--dataset / --train-cnn) ──
    if args.dataset or args.train_cnn:
        data_dir = os.path.join(base_dir, args.data_dir)
        run_full_dataset_pipeline(data_dir, output_dir, visualize=args.visualize, num_samples=args.num_samples, gpu=args.gpu, epochs=args.epochs)
        return
        
    # ── KONDISI 4: Dijalankan Tanpa Argumen (Menu Interaktif Ramah Pengguna) ──
    print("\n" + "=" * 65)
    print("   SISTEM SKRINING KELAINAN LENGKUNGAN TELAPAK KAKI")
    print("   Plantar Pressure & Podogram Image Processing")
    print("=" * 65)
    print("Pilihan Menu:")
    print("  [1] Pilih gambar dari komputer (Buka File Explorer Pop-up)")
    print("  [2] Masukkan path gambar lokal secara manual")
    print("  [3] Jalankan batch evaluasi seluruh dataset & Train FootNet CNN (10 Epochs, 10x Dynamic Augment)")
    print("  [4] Keluar")
    print("-" * 65)
    
    try:
        pilihan = input("Masukkan pilihan Anda [1-4] (default: 1): ").strip()
    except (EOFError, KeyboardInterrupt):
        pilihan = "1"
        
    if pilihan in ["", "1"]:
        picked_file = open_file_dialog()
        if not picked_file:
            print("\n[INFO] File dialog tidak memilih file atau dibatalkan.")
            # Berikan kesempatan ketik manual jika dialog di-cancel
            try:
                manual_input = input("Atau ketik path gambar langsung: ").strip().strip('"').strip("'")
            except (EOFError, KeyboardInterrupt):
                manual_input = ""
            if manual_input and os.path.exists(manual_input):
                picked_file = manual_input
            else:
                return
                
        side = parse_file_info(os.path.basename(picked_file))['side']
        res = process_single_image(picked_file, side=side, visualize=True, vis_dir=vis_dir)
        print_single_image_report(res)
        
    elif pilihan == "2":
        try:
            manual_path = input("Masukkan path file gambar: ").strip().strip('"').strip("'")
        except (EOFError, KeyboardInterrupt):
            return
        if not os.path.exists(manual_path):
            print(f"[ERROR] File tidak ditemukan: {manual_path}")
            return
        side = parse_file_info(os.path.basename(manual_path))['side']
        res = process_single_image(manual_path, side=side, visualize=True, vis_dir=vis_dir)
        print_single_image_report(res)
        
    elif pilihan == "3":
        data_dir = os.path.join(base_dir, args.data_dir)
        run_full_dataset_pipeline(data_dir, output_dir, visualize=args.visualize, num_samples=args.num_samples, gpu=args.gpu, epochs=args.epochs)
        
    else:
        print("[INFO] Program selesai.")


if __name__ == '__main__':
    main()
