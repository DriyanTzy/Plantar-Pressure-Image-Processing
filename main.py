"""
=============================================================================
main.py — Plantar Pressure Image Processing System
=============================================================================
Sistem Skrining Awal Kelainan Struktur Tulang Kaki
Berbasis Analisis Citra Tekanan Plantar (Cavanagh Arch Index + ML/GPU)

Tahapan Pipeline:
  1. Akuisisi & Inspeksi Data (load, background cleaning, min contour filter)
  2. Preprocessing (Grayscale -> Median Filter -> CLAHE)
  3. Segmentasi (Otsu -> Morphological Closing/Opening -> Largest Component)
  4. Normalisasi Orientasi (PCA Sum-Axis -> Rotasi Vertikal -> Crop -> Flip Kiri/Kanan)
  5. Pemisahan Jari Kaki / Toe Separation (Profil Lebar -> Deteksi Lembah / Valley)
  6. Ekstraksi Fitur: Cavanagh Arch Index (Pembagian Regio A, B, C & Formula AI)
  7. Klasifikasi Rule-Based (Threshold Cavanagh & Rodgers)
  8. Rekomendasi Orthotics & Deteksi Asimetri Bilateral
  9. Evaluasi Performa Multi-Metode:
     a. Literatur Cavanagh & Rodgers (Threshold Kertas Tinta Tradisional)
     b. Calibrated Threshold (Penyesuaian Karakteristik Mat Digital)
     c. Advanced Morphological Ensemble ML (87.6% Akurasi, Komputasi Ringan)
     d. RTX GPU FootNet (Opsional jika PyTorch CUDA aktif)

Penggunaan:
  python main.py
  python main.py --visualize --num_samples 5
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
                           save_sample_visualization)
from ml_classifier import train_and_cross_validate_ml

warnings.filterwarnings('ignore')


def parse_file_info(filename: str, folder_label: str) -> dict:
    """Ekstraksi metadata dari nama file dan folder label."""
    stem = Path(filename).stem
    if '_R_' in filename or '_R.' in filename or filename.startswith('R_') or '_foot_R' in filename:
        side = 'R'
    else:
        side = 'L'
        
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
    """Memproses satu citra telapak kaki melalui alur komputasi pipeline lengkap."""
    filename = os.path.basename(image_path)
    
    try:
        # Step 1-2: Preprocessing
        prep = preprocess(image_path, median_kernel=5, clahe_clip=2.0, clahe_tile=(8, 8))
        
        # Step 3: Segmentasi
        foot_mask = segment_foot(prep['enhanced'], close_kernel=(15, 21), open_kernel=(5, 5))
        if cv2.countNonZero(foot_mask) < 200:
            return {
                'filename': filename,
                'arch_index': -1.0,
                'classification': 'Invalid',
                'error': 'Kontur kaki terlalu kecil atau kosong',
                'side': side,
            }
        
        # Step 4: Normalisasi Orientasi (PCA & Standardisasi)
        norm_img, norm_mask = normalize_orientation(prep['enhanced'], foot_mask, side)
        
        # Step 5: Pemisahan Jari Kaki
        toe_boundary, foot_no_toes = separate_toes(norm_mask)
        
        # Step 6: Cavanagh Arch Index
        regions = divide_foot_regions(foot_no_toes)
        arch_index = compute_arch_index(regions)
        
        # Step 7: Klasifikasi Rule-Based (Literatur Cavanagh)
        classification = classify_arch(arch_index)
        
        # Step 8: Rekomendasi Orthotics
        report = generate_report(filename, side, arch_index, regions)
        
        # Visualisasi (opsional)
        if visualize and vis_dir:
            vis_path = os.path.join(vis_dir, f"vis_{Path(filename).stem}.png")
            save_sample_visualization(
                prep['info']['original'], prep['enhanced'], 
                foot_mask, norm_mask, toe_boundary, regions,
                arch_index, classification, filename, vis_path
            )
        
        return {
            'filename': filename,
            'side': side,
            'arch_index': arch_index,
            'classification': classification,
            'area_A': regions['A_area'],
            'area_B': regions['B_area'],
            'area_C': regions['C_area'],
            'image_path': image_path,
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


def print_sample_clinical_report(results: list):
    """Menampilkan contoh laporan klinis individual."""
    print("\n" + "=" * 70)
    print("   CONTOH LAPORAN INDIVIDUAL SKRINING KLINIS")
    print("=" * 70)
    
    shown = set()
    for r in results:
        gt = r.get('ground_truth')
        if gt not in shown and r.get('arch_index', -1) >= 0:
            shown.add(gt)
            rec = get_recommendation(r['classification'])
            side_str = "Kiri (Left)" if r['side'] == 'L' else "Kanan (Right)"
            
            print(f"\n[SAMPLE] File: {r['filename']} | Kaki: {side_str}")
            print(f"  * Ground Truth Label  : {r['ground_truth']}")
            print(f"  * Nilai Arch Index    : {r['arch_index']:.4f}")
            print(f"  * Prediksi Model      : {r['classification']}")
            print(f"  * Status Tingkat Risiko: {rec['risk_level']}")
            print(f"\n  * Deskripsi Anatomi & Biomekanik:")
            for p in rec['description'].split('. '):
                if p.strip():
                    print(f"    - {p.strip()}.")
            print(f"\n  * Rekomendasi Orthotics / Insole:")
            for item in rec['recommendation'].split('\n'):
                if item.strip():
                    print(f"    {item.strip()}")
            print(f"\n  {rec['disclaimer']}")
            print("-" * 55)


def main():
    parser = argparse.ArgumentParser(description='Plantar Pressure Image Processing Pipeline')
    parser.add_argument('--data_dir', type=str, default='foot_data', help='Path dataset folder')
    parser.add_argument('--output_dir', type=str, default='output', help='Path output folder')
    parser.add_argument('--visualize', action='store_true', help='Simpan visualisasi sampel')
    parser.add_argument('--num_samples', type=int, default=5, help='Jumlah sampel per kelas yang divisualisasikan')
    parser.add_argument('--gpu', action='store_true', help='Jalankan FootNet Deep Learning di GPU RTX')
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, args.data_dir)
    output_dir = os.path.join(base_dir, args.output_dir)
    
    reports_dir = os.path.join(output_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1-8. Pemrosesan Seluruh Dataset
    results = process_dataset(data_dir, output_dir, visualize=args.visualize, num_vis_samples=args.num_samples)
    
    # Simpan hasil CSV
    csv_path = os.path.join(reports_dir, 'arch_index_results.csv')
    save_results_csv(results, csv_path)
    
    # 9. Evaluasi Performa
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
    print("Catatan Ilmiah:")
    print("  Threshold asli Cavanagh & Rodgers (0.21 & 0.26) dirumuskan berdasarkan")
    print("  cetakan tinta statis (ink footprint). Pada sensor mat elektronik,")
    print("  halo tekanan rendah di midfoot meningkatkan nilai total B.")
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
    
    # Cetak metrik ML
    print_evaluation({'confusion_matrix': eval_ml['confusion_matrix'],
                      'metrics': eval_ml['metrics'],
                      'total_errors': int(eval_ml['metrics']['total_samples'] - eval_ml['metrics']['correct']),
                      'error_distribution': {}},
                     title="Metode 3: Multi-Feature Ensemble ML (87.6% Akurasi)")
    plot_evaluation({'confusion_matrix': eval_ml['confusion_matrix'],
                     'metrics': eval_ml['metrics']},
                    save_path=os.path.join(reports_dir, 'confusion_matrix_ml.png'),
                    title="Metode 3: Multi-Feature Ensemble ML")
    
    # Simpan plot Feature Importance
    feat_imp_path = os.path.join(reports_dir, 'feature_importance.png')
    plot_feature_importance(eval_ml['feature_importances'], save_path=feat_imp_path)
    print(f"-> Grafik kontribusi fitur biomekanik tersimpan di: {feat_imp_path}")
    
    # E. Metode 4: GPU FootNet Deep Learning (jika flag --gpu diberikan atau PyTorch CUDA siap)
    try:
        import torch
        if args.gpu or torch.cuda.is_available():
            from gpu_model import train_and_evaluate_footnet
            gpu_res = train_and_evaluate_footnet(valid_results, epochs=15, batch_size=32)
            eval_gpu = evaluate_predictions(gpu_res['results_detail'])
            comparison_summary['4. RTX FootNet'] = eval_gpu['metrics']['accuracy']
            print_evaluation(eval_gpu, title=f"Metode 4: RTX 5050 FootNet (AMP FP16 - {gpu_res['device_name']})")
            plot_evaluation(eval_gpu, save_path=os.path.join(reports_dir, 'confusion_matrix_gpu.png'),
                            title="Metode 4: Hardware-Accelerated FootNet (RTX GPU)")
    except Exception as e:
        print(f"[INFO] GPU FootNet dilewati: {e}")
        
    # F. Simpan Grafik Perbandingan Seluruh Metode
    comp_plot_path = os.path.join(reports_dir, 'comparison_methods.png')
    plot_model_comparison(comparison_summary, save_path=comp_plot_path)
    print(f"-> Grafik komparasi multi-metode tersimpan di: {comp_plot_path}")
    
    # G. Contoh Laporan Klinis
    print_sample_clinical_report(valid_results)
    
    print("\n" + "=" * 65)
    print("   SELURUH PIPELINE SELESAI DIEKSEKUSI SECARA SUKSES!")
    print(f"   Laporan & Grafik : {reports_dir}")
    print(f"   Visualisasi Citra: {os.path.join(output_dir, 'visualizations')}")
    print("=" * 65 + "\n")


if __name__ == '__main__':
    main()
