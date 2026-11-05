"""
=============================================================================
Module: visualization.py
Visualisasi Pipeline — Menampilkan hasil setiap tahap proses
=============================================================================
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_preprocessing_steps(original_bgr, gray, cleaned, filtered, enhanced,
                             save_path=None, title=""):
    """Visualisasi langkah-langkah preprocessing (Step 1-2)."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle(f"Pipeline Preprocessing -- {title}", fontsize=13, fontweight='bold')
    
    if len(original_bgr.shape) == 3:
        orig_disp = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    else:
        orig_disp = original_bgr
        
    images = [
        (orig_disp, "1. Original"),
        (gray, "2. Grayscale Norm"),
        (cleaned, "3. Clean Background"),
        (filtered, "4. Median Filter (5x5)"),
        (enhanced, "5. Enhanced / CLAHE"),
    ]
    
    for ax, (img, label) in zip(axes, images):
        if len(img.shape) == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap='gray')
        ax.set_title(label, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_regions(mask_no_toes, regions, arch_index, classification,
                 save_path=None, title=""):
    """Visualisasi pembagian region A (Forefoot), B (Midfoot), C (Heel)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(f"Analisis Cavanagh Arch Index -- {title}", fontsize=13, fontweight='bold')
    
    h, w = mask_no_toes.shape[:2]
    region_vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    region_vis[regions['A_mask'] > 0] = [220, 70, 70]
    region_vis[regions['B_mask'] > 0] = [70, 200, 70]
    region_vis[regions['C_mask'] > 0] = [70, 110, 230]
    
    axes[0].imshow(region_vis)
    bounds = regions.get('boundaries', (0, 0, 0, 0))
    if bounds[0] != bounds[3]:
        axes[0].axhline(y=bounds[1], color='yellow', linestyle='--', linewidth=1.5)
        axes[0].axhline(y=bounds[2], color='yellow', linestyle='--', linewidth=1.5)
    
    axes[0].set_title("Region: Forefoot (A) | Midfoot (B) | Heel (C)", fontsize=11)
    axes[0].axis('off')
    
    legend_patches = [
        mpatches.Patch(color=[220/255, 70/255, 70/255], 
                       label=f"A (Forefoot): {regions['A_area']} px"),
        mpatches.Patch(color=[70/255, 200/255, 70/255], 
                       label=f"B (Midfoot):  {regions['B_area']} px"),
        mpatches.Patch(color=[70/255, 110/255, 230/255], 
                       label=f"C (Heel):     {regions['C_area']} px"),
    ]
    axes[0].legend(handles=legend_patches, loc='lower right', fontsize=9)
    
    total = regions['A_area'] + regions['B_area'] + regions['C_area']
    info_text = (
        "HASIL EKSTRAKSI FITUR\n"
        "------------------------------------\n\n"
        f"Area A (Forefoot) : {regions['A_area']:>8} px\n"
        f"Area B (Midfoot)  : {regions['B_area']:>8} px\n"
        f"Area C (Heel)     : {regions['C_area']:>8} px\n"
        f"Total Area        : {total:>8} px\n\n"
        "------------------------------------\n"
        f"Arch Index (AI)   : {arch_index:.4f}\n"
        "Rumus: AI = B / (A + B + C)\n\n"
        "------------------------------------\n"
        f"Klasifikasi       : {classification}\n"
        "------------------------------------\n\n"
        "Threshold Cavanagh & Rodgers:\n"
        "  AI <= 0.21        : High Arch\n"
        "  0.21 < AI < 0.26  : Normal\n"
        "  AI >= 0.26        : Flat Foot"
    )
    
    axes[1].text(0.1, 0.95, info_text, transform=axes[1].transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='#fbfbe8', edgecolor='#ccc'))
    axes[1].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_evaluation(eval_result, save_path=None, title="Evaluasi Performa"):
    """Plot confusion matrix (count & normalized) serta metrik per kelas."""
    cm = eval_result['confusion_matrix']
    metrics = eval_result['metrics']
    classes = ['High Arch', 'Normal', 'Flat Foot']
    
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    
    # 1. Confusion Matrix Raw Count
    ax = axes[0]
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title("Confusion Matrix (Jumlah Sampel)", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=25, ha='right', fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_ylabel('Ground Truth', fontsize=10)
    ax.set_xlabel('Prediksi Model', fontsize=10)
    
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=13, fontweight='bold')
    
    # 2. Normalized Confusion Matrix
    ax2 = axes[1]
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
    im2 = ax2.imshow(cm_norm, interpolation='nearest', cmap='Greens', vmin=0, vmax=1)
    ax2.set_title("Normalized Confusion Matrix (%)", fontsize=11)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    ax2.set_xticks(range(len(classes)))
    ax2.set_yticks(range(len(classes)))
    ax2.set_xticklabels(classes, rotation=25, ha='right', fontsize=9)
    ax2.set_yticklabels(classes, fontsize=9)
    ax2.set_ylabel('Ground Truth', fontsize=10)
    ax2.set_xlabel('Prediksi Model', fontsize=10)
    
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax2.text(j, i, f"{cm_norm[i, j]:.1%}",
                     ha="center", va="center",
                     color="white" if cm_norm[i, j] > 0.5 else "black",
                     fontsize=11, fontweight='bold')
    
    # 3. Bar chart metrik per kelas
    ax3 = axes[2]
    x = np.arange(len(classes))
    width = 0.2
    
    sens = [metrics['per_class'][c]['sensitivity'] for c in classes]
    spec = [metrics['per_class'][c]['specificity'] for c in classes]
    prec = [metrics['per_class'][c]['precision'] for c in classes]
    f1s = [metrics['per_class'][c]['f1_score'] for c in classes]
    
    bars1 = ax3.bar(x - 1.5*width, sens, width, label='Sensitivity', color='#e74c3c')
    bars2 = ax3.bar(x - 0.5*width, spec, width, label='Specificity', color='#3498db')
    bars3 = ax3.bar(x + 0.5*width, prec, width, label='Precision', color='#2ecc71')
    bars4 = ax3.bar(x + 1.5*width, f1s, width, label='F1-Score', color='#f39c12')
    
    ax3.set_ylabel('Skor (0.0 - 1.0)', fontsize=10)
    ax3.set_title(f"Metrik per Kelas (Akurasi: {metrics['accuracy']:.1%})", fontsize=11)
    ax3.set_xticks(x)
    ax3.set_xticklabels(classes, fontsize=9)
    ax3.legend(fontsize=8, loc='lower right')
    ax3.set_ylim(0, 1.15)
    ax3.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{height:.2f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_ai_distribution(results: list, save_path=None):
    """Plot distribusi Arch Index per kelas ground truth."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Distribusi Nilai Arch Index per Kelas Ground Truth", fontsize=13, fontweight='bold')
    
    classes = ['High Arch', 'Normal', 'Flat Foot']
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    
    ax = axes[0]
    for cls, color in zip(classes, colors):
        ai_values = [r['arch_index'] for r in results 
                     if r.get('ground_truth') == cls and r.get('arch_index', -1) >= 0]
        if ai_values:
            ax.hist(ai_values, bins=25, alpha=0.5, label=f"{cls} (n={len(ai_values)})", 
                    color=color, edgecolor='black', linewidth=0.5)
    
    ax.axvline(x=0.21, color='red', linestyle='--', linewidth=2, label='Threshold 0.21 (Cavanagh)')
    ax.axvline(x=0.26, color='blue', linestyle='--', linewidth=2, label='Threshold 0.26 (Cavanagh)')
    
    ax.set_xlabel('Arch Index', fontsize=10)
    ax.set_ylabel('Frekuensi', fontsize=10)
    ax.set_title('Histogram Sebaran Arch Index', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    ax2 = axes[1]
    data_per_class = []
    for cls in classes:
        ai_values = [r['arch_index'] for r in results 
                     if r.get('ground_truth') == cls and r.get('arch_index', -1) >= 0]
        data_per_class.append(ai_values)
    
    bp = ax2.boxplot(data_per_class, labels=classes, patch_artist=True,
                     medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax2.axhline(y=0.21, color='red', linestyle='--', linewidth=1.5, label='Threshold 0.21')
    ax2.axhline(y=0.26, color='blue', linestyle='--', linewidth=1.5, label='Threshold 0.26')
    
    ax2.set_ylabel('Arch Index', fontsize=10)
    ax2.set_title('Boxplot Distribusi Arch Index', fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(importances: dict, save_path=None):
    """Plot ranking kontribusi fitur biomekanik citra."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_items = sorted(importances.items(), key=lambda x: x[1])
    features = [k for k, v in sorted_items]
    scores = [v for k, v in sorted_items]
    
    bars = ax.barh(features, scores, color='#2980b9', edgecolor='black', alpha=0.85)
    ax.set_xlabel('Importance Score (Relative Weight)', fontsize=10)
    ax.set_title('Kontribusi Fitur Biomekanik Citra Plantar terhadap Klasifikasi', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f'{w:.3f}', xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=9)
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_model_comparison(comparison_data: dict, save_path=None):
    """Plot perbandingan akurasi multi-metode (Rule vs Calibrated vs ML Ensemble)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    methods = list(comparison_data.keys())
    accuracies = [v * 100 for v in comparison_data.values()]
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    
    bars = ax.bar(methods, accuracies, color=colors[:len(methods)], width=0.45, edgecolor='black', alpha=0.9)
    ax.set_ylabel('Akurasi Keseluruhan (%)', fontsize=11)
    ax.set_title('Perbandingan Performa: Rule-Based vs Sensor-Calibrated vs Multi-Feature Model', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_sample_visualization(original_bgr, enhanced, foot_mask, 
                             norm_mask, toe_boundary, regions,
                             arch_index, classification, filename,
                             save_path, modality_str: str = "Citra Tekanan"):
    """Visualisasi multi-panel satu sampel (4 tahap visual)."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.suptitle(f"{filename}  |  Arch Index = {arch_index:.4f}  |  Klasifikasi: {classification}  |  [{modality_str}]",
                 fontsize=12, fontweight='bold')
    
    # Panel 1: Citra Asli Input
    if len(original_bgr.shape) == 3:
        axes[0].imshow(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
    else:
        axes[0].imshow(original_bgr, cmap='gray')
    axes[0].set_title("1. Citra Asli Input")
    axes[0].axis('off')
    
    # Panel 2: Enhanced + Segmentasi Mask
    overlay = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    mask_layer = np.zeros_like(overlay)
    mask_layer[foot_mask > 0] = [0, 220, 0]
    blended = cv2.addWeighted(overlay, 0.7, mask_layer, 0.3, 0)
    axes[1].imshow(blended)
    axes[1].set_title("2. Preprocessed & Mask")
    axes[1].axis('off')
    
    # Panel 3: Normalisasi Orientasi & Pemotongan Jari Kaki
    mask_rgb = cv2.cvtColor(norm_mask, cv2.COLOR_GRAY2RGB)
    if 0 < toe_boundary < norm_mask.shape[0]:
        cv2.line(mask_rgb, (0, toe_boundary), (norm_mask.shape[1], toe_boundary), (255, 230, 0), 2)
    axes[2].imshow(mask_rgb)
    axes[2].set_title(f"3. Normalisasi & Toe Cut (y={toe_boundary})")
    axes[2].axis('off')
    
    # Panel 4: Regio Cavanagh (A, B, C)
    h, w = norm_mask.shape[:2]
    region_vis = np.zeros((h, w, 3), dtype=np.uint8)
    region_vis[regions['A_mask'] > 0] = [220, 70, 70]
    region_vis[regions['B_mask'] > 0] = [70, 200, 70]
    region_vis[regions['C_mask'] > 0] = [70, 110, 230]
    axes[3].imshow(region_vis)
    
    axes[3].set_title(f"4. Regio (A={regions['A_area']}, B={regions['B_area']}, C={regions['C_area']})")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
