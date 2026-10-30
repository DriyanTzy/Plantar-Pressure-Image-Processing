"""
=============================================================================
Module: evaluation.py
Pipeline Step 9 — Evaluasi Performa (Confusion Matrix, Akurasi, dll.)
=============================================================================
Fungsi utama:
  - evaluate_predictions() : Hitung confusion matrix, akurasi, sensitivity,
                             specificity terhadap ground truth label.
  - analyze_errors()       : Identifikasi sampel yang salah diklasifikasikan.
  - print_evaluation()     : Cetak laporan evaluasi ke console (ASCII safe).
=============================================================================
"""

import numpy as np
from collections import Counter
from tabulate import tabulate


# Label mapping
LABEL_MAP = {
    'H': 'High Arch',
    'N': 'Normal',
    'L': 'Flat Foot',
}
CLASS_NAMES = ['High Arch', 'Normal', 'Flat Foot']


def build_confusion_matrix(y_true: list, y_pred: list,
                            classes: list = None) -> np.ndarray:
    """
    Bangun confusion matrix secara manual (tanpa dependensi eksternal).
    
    Args:
        y_true: List label ground truth.
        y_pred: List label prediksi.
        classes: Urutan nama kelas.
        
    Returns:
        np.ndarray confusion matrix (baris=true, kolom=pred).
    """
    if classes is None:
        classes = CLASS_NAMES
    
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    for true, pred in zip(y_true, y_pred):
        if true in class_to_idx and pred in class_to_idx:
            cm[class_to_idx[true]][class_to_idx[pred]] += 1
    
    return cm


def compute_metrics(cm: np.ndarray, classes: list = None) -> dict:
    """
    Hitung metrik evaluasi komprehensif dari confusion matrix:
      - Sensitivity / Recall
      - Specificity
      - Precision
      - F1-Score
      - Overall Accuracy & Macro Averages
      
    Args:
        cm: Confusion matrix numpy array.
        classes: List nama kelas.
        
    Returns:
        dict berisi hasil metrik per kelas dan agregat macro.
    """
    if classes is None:
        classes = CLASS_NAMES
    
    n = len(classes)
    total = int(cm.sum())
    correct = int(np.trace(cm))
    accuracy = correct / total if total > 0 else 0.0
    
    per_class = {}
    for i, cls in enumerate(classes):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(total - tp - fn - fp)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * sensitivity / (precision + sensitivity)
              if (precision + sensitivity) > 0 else 0.0)
        
        per_class[cls] = {
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'sensitivity': round(sensitivity, 4),
            'specificity': round(specificity, 4),
            'precision': round(precision, 4),
            'f1_score': round(f1, 4),
            'support': int(tp + fn),
        }
    
    macro_sens = float(np.mean([v['sensitivity'] for v in per_class.values()]))
    macro_spec = float(np.mean([v['specificity'] for v in per_class.values()]))
    macro_prec = float(np.mean([v['precision'] for v in per_class.values()]))
    macro_f1 = float(np.mean([v['f1_score'] for v in per_class.values()]))
    
    return {
        'accuracy': round(accuracy, 4),
        'total_samples': total,
        'correct': correct,
        'per_class': per_class,
        'macro_avg': {
            'sensitivity': round(macro_sens, 4),
            'specificity': round(macro_spec, 4),
            'precision': round(macro_prec, 4),
            'f1_score': round(macro_f1, 4),
        }
    }


def analyze_errors(results: list) -> tuple:
    """
    Identifikasi dan analisis sampel yang salah diklasifikasikan.
    
    Args:
        results: List dict hasil prediksi.
        
    Returns:
        Tuple (errors_list, error_distribution_dict).
    """
    errors = []
    for r in results:
        if r.get('ground_truth') != r.get('prediction'):
            errors.append({
                'filename': r.get('filename'),
                'ground_truth': r.get('ground_truth'),
                'prediction': r.get('prediction'),
                'arch_index': r.get('arch_index'),
                'error_type': f"{r.get('ground_truth')} -> {r.get('prediction')}",
            })
    
    error_counts = Counter(e['error_type'] for e in errors)
    return errors, dict(error_counts)


def evaluate_predictions(results: list) -> dict:
    """
    Pipeline evaluasi lengkap terhadap list hasil.
    """
    y_true = [r['ground_truth'] for r in results]
    y_pred = [r['prediction'] for r in results]
    
    cm = build_confusion_matrix(y_true, y_pred)
    metrics = compute_metrics(cm)
    errors, error_distribution = analyze_errors(results)
    
    return {
        'confusion_matrix': cm,
        'metrics': metrics,
        'errors': errors,
        'error_distribution': error_distribution,
        'total_errors': len(errors),
    }


def print_evaluation(eval_result: dict, title: str = "Klasifikasi Literatur Cavanagh"):
    """
    Cetak laporan evaluasi ke console secara bersih dan aman untuk semua encoding.
    """
    metrics = eval_result['metrics']
    cm = eval_result['confusion_matrix']
    
    print("\n" + "=" * 70)
    print(f"   LAPORAN EVALUASI PERFORMA SISTEM")
    print(f"   {title}")
    print("=" * 70)
    
    print(f"\n{'Total Sampel':<25}: {metrics['total_samples']}")
    print(f"{'Prediksi Benar':<25}: {metrics['correct']}")
    print(f"{'Prediksi Salah':<25}: {eval_result['total_errors']}")
    print(f"{'Overall Accuracy':<25}: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    
    print("\n" + "-" * 55)
    print("CONFUSION MATRIX")
    print("-" * 55)
    print(f"{'True \\ Pred':<15}", end="")
    for cls in CLASS_NAMES:
        print(f"{cls:>13}", end="")
    print()
    print("-" * 55)
    for i, cls in enumerate(CLASS_NAMES):
        print(f"{cls:<15}", end="")
        for j in range(len(CLASS_NAMES)):
            print(f"{cm[i][j]:>13}", end="")
        print()
    
    print("\n" + "-" * 55)
    print("METRIK EVALUASI PER KELAS")
    print("-" * 55)
    
    table_data = []
    for cls in CLASS_NAMES:
        m = metrics['per_class'][cls]
        table_data.append([
            cls,
            m['support'],
            f"{m['sensitivity']:.4f}",
            f"{m['specificity']:.4f}",
            f"{m['precision']:.4f}",
            f"{m['f1_score']:.4f}",
        ])
    
    ma = metrics['macro_avg']
    table_data.append([
        'Macro Avg', metrics['total_samples'],
        f"{ma['sensitivity']:.4f}",
        f"{ma['specificity']:.4f}",
        f"{ma['precision']:.4f}",
        f"{ma['f1_score']:.4f}",
    ])
    
    print(tabulate(table_data,
                   headers=['Kelas', 'Support', 'Sensitivity', 
                            'Specificity', 'Precision', 'F1-Score'],
                   tablefmt='grid'))
    
    if eval_result['total_errors'] > 0:
        print("\n" + "-" * 55)
        print("DISTRIBUSI ERROR")
        print("-" * 55)
        for error_type, count in sorted(eval_result['error_distribution'].items(),
                                         key=lambda x: -x[1]):
            print(f"  * {error_type:<28}: {count} sampel")
    
    print("\n" + "=" * 70)
