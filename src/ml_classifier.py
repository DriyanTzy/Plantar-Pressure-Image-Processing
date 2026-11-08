"""
=============================================================================
Module: ml_classifier.py
Robust Machine Learning Classifier dengan Proteksi Anti-Overfitting
=============================================================================
Penyempurnaan Anti-Overfitting (Generalisasi Sejati):
  1. Group-Aware Cross Validation (GroupKFold):
     Mencegah data leakage. Citra dasar dan seluruh varian augmentasinya
     (flip horizontal, flip vertikal, dll.) DIJAMIN berada di fold yang sama.
     Model TIDAK BISA sekadar menghafal (memorize) bentuk kaki individu.
  2. Regularisasi Ketat:
     - L2 regularization (l2_regularization=2.0)
     - Min samples leaf (min_samples_leaf=15)
     - Max depth dibatasi (max_depth=4)
     - Learning rate terukur (0.05)
  3. Feature Importance Ilmiah:
     Mengukur bobot matematis fitur morfologi telapak kaki.
=============================================================================
"""

import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CLASS_NAMES = ['High Arch', 'Normal', 'Flat Foot']


def extract_base_id(filename: str) -> str:
    """
    Mengekstrak ID citra dasar (induk) dengan membuang tag augmentasi.
    Contoh: 'aiji_foot_R_11_H_aug_fliph.png' -> 'aiji_foot_11_H'
    """
    stem = Path(filename).stem
    # Hapus tag augmentasi
    clean = stem.split('_aug')[0]
    # Normalisasi sisi agar L dan R dari subjek yang sama tetap 1 grup
    clean = clean.replace('_foot_L_', '_foot_').replace('_foot_R_', '_foot_')
    return clean


def prepare_feature_matrix(results_list: list) -> tuple:
    """Mengekstrak matriks fitur multidimensi dan grup induk."""
    valid = [r for r in results_list if r.get('arch_index', -1) >= 0]
    
    data = []
    groups = []
    
    for r in valid:
        A = float(r.get('area_A', 0))
        B = float(r.get('area_B', 0))
        C = float(r.get('area_C', 0))
        tot = max(1.0, A + B + C)
        ai = float(r.get('arch_index', 0))
        fname = r.get('filename', '')
        
        data.append({
            'filename': fname,
            'ground_truth': r.get('ground_truth', 'Normal'),
            'arch_index': ai,
            'ratio_A': A / tot,
            'ratio_B': B / tot,
            'ratio_C': C / tot,
            'area_A': A,
            'area_B': B,
            'area_C': C,
            'total_contact_area': tot,
            'fore_to_heel_ratio': A / max(1.0, C),
        })
        groups.append(extract_base_id(fname))
        
    df = pd.DataFrame(data)
    feature_cols = [
        'arch_index', 'ratio_A', 'ratio_B', 'ratio_C',
        'area_A', 'area_B', 'area_C', 'total_contact_area',
        'fore_to_heel_ratio'
    ]
    
    X = df[feature_cols].values
    y = df['ground_truth'].values
    groups = np.array(groups)
    
    return X, y, groups, df, feature_cols


def train_and_cross_validate_ml(results_list: list) -> dict:
    """
    Evaluasi performa menggunakan GroupKFold (Zero Data Leakage).
    Menghasilkan skor akurasi yang jujur, presisi, dan generalisir untuk data luar.
    """
    X, y, groups, df, feature_cols = prepare_feature_matrix(results_list)
    
    # Model teregulasi ketat untuk mencegah memorisasi (anti-overfitting)
    clf = HistGradientBoostingClassifier(
        max_iter=80,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=15,
        l2_regularization=2.0,
        random_state=42
    )
    
    # 5-Fold GroupKFold berdasarkan ID subjek/induk kaki
    gkf = GroupKFold(n_splits=5)
    y_pred = cross_val_predict(clf, X, y, groups=groups, cv=gkf)
    
    # Latih model final pada seluruh data teregulasi
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=5, 
        min_samples_leaf=10, 
        random_state=42, 
        n_jobs=-1
    )
    rf.fit(X, y)
    importances = dict(zip(feature_cols, [round(float(v), 4) for v in rf.feature_importances_]))
    
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred, labels=CLASS_NAMES)
    
    from evaluation import compute_metrics
    metrics = compute_metrics(cm, CLASS_NAMES)
    
    eval_result = {
        'confusion_matrix': cm,
        'metrics': metrics,
        'accuracy': acc,
        'y_true': y.tolist(),
        'y_pred': y_pred.tolist(),
        'feature_importances': importances,
        'model_rf': rf,
        'feature_cols': feature_cols,
    }
    
    return eval_result
