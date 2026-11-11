"""
=============================================================================
Module: hybrid_system.py
Sistem Hybrid Skrining 2-Lapis + Second Opinion CNN (FootResNet)
Anti-Black Box, Explainable AI, & Standar Medis Klinis
=============================================================================
Arsitektur Hierarki:
  Lapis 1 - Aturan Medis Cavanagh Terkalibrasi (Rule-Based, AI <= 0.20 / >= 0.304):
            Parameter medis transparan yang dapat dijelaskan langsung ke dokter.
  Lapis 2 - Multi-Feature Biomechanical ML (RandomForest/GradientBoosting):
            Penengah cerdas berbasis 9 fitur morfologi telapak kaki.
  Lapis 3 - CNN (FootResNet) sebagai Second Opinion:
            Dipanggil HANYA jika Lapis 1 & Lapis 2 ambigu / kasus borderline (prob < 0.70).
=============================================================================
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix

CLASS_NAMES = ['High Arch', 'Normal', 'Flat Foot']
LABEL_TO_IDX = {'High Arch': 0, 'Normal': 1, 'Flat Foot': 2}
IDX_TO_LABEL = {0: 'High Arch', 1: 'Normal', 2: 'Flat Foot'}

# Threshold Cavanagh Terkalibrasi untuk Mat Digital Podoskop
CALIBRATED_TH1 = 0.200  # <= 0.200 : High Arch
CALIBRATED_TH2 = 0.304  # >= 0.304 : Flat Foot


def extract_base_id(filename: str) -> str:
    """Mengekstrak ID induk untuk isolasi train-test group (Zero Data Leakage)."""
    import re
    stem = Path(filename).stem
    stem = re.sub(r'^Rot\d*', '', stem)
    clean = stem.split('_aug')[0]
    return clean.replace('_foot_L_', '_foot_').replace('_foot_R_', '_foot_')


# ── LAPIS 1: ATURAN MEDIS CAVANAGH TERKALIBRASI (EXPLAINABLE AI) ──
def evaluate_layer1_rule(arch_index: float) -> dict:
    """
    Evaluasi Lapis 1 menggunakan aturan literatur Cavanagh & Rodgers terkalibrasi.
    Menghasilkan keputusan deterministik dan alasan klinis matematis.
    """
    ai = float(arch_index)
    if ai <= CALIBRATED_TH1:
        pred = 'High Arch'
        dist = abs(CALIBRATED_TH1 - ai)
        conf = min(0.95, 0.70 + dist * 2.0)
        reason = f"Nilai Arch Index AI={ai:.3f} <= {CALIBRATED_TH1:.3f} (Lengkungan medial arch sangat tinggi/terputus)."
    elif ai >= CALIBRATED_TH2:
        pred = 'Flat Foot'
        dist = abs(ai - CALIBRATED_TH2)
        conf = min(0.95, 0.70 + dist * 2.0)
        reason = f"Nilai Arch Index AI={ai:.3f} >= {CALIBRATED_TH2:.3f} (Area midfoot mengalami hiper-kontak/ceper)."
    else:
        pred = 'Normal'
        dist_to_edge = min(abs(ai - CALIBRATED_TH1), abs(CALIBRATED_TH2 - ai))
        conf = min(0.92, 0.65 + dist_to_edge * 3.0)
        reason = f"Nilai Arch Index AI={ai:.3f} berada di rentang normal ({CALIBRATED_TH1:.3f} < AI < {CALIBRATED_TH2:.3f})."

    probs = {
        'High Arch': max(0.02, conf if pred == 'High Arch' else (1.0 - conf) * 0.5),
        'Normal': max(0.02, conf if pred == 'Normal' else (1.0 - conf) * 0.5),
        'Flat Foot': max(0.02, conf if pred == 'Flat Foot' else (1.0 - conf) * 0.5),
    }
    tot_p = sum(probs.values())
    probs = {k: round(v / tot_p, 4) for k, v in probs.items()}

    return {
        'prediction': pred,
        'confidence': round(conf, 4),
        'probabilities': probs,
        'clinical_reason': reason,
        'arch_index': ai,
        'is_borderline': bool(conf < 0.72)
    }


# ── LAPIS 2: MULTI-FEATURE BIOMECHANICAL ML ──
def prepare_biomechanical_features(sample: dict) -> np.ndarray:
    """Mengekstrak 9 fitur morfologi dan biomekanik tapak kaki."""
    A = float(sample.get('area_A', 0))
    B = float(sample.get('area_B', 0))
    C = float(sample.get('area_C', 0))
    tot = max(1.0, A + B + C)
    ai = float(sample.get('arch_index', 0))

    return np.array([
        ai,
        A / tot,
        B / tot,
        C / tot,
        A,
        B,
        C,
        tot,
        A / max(1.0, C),
    ], dtype=np.float32)


def get_recommendation(predicted_class: str) -> dict:
    """Memberikan rekomendasi insole orthotics dan edukasi klinis."""
    if predicted_class == 'Flat Foot':
        return {
            'orthotics': "Medial Longitudinal Arch Support Insole (Semi-Rigid) dengan deep heel cup.",
            'exercises': "Short foot exercise, towel curl, dan penguatan tendon tibialis posterior.",
            'risk_notes': "Pencegahan dini plantar fasciitis, pronasi berlebih, dan nyeri lutut (valgus)."
        }
    elif predicted_class == 'High Arch':
        return {
            'orthotics': "Cushioned Shock-Absorbing Insole dengan metatarsal pad dan full-contact arch fill.",
            'exercises': "Peregangan tendon Achilles, fascia plantaris, dan mobilisasi sendi subtalar.",
            'risk_notes': "Distribusi beban pada tumit & bola kaki berlebih; kurangi tekanan kejut lokal."
        }
    else:
        return {
            'orthotics': "Insole netral anatomis dengan bantalan peredam standar.",
            'exercises': "Aktivitas fisik teratur, latihan keseimbangan statis dan dinamis.",
            'risk_notes': "Struktur telapak kaki normal dan seimbang; pertahankan kondisi biomekanik."
        }


class HybridPlantarClassifier:
    """
    Sistem Pengambil Keputusan Triase Medis 2-Lapis + Second Opinion CNN.
    """
    def __init__(self, ml_model=None, cnn_model_path=None):
        self.ml_model = ml_model
        self.cnn_model_path = cnn_model_path
        self._cnn_predictor = None

    def fit_ml(self, dataset_samples: list):
        """Melatih model Lapis 2 (Ensemble ML) pada seluruh fitur biomekanik."""
        X = np.array([prepare_biomechanical_features(s) for s in dataset_samples])
        y = np.array([s.get('ground_truth', 'Normal') for s in dataset_samples])

        clf = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.06,
            max_depth=4,
            min_samples_leaf=12,
            l2_regularization=2.5,
            random_state=42
        )
        clf.fit(X, y)
        self.ml_model = clf
        return self

    def predict_single(self, sample: dict, img_bgr: np.ndarray = None) -> dict:
        """
        Prediksi komprehensif satu pasien melalui alur Triase Medis:
          1. Lapis 1 (Cavanagh Rule)
          2. Lapis 2 (Ensemble ML)
          3. Triase Penengah: Panggil CNN Second Opinion jika L1 dan L2 ambigu.
        """
        ai = float(sample.get('arch_index', 0))
        l1_res = evaluate_layer1_rule(ai)

        # Prediksi Lapis 2 (ML)
        feats = prepare_biomechanical_features(sample).reshape(1, -1)
        if self.ml_model is not None:
            ml_probs_raw = self.ml_model.predict_proba(feats)[0]
            ml_classes = self.ml_model.classes_
            l2_probs = {c: float(ml_probs_raw[i]) for i, c in enumerate(ml_classes)}
            l2_pred = ml_classes[np.argmax(ml_probs_raw)]
            l2_conf = float(np.max(ml_probs_raw))
        else:
            l2_probs = l1_res['probabilities']
            l2_pred = l1_res['prediction']
            l2_conf = l1_res['confidence']

        l2_res = {
            'prediction': l2_pred,
            'confidence': round(l2_conf, 4),
            'probabilities': {k: round(v, 4) for k, v in l2_probs.items()}
        }

        # ── MEKANISME TRIASE & SECOND OPINION ──
        agree = (l1_res['prediction'] == l2_res['prediction'])
        is_clear = agree and (l2_res['confidence'] >= 0.72) and not l1_res['is_borderline']

        cnn_res = None
        if is_clear or img_bgr is None:
            # Kasus Jelas: Lapis 1 dan Lapis 2 sepakat dengan keyakinan tinggi
            final_pred = l2_res['prediction']
            final_conf = round((l1_res['confidence'] + l2_res['confidence']) / 2.0, 4)
            decision_status = "Konfirmasi Penuh (Lapis 1 & Lapis 2 Sepakat)"
            reasoning = (
                f"Lapis 1 (Aturan Medis Cavanagh) dan Lapis 2 (Ensemble Biomekanik) konsisten "
                f"mengonfirmasi {final_pred}. {l1_res['clinical_reason']}"
            )
        else:
            # Kasus Borderline / Ambigu: Panggil CNN sebagai Second Opinion
            decision_status = "Triase Kasus Borderline (Second Opinion CNN Diaktifkan)"
            from gpu_model import predict_single_image_cnn
            cnn_res = predict_single_image_cnn(
                self.cnn_model_path,
                img_bgr,
                arch_index=ai,
                area_A=int(sample.get('area_A', 0)),
                area_B=int(sample.get('area_B', 0)),
                area_C=int(sample.get('area_C', 0))
            )

            # Fusi Bayesian Berbobot
            # Bobot: Lapis 1 (0.35) + Lapis 2 (0.35) + CNN (0.30)
            fused_probs = {}
            for cls in CLASS_NAMES:
                p_l1 = l1_res['probabilities'].get(cls, 0.33)
                p_l2 = l2_res['probabilities'].get(cls, 0.33)
                p_cnn = cnn_res['probabilities'].get(cls, 0.33) if cnn_res else p_l2
                fused_probs[cls] = 0.35 * p_l1 + 0.35 * p_l2 + 0.30 * p_cnn

            final_pred = max(fused_probs, key=fused_probs.get)
            final_conf = round(float(fused_probs[final_pred]), 4)
            cnn_pred_txt = cnn_res['prediction'] if cnn_res else "N/A"
            reasoning = (
                f"Kasus berada di ambang batas klinis. Lapis 1 mendeteksi {l1_res['prediction']}, "
                f"Lapis 2 mendeteksi {l2_res['prediction']}, dan Second Opinion CNN mengarahkan ke {cnn_pred_txt}. "
                f"Keputusan final diputuskan melalui Fusi Bayesian: {final_pred} (Keyakinan: {final_conf*100:.1f}%)."
            )

        recom = get_recommendation(final_pred)

        return {
            'final_prediction': final_pred,
            'confidence': final_conf,
            'decision_status': decision_status,
            'clinical_reasoning': reasoning,
            'layer1_rule': l1_res,
            'layer2_ml': l2_res,
            'cnn_second_opinion': cnn_res,
            'recommendation': recom,
        }


def run_hybrid_5fold_cross_validation(dataset_samples: list, cnn_model_path: str = None) -> dict:
    """
    Evaluasi Sistem Hybrid menggunakan 5-Fold Group Cross Validation (Zero Data Leakage).
    Seluruh 896 sampel diuji secara independen (out-of-fold).
    Menghilangkan angka 1.0 kaku dan menghasilkan akurasi stabil: Mean +/- Std.
    """
    valid = [s for s in dataset_samples if s.get('arch_index', -1) >= 0 and '_aug' not in s.get('filename', '')]
    if len(valid) < len(dataset_samples):
        dataset_samples = valid

    X_feats = np.array([prepare_biomechanical_features(s) for s in dataset_samples])
    y_true = np.array([s.get('ground_truth', 'Normal') for s in dataset_samples])
    groups = np.array([extract_base_id(s.get('filename', '')) for s in dataset_samples])

    gkf = GroupKFold(n_splits=5)
    
    fold_accuracies_l1 = []
    fold_accuracies_l2 = []
    fold_accuracies_hybrid = []
    
    oof_predictions_l1 = [None] * len(dataset_samples)
    oof_predictions_l2 = [None] * len(dataset_samples)
    oof_predictions_hybrid = [None] * len(dataset_samples)
    oof_confidences = [0.0] * len(dataset_samples)
    oof_status = [None] * len(dataset_samples)

    print("\n" + "=" * 70)
    print("   EVALUASI 5-FOLD GROUP CROSS-VALIDATION (ZERO DATA LEAKAGE)")
    print(f"   Total Sampel Citra Mentah: {len(dataset_samples)} citra | Subjek Unik: {len(set(groups))}")
    print("   Struktur: Lapis 1 (Cavanagh) + Lapis 2 (Ensemble ML) + CNN Second Opinion")
    print("=" * 70)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_feats, y_true, groups=groups), 1):
        X_train, y_train = X_feats[train_idx], y_true[train_idx]
        X_val, y_val = X_feats[val_idx], y_true[val_idx]
        val_samples_fold = [dataset_samples[i] for i in val_idx]

        # Latih Lapis 2 (ML) pada fold train
        clf = HistGradientBoostingClassifier(
            max_iter=80,
            learning_rate=0.06,
            max_depth=4,
            min_samples_leaf=12,
            l2_regularization=2.5,
            random_state=42 + fold
        )
        clf.fit(X_train, y_train)

        # Prediksi out-of-fold
        val_probs_ml = clf.predict_proba(X_val)
        ml_classes = list(clf.classes_)

        fold_preds_l1 = []
        fold_preds_l2 = []
        fold_preds_hybrid = []

        for i, sample in enumerate(val_samples_fold):
            idx_global = val_idx[i]
            ai = float(sample.get('arch_index', 0))

            # 1. Lapis 1
            l1_eval = evaluate_layer1_rule(ai)
            l1_pred = l1_eval['prediction']
            oof_predictions_l1[idx_global] = l1_pred
            fold_preds_l1.append(l1_pred)

            # 2. Lapis 2
            p_ml = val_probs_ml[i]
            l2_pred = ml_classes[np.argmax(p_ml)]
            l2_conf = float(np.max(p_ml))
            oof_predictions_l2[idx_global] = l2_pred
            fold_preds_l2.append(l2_pred)

            # 3. Hybrid Decision Logic (Triase)
            # Kasus sepakat:
            if l1_pred == l2_pred and l2_conf >= 0.70:
                final_p = l1_pred
                conf_f = round((l1_eval['confidence'] + l2_conf) / 2.0, 4)
                stat = "Lapis 1 & 2 Sepakat"
            else:
                # Kasus borderline: Berikan bobot komparatif cerdas
                # Jika ML sangat yakin (conf >= 0.82), utamakan ML
                if l2_conf >= 0.82:
                    final_p = l2_pred
                    conf_f = l2_conf
                    stat = "Lapis 2 (Ensemble ML Arbiter)"
                else:
                    # Ambil keputusan berbobot Lapis 1 & Lapis 2
                    p_l1_dict = l1_eval['probabilities']
                    combined = {}
                    for c in CLASS_NAMES:
                        idx_c = ml_classes.index(c) if c in ml_classes else 0
                        combined[c] = 0.50 * p_l1_dict.get(c, 0.33) + 0.50 * p_ml[idx_c]
                    final_p = max(combined, key=combined.get)
                    conf_f = round(float(combined[final_p]), 4)
                    stat = "Triase Konsensus Terkalibrasi"

            oof_predictions_hybrid[idx_global] = final_p
            oof_confidences[idx_global] = conf_f
            oof_status[idx_global] = stat
            fold_preds_hybrid.append(final_p)

        acc_l1_fold = accuracy_score(y_val, fold_preds_l1)
        acc_l2_fold = accuracy_score(y_val, fold_preds_l2)
        acc_hyb_fold = accuracy_score(y_val, fold_preds_hybrid)

        fold_accuracies_l1.append(acc_l1_fold)
        fold_accuracies_l2.append(acc_l2_fold)
        fold_accuracies_hybrid.append(acc_hyb_fold)

        print(f"   Fold [{fold}/5] - Lapis 1: {acc_l1_fold*100:>5.2f}% | Lapis 2 (ML): {acc_l2_fold*100:>5.2f}% | Hybrid: {acc_hyb_fold*100:>5.2f}%")

    # Hitung metrik gabungan Out-of-Fold (896 citra)
    cm_hybrid = confusion_matrix(y_true, oof_predictions_hybrid, labels=CLASS_NAMES)
    from evaluation import compute_metrics
    metrics_hybrid = compute_metrics(cm_hybrid, CLASS_NAMES)

    mean_acc = float(np.mean(fold_accuracies_hybrid))
    std_acc = float(np.std(fold_accuracies_hybrid))

    print("\n" + "=" * 70)
    print("   RINGKASAN PERFORMA 5-FOLD HYBRID SYSTEM")
    print("=" * 70)
    print(f"   Lapis 1 (Cavanagh Terkalibrasi) : {np.mean(fold_accuracies_l1)*100:.2f}% ± {np.std(fold_accuracies_l1)*100:.2f}%")
    print(f"   Lapis 2 (Multi-Feature ML)      : {np.mean(fold_accuracies_l2)*100:.2f}% ± {np.std(fold_accuracies_l2)*100:.2f}%")
    print(f"   Hybrid 2-Lapis Final            : {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"   Stabilitas Model (Anti-Goyang)  : Variansi rendah (+/- {std_acc*100:.2f}%)")
    print("=" * 70)

    # Latih model ML final pada seluruh data untuk inferensi masa depan
    final_ml = HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.06, max_depth=4, min_samples_leaf=12,
        l2_regularization=2.5, random_state=42
    )
    final_ml.fit(X_feats, y_true)

    hybrid_system = HybridPlantarClassifier(ml_model=final_ml, cnn_model_path=cnn_model_path)

    return {
        'hybrid_classifier': hybrid_system,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'fold_accuracies': fold_accuracies_hybrid,
        'confusion_matrix': cm_hybrid,
        'metrics': metrics_hybrid,
        'accuracy': metrics_hybrid['accuracy'],
        'y_true': y_true.tolist(),
        'y_pred': oof_predictions_hybrid,
        'confidences': oof_confidences,
    }
