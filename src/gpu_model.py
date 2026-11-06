"""
=============================================================================
Module: gpu_model.py
Deep Learning CNN Classifier (FootNet) for Plantar Pressure Images
=============================================================================
Fitur Utama:
  1. Arsitektur FootNet: CNN Multi-Modal menggabungkan visual kontur spasial
     (3-Block ConvNet) dan fitur biomekanik klinis (Arch Index & Rasio Area A/B/C).
  2. Optimal Hardware:
     - GPU NVIDIA CUDA (RTX Series) otomatis aktif jika tersedia (FP16 AMP).
     - Otomatis fallback ke CPU multithreading jika CUDA tidak tersedia.
  3. Caching Cepat: Citra di-cache dalam memori sehingga 15 epoch selesai dalam hitungan detik.
  4. Penyimpanan Model: Model terlatih otomatis disimpan ke output/models/footnet_cnn.pt.
=============================================================================
"""

import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

CLASS_NAMES = ['High Arch', 'Normal', 'Flat Foot']
LABEL_TO_IDX = {'High Arch': 0, 'Normal': 1, 'Flat Foot': 2}
IDX_TO_LABEL = {0: 'High Arch', 1: 'Normal', 2: 'Flat Foot'}


class FootNet(nn.Module):
    """
    Arsitektur Convolutional Neural Network multi-modal ringan
    untuk citra distribusi tekanan plantar (64x64).
    """
    def __init__(self, num_classes: int = 3, num_clinical_features: int = 5):
        super(FootNet, self).__init__()
        
        # Ekstraktor Fitur Visual Spasial (CNN 3 Blok)
        self.features = nn.Sequential(
            # Blok 1: 64x64x3 -> 32x32x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Blok 2: 32x32x32 -> 16x16x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Blok 3: 16x16x64 -> 8x8x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Global Average Pooling -> 128
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Ekstraktor Fitur Biomekanik Klinis (Arch Index, Rasio Luas A, B, C)
        self.clinical_mlp = nn.Sequential(
            nn.Linear(num_clinical_features, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True)
        )
        
        # Klasifikasi Gabungan Multi-Modal
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 + 32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x_img, x_clinical):
        feat_img = self.features(x_img)
        feat_img = feat_img.view(feat_img.size(0), -1)
        
        feat_clin = self.clinical_mlp(x_clinical)
        
        fused = torch.cat((feat_img, feat_clin), dim=1)
        out = self.classifier(fused)
        return out


class CachedPlantarDataset(Dataset):
    """Dataset yang menyimpan gambar dan fitur dalam memori RAM untuk kecepatan maksimal."""
    def __init__(self, samples: list, img_size: int = 64):
        self.img_tensors = []
        self.clin_tensors = []
        self.targets = []
        self.filenames = []
        
        for item in samples:
            img_path = item.get('image_path') or item.get('info', {}).get('path', '')
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
                
            tensor_img = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0
            
            A = float(item.get('area_A', 0))
            B = float(item.get('area_B', 0))
            C = float(item.get('area_C', 0))
            tot = max(1.0, A + B + C)
            ai = float(item.get('arch_index', 0))
            
            clin = np.array([
                ai,
                A / tot,
                B / tot,
                C / tot,
                tot / 50000.0
            ], dtype=np.float32)
            tensor_clin = torch.from_numpy(clin)
            
            lbl_idx = LABEL_TO_IDX.get(item.get('ground_truth'), 1)
            
            self.img_tensors.append(tensor_img)
            self.clin_tensors.append(tensor_clin)
            self.targets.append(lbl_idx)
            self.filenames.append(item.get('filename', ''))
            
        self.img_tensors = torch.stack(self.img_tensors)
        self.clin_tensors = torch.stack(self.clin_tensors)
        self.targets = torch.tensor(self.targets, dtype=torch.long)
        
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, idx):
        return (self.img_tensors[idx],
                self.clin_tensors[idx],
                self.targets[idx],
                self.filenames[idx])


def get_optimal_device() -> torch.device:
    """Mendeteksi ketersediaan GPU NVIDIA CUDA."""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        print(f"[AKSELERASI GPU AKTIF] Device: {gpu_name} (VRAM: {vram_mb:.0f} MB)")
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        print("[MODE CPU AKTIF] Akselerasi CPU Multithreading.")
    return device


def train_and_evaluate_footnet(dataset_samples: list,
                               epochs: int = 15,
                               batch_size: int = 32,
                               lr: float = 0.001,
                               save_model_path: str = None) -> dict:
    """
    Melatih FootNet CNN dengan 15 Epochs.
    """
    device = get_optimal_device()
    use_cuda = (device.type == 'cuda')
    
    # Filter sampel valid
    valid_samples = [s for s in dataset_samples if s.get('arch_index', -1) >= 0]
    if not valid_samples:
        raise ValueError("Tidak ada data sampel valid untuk pelatihan FootNet.")
        
    labels = [LABEL_TO_IDX.get(s['ground_truth'], 1) for s in valid_samples]
    
    # 80% Train, 20% Test
    train_idx, test_idx = train_test_split(
        range(len(valid_samples)),
        test_size=0.20,
        stratify=labels,
        random_state=42
    )
    
    train_samples = [valid_samples[i] for i in train_idx]
    test_samples = [valid_samples[i] for i in test_idx]
    
    print("\n" + "=" * 68)
    print(f"   MEMUAT & CACHING DATASET ({len(valid_samples)} CITRA)...")
    print("=" * 68)
    t0_cache = time.time()
    train_dataset = CachedPlantarDataset(train_samples, img_size=64)
    test_dataset = CachedPlantarDataset(test_samples, img_size=64)
    print(f"[OK] Cache selesai dalam {time.time() - t0_cache:.2f} detik.")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = FootNet(num_classes=3, num_clinical_features=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)
    
    print("\n" + "=" * 68)
    print(f"   PELATIHAN DEEP LEARNING CNN (FOOTNET) - {epochs} EPOCHS")
    print(f"   Train Set: {len(train_samples)} citra | Test Set: {len(test_samples)} citra")
    print("=" * 68)
    
    history = {'train_loss': [], 'train_acc': []}
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for imgs, clins, targets, _ in train_loader:
            imgs = imgs.to(device)
            clins = clins.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=use_cuda):
                outputs = model(imgs, clins)
                loss = criterion(outputs, targets)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * targets.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)
            
        scheduler.step()
        train_acc = train_correct / train_total
        avg_loss = train_loss / train_total
        
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        
        elapsed = time.time() - start_time
        print(f"   Epoch [{epoch:>2}/{epochs}] | Loss: {avg_loss:.4f} | "
              f"Train Accuracy: {train_acc*100:>5.2f}% | Durasi: {elapsed:>4.1f}s")
        
    train_duration = time.time() - start_time
    print(f"\n[OK] Pelatihan 15 Epochs Selesai dalam {train_duration:.2f} detik!")
    
    # Evaluasi pada Data Uji (Test Set)
    model.eval()
    all_preds = []
    all_targets = []
    results_detail = []
    
    with torch.no_grad():
        for imgs, clins, targets, fnames in test_loader:
            imgs = imgs.to(device)
            clins = clins.to(device)
            
            with torch.amp.autocast('cuda', enabled=use_cuda):
                outputs = model(imgs, clins)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1).cpu().numpy()
                
            targets_np = targets.numpy()
            probs_np = probs.cpu().numpy()
            
            for p, t, prob, fn in zip(preds, targets_np, probs_np, fnames):
                pred_label = IDX_TO_LABEL[p]
                true_label = IDX_TO_LABEL[t]
                all_preds.append(pred_label)
                all_targets.append(true_label)
                results_detail.append({
                    'filename': fn,
                    'ground_truth': true_label,
                    'prediction': pred_label,
                    'confidence': float(prob[p]),
                    'probs': {IDX_TO_LABEL[i]: float(prob[i]) for i in range(3)}
                })
                
    # Hitung Confusion Matrix & Metrik
    cm = confusion_matrix(all_targets, all_preds, labels=CLASS_NAMES)
    
    # Hitung metrik evaluasi standar
    from evaluation import compute_metrics
    metrics = compute_metrics(cm, CLASS_NAMES)
    
    # Simpan model jika path ditentukan
    if save_model_path:
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'class_names': CLASS_NAMES,
            'label_to_idx': LABEL_TO_IDX,
            'epochs': epochs,
            'accuracy': metrics['accuracy'],
        }, save_model_path)
        print(f"-> Model CNN tersimpan di: {save_model_path}")
        
    if use_cuda:
        torch.cuda.empty_cache()
        
    return {
        'model': model,
        'epochs': epochs,
        'history': history,
        'accuracy': metrics['accuracy'],
        'confusion_matrix': cm,
        'metrics': metrics,
        'y_true': all_targets,
        'y_pred': all_preds,
        'results_detail': results_detail,
        'train_duration': train_duration,
        'device_name': torch.cuda.get_device_name(0) if use_cuda else 'CPU',
    }


def predict_single_image_cnn(model, img_bgr: np.ndarray, regions: dict,
                             arch_index: float, device: torch.device = None) -> dict:
    """
    Prediksi kelas telapak kaki menggunakan model CNN FootNet terlatih.
    """
    if device is None:
        device = next(model.parameters()).device
        
    model.eval()
    
    # Preprocessing citra (64x64)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, (64, 64), interpolation=cv2.INTER_AREA)
    tensor_img = torch.from_numpy(img_res.transpose((2, 0, 1))).float().unsqueeze(0) / 255.0
    tensor_img = tensor_img.to(device)
    
    A = float(regions.get('A_area', 0))
    B = float(regions.get('B_area', 0))
    C = float(regions.get('C_area', 0))
    tot = max(1.0, A + B + C)
    
    clin = np.array([[
        float(arch_index),
        A / tot,
        B / tot,
        C / tot,
        tot / 50000.0
    ]], dtype=np.float32)
    tensor_clin = torch.from_numpy(clin).to(device)
    
    with torch.no_grad():
        out = model(tensor_img, tensor_clin)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        
    pred_label = IDX_TO_LABEL[pred_idx]
    confidence = float(probs[pred_idx])
    
    return {
        'prediction': pred_label,
        'confidence': confidence,
        'probs': {IDX_TO_LABEL[i]: float(probs[i]) for i in range(3)}
    }
