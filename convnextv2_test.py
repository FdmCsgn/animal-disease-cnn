import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ConvNextV2Model
from collections import Counter
from PIL import Image
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)

# ═══════════════════════════════════════════════════════════
# 1. MODEL TANIMI (eğitimle aynı olmalı)
# ═══════════════════════════════════════════════════════════
class ConvNextV2Classifier(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.backbone   = ConvNextV2Model.from_pretrained("facebook/convnextv2-base-22k-224")
        hidden_size     = self.backbone.config.hidden_sizes[-1]
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        outputs  = self.backbone(x)
        features = outputs.last_hidden_state.mean(dim=[-2, -1])
        return self.classifier(features)


# ═══════════════════════════════════════════════════════════
# 2. DATASET
# ═══════════════════════════════════════════════════════════
class AnimalDiseaseDataset(Dataset):
    def __init__(self, root_dir, mode='test', transform=None):
        self.transform = transform
        self.samples   = []

        split_dir = os.path.join(root_dir, mode)

        if mode == 'train':
            suffixes = ('_train_augmented', '_train')
        elif mode == 'val':
            suffixes = ('_validation',)
        elif mode == 'test':
            suffixes = ('_test',)

        class_set = set()
        for folder in sorted(os.listdir(split_dir)):
            folder_path = os.path.join(split_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            for suffix in suffixes:
                if folder.endswith(suffix):
                    class_set.add(folder.replace(suffix, ''))
                    break

        self.classes      = sorted(list(class_set))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        for folder in sorted(os.listdir(split_dir)):
            folder_path = os.path.join(split_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            class_name = None
            for suffix in suffixes:
                if folder.endswith(suffix):
                    class_name = folder.replace(suffix, '')
                    break
            if class_name is None or class_name not in self.class_to_idx:
                continue
            class_idx = self.class_to_idx[class_name]
            for img_file in os.listdir(folder_path):
                if img_file.lower().endswith(extensions):
                    self.samples.append((os.path.join(folder_path, img_file), class_idx))

        print(f"  [{mode}] {len(self.classes)} sınıf | {len(self.samples)} resim")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            with open(img_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
        except:
            img = Image.new('RGB', (224, 224))
        if self.transform:
            img = self.transform(img)
        return img, label


# ═══════════════════════════════════════════════════════════
# 3. TAHMİN FONKSİYONU
# ═══════════════════════════════════════════════════════════
def get_predictions(model, loader, device):
    model.eval()
    all_preds   = []
    all_labels  = []
    all_probs   = []

    with torch.no_grad():
        for images, labels in loader:
            images  = images.to(device)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)
            preds   = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ═══════════════════════════════════════════════════════════
# 4. METRİK HESAPLAMA
# ═══════════════════════════════════════════════════════════
def calculate_metrics(labels, preds, probs, split_name, classes):
    print(f"\n{'='*60}")
    print(f"📊 {split_name.upper()} SETİ METRİKLERİ")
    print(f"{'='*60}")

    acc       = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall    = recall_score(labels, preds, average='weighted', zero_division=0)
    f1        = f1_score(labels, preds, average='weighted', zero_division=0)

    # AUC-ROC (multi-class için OvR)
    try:
        auc = roc_auc_score(labels, probs, multi_class='ovr', average='weighted')
    except Exception as e:
        auc = 0.0
        print(f"  ⚠️  AUC hesaplanamadı: {e}")

    print(f"  ✅ Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  ✅ Precision : {precision:.4f}")
    print(f"  ✅ Recall    : {recall:.4f}")
    print(f"  ✅ F1 Score  : {f1:.4f}")
    print(f"  ✅ AUC-ROC   : {auc:.4f}")

    return {"accuracy": acc, "precision": precision, "recall": recall,
            "f1": f1, "auc_roc": auc}


# ═══════════════════════════════════════════════════════════
# 5. CONFUSION MATRIX
# ═══════════════════════════════════════════════════════════
def plot_confusion_matrix(labels, preds, classes, split_name, save_dir):
    cm = confusion_matrix(labels, preds)

    # Uzun sınıf isimlerini kısalt
    short_classes = [c[:20] for c in classes]

    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=short_classes,
                yticklabels=short_classes,
                ax=ax)
    ax.set_xlabel('Tahmin Edilen', fontsize=12)
    ax.set_ylabel('Gerçek', fontsize=12)
    ax.set_title(f'Confusion Matrix - {split_name}', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"confusion_matrix_{split_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Confusion matrix kaydedildi: {save_path}")


# ═══════════════════════════════════════════════════════════
# 6. METRİK KARŞILAŞTIRMA GRAFİĞİ
# ═══════════════════════════════════════════════════════════
def plot_metrics_comparison(all_metrics, save_dir):
    splits  = list(all_metrics.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
    labels  = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']

    x     = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, split in enumerate(splits):
        values = [all_metrics[split][m] for m in metrics]
        ax.bar(x + i * width, values, width, label=split.upper())

    ax.set_xlabel('Metrik')
    ax.set_ylabel('Değer')
    ax.set_title('Train / Val / Test Metrik Karşılaştırması')
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.1)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "metrics_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Metrik karşılaştırma grafiği kaydedildi: {save_path}")


# ═══════════════════════════════════════════════════════════
# 7. ÇALIŞTIR
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    # ── Ayarlar ───────────────────────────────────────────
    CHECKPOINT  = "checkpoints_cutmix/cutmix_best.pth"
    ROOT_DIR    = "data_split"
    SAVE_DIR    = "evaluation_results"
    BATCH_SIZE  = 8
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Cihaz: {device}")

    # ── Checkpoint yükle ──────────────────────────────────
    print(f"\n📦 Model yükleniyor: {CHECKPOINT}")
    checkpoint  = torch.load(CHECKPOINT, map_location=device)
    classes     = checkpoint['classes']
    num_classes = checkpoint['num_classes']
    print(f"   Sınıf sayısı : {num_classes}")
    print(f"   En iyi Val Acc: {checkpoint['val_acc']:.4f}")

    model = ConvNextV2Classifier(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Model yüklendi!")

    # ── Transform ─────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ── Dataset & DataLoader ──────────────────────────────
    print("\n📂 Datasetler yükleniyor...")
    splits = {
        'train' : AnimalDiseaseDataset(ROOT_DIR, mode='train', transform=transform),
        'val'   : AnimalDiseaseDataset(ROOT_DIR, mode='val',   transform=transform),
        'test'  : AnimalDiseaseDataset(ROOT_DIR, mode='test',  transform=transform),
    }

    loaders = {
        split: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        for split, ds in splits.items()
    }

    # ── Tüm splitler için değerlendirme ───────────────────
    all_metrics = {}
    for split_name, loader in loaders.items():
        print(f"\n🔍 {split_name.upper()} değerlendiriliyor...")
        labels, preds, probs = get_predictions(model, loader, device)
        metrics = calculate_metrics(labels, preds, probs, split_name, classes)
        all_metrics[split_name] = metrics
        plot_confusion_matrix(labels, preds, classes, split_name, SAVE_DIR)

    # ── Karşılaştırma grafiği ─────────────────────────────
    print("\n📊 Karşılaştırma grafiği oluşturuluyor...")
    plot_metrics_comparison(all_metrics, SAVE_DIR)

    # ── Özet tablo ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"📋 ÖZET TABLO")
    print(f"{'='*60}")
    print(f"{'Metrik':<12} {'Train':>10} {'Val':>10} {'Test':>10}")
    print(f"{'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']:
        row = f"{metric:<12}"
        for split in ['train', 'val', 'test']:
            row += f" {all_metrics[split][metric]:>10.4f}"
        print(row)

    print(f"\n✅ Tüm sonuçlar '{SAVE_DIR}' klasörüne kaydedildi!")