import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from transformers import CvtForImageClassification, AutoFeatureExtractor
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR        = r"C:\Users\ardac\OneDrive\Masaüstü\YL_project\data_split"
OUTPUT_DIR      = r"C:\Users\ardac\OneDrive\Masaüstü\YL_project\outputs_cvt21"
IMG_SIZE        = 224       # CvT-21 için standart boyut
BATCH_SIZE      = 8         # RTX 3050 4GB için güvenli değer
ACCUM_STEPS     = 4         # Efektif batch = 8x4 = 32
NUM_WORKERS     = 0         # Windows için 0
PIN_MEMORY      = True
USE_AMP         = True      # Mixed precision — VRAM tasarrufu
 
FREEZE_EPOCHS   = 5
FINETUNE_EPOCHS = 5
 
FREEZE_LR       = 1e-3
FINETUNE_LR     = 1e-5
 
AUGMENTATION_MODE = "both"  # "both" | "mixup" | "cutmix" | "none"
MIXUP_ALPHA     = 0.4
CUTMIX_ALPHA    = 1.0
MIXUP_PROB      = 0.5
 
SEED            = 42
SUBSET_RATIO    = 0.4
# ─────────────────────────────────────────────
# SEED & DEVICE
# ─────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 
set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Kullanılan cihaz: {device}")
if device.type == "cuda":
    print(f"   GPU : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
 
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ─────────────────────────────────────────────
# TRANSFORM
# CvT-21 ImageNet normalizasyonu kullanır
# ─────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
 
val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
# ─────────────────────────────────────────────
# SAFE DATASET
# ─────────────────────────────────────────────
from torchvision.datasets import ImageFolder
 
class SafeImageFolder(ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"   🔍 Dosyalar doğrulanıyor: {self.root}")
        valid_samples = []
        skipped = 0
        for path, label in tqdm(self.samples, desc="   Taranıyor", leave=False):
            try:
                with open(path, 'rb') as f:
                    f.read(10)
                valid_samples.append((path, label))
            except (FileNotFoundError, OSError):
                skipped += 1
        if skipped > 0:
            print(f"   ⚠️  {skipped} dosya atlandı")
        self.samples = valid_samples
        self.targets = [s[1] for s in valid_samples]
        print(f"   ✅ Geçerli dosya sayısı: {len(self.samples)}")
# ─────────────────────────────────────────────
# DATASET & DATALOADER
# ─────────────────────────────────────────────
print("\n📂 Veri seti yükleniyor...")
train_dataset = SafeImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_dataset   = SafeImageFolder(os.path.join(DATA_DIR, "val"),   transform=val_test_transform)
test_dataset  = SafeImageFolder(os.path.join(DATA_DIR, "test"),  transform=val_test_transform)
 
NUM_CLASSES = len(train_dataset.classes)
print(f"   Sınıf sayısı : {NUM_CLASSES}")
print(f"   Train        : {len(train_dataset)} görsel")
print(f"   Validation   : {len(val_dataset)} görsel")
print(f"   Test         : {len(test_dataset)} görsel")
 
def get_weighted_sampler(dataset):
    targets = [s[1] for s in dataset.samples]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[t] for t in targets]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
 
sampler = get_weighted_sampler(train_dataset)
subset_size = int(len(train_dataset) * SUBSET_RATIO)
subset_sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor([sampler.weights[i] for i in range(len(train_dataset))]),
    num_samples=subset_size,
    replacement=True
)
 
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=subset_sampler,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
 
print(f"   Her epoch: {subset_size} görsel / {subset_size//BATCH_SIZE} batch")

# ─────────────────────────────────────────────
# MixUp / CutMix
# ─────────────────────────────────────────────
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    idx = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam
 
def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0)).to(x.device)
    _, _, H, W = x.size()
    cut_rat = np.sqrt(1 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return mixed_x, y, y[idx], lam
 
def mixed_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ─────────────────────────────────────────────
# MODEL — CvT-21 (HuggingFace transformers)
#
# CvT-21 mimarisi:
#   model.cvt          → backbone (3 stage: stage 0/1/2)
#     .stage_0 / .stage_1 / .stage_2  → convolutional transformer blokları
#   model.layernorm    → son normalizasyon
#   model.classifier   → Linear(384, num_labels)  ← değiştirdiğimiz yer
# ─────────────────────────────────────────────
from transformers import CvtForImageClassification
import huggingface_hub
print("\n🔧 CvT-21 modeli yükleniyor (HuggingFace)...")

model = CvtForImageClassification.from_pretrained(
    'microsoft/cvt-21',
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True,
    use_safetensors=True,
    torch_dtype=torch.float32   # ← bunu ekle
)
model = model.to(device)

# ─────────────────────────────────────────────
# FREEZE / UNFREEZE
#
# CvT-21 katman yapısı:
#   model.cvt.stage_0  → ilk stage (en düşük seviye özellikler)
#   model.cvt.stage_1  → orta stage
#   model.cvt.stage_2  → son stage (en yüksek seviye özellikler)
#   model.layernorm    → normalizasyon
#   model.classifier   → sınıflandırıcı başlığı
# ─────────────────────────────────────────────
def freeze_backbone(model):
    """Sadece classifier'ı eğitilebilir bırak, cvt backbone'u dondur"""
    for param in model.cvt.parameters():
        param.requires_grad = False
    for param in model.layernorm.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
 
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"   ❄️  Backbone donduruldu.")
    print(f"       Eğitilebilir: {trainable:,} / Toplam: {total:,} parametre")
 
def unfreeze_all(model):
    """Tüm parametreleri aç"""
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   🔥 Tüm model açıldı. Eğitilebilir parametre: {trainable:,}")

# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────
def evaluate(model, loader, criterion, phase="val"):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    torch.cuda.empty_cache()
 
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type="cuda", enabled=USE_AMP):
                # HuggingFace modeli pixel_values ile çalışır
                outputs = model(pixel_values=images)
                logits  = outputs.logits
                loss    = criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            del images, labels, outputs, logits, loss
 
    torch.cuda.empty_cache()
    avg_loss = total_loss / len(loader)
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec  = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1   = f1_score(all_labels, all_preds, average='macro', zero_division=0)
 
    return avg_loss, acc, prec, rec, f1, all_preds, all_labels
 
 # ─────────────────────────────────────────────
# CONFUSION MATRIX KAYDET
# ─────────────────────────────────────────────
def save_confusion_matrix(labels, preds, class_names, save_path, title="Confusion Matrix"):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(28, 24))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, annot_kws={"size": 6})
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Tahmin", fontsize=10)
    ax.set_ylabel("Gerçek", fontsize=10)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   💾 Confusion matrix: {save_path}")

# ─────────────────────────────────────────────
# METRİK GRAFİĞİ
# ─────────────────────────────────────────────
def save_metrics_plot(history, save_path, title="Eğitim Grafiği"):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=14)
    metrics = [
        ('train_loss', 'val_loss', 'Loss'),
        ('train_acc',  'val_acc',  'Accuracy'),
        ('train_f1',   'val_f1',   'F1 Score'),
        ('train_prec', 'val_prec', 'Precision'),
        ('train_rec',  'val_rec',  'Recall'),
    ]
    for i, (tr_key, val_key, label) in enumerate(metrics):
        ax = axes[i // 3][i % 3]
        ax.plot(epochs, history[tr_key], 'b-o', label='Train')
        ax.plot(epochs, history[val_key], 'r-o', label='Val')
        ax.set_title(label); ax.set_xlabel('Epoch')
        ax.legend(); ax.grid(True)
    axes[1][2].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"   💾 Grafik: {save_path}")

# ─────────────────────────────────────────────
# EĞİTİM FONKSİYONU
# ─────────────────────────────────────────────
def train_phase(model, train_loader, val_loader, optimizer, scheduler,
                criterion, num_epochs, phase_name, aug_mode, class_names):
 
    history = {k: [] for k in ['train_loss','val_loss','train_acc','val_acc',
                                'train_f1','val_f1','train_prec','val_prec',
                                'train_rec','val_rec']}
    best_val_f1    = 0.0
    best_model_path = os.path.join(OUTPUT_DIR, f"best_{phase_name}.pth")
 
    cm_dir = os.path.join(OUTPUT_DIR, f"confusion_matrices_{phase_name}")
    os.makedirs(cm_dir, exist_ok=True)
 
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
 
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []
        start = time.time()
        torch.cuda.empty_cache()
 
        skipped_batches = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"[{phase_name}] Epoch {epoch}/{num_epochs}")
 
        for batch_idx, batch_data in enumerate(pbar):
            try:
                images, labels = batch_data
                images, labels = images.to(device), labels.to(device)
 
                use_aug = aug_mode != "none"
                y_a, y_b, lam = labels, labels, 1.0
                if use_aug:
                    use_mixup = (random.random() < MIXUP_PROB) if aug_mode == "both" else (aug_mode == "mixup")
                    if use_mixup:
                        images, y_a, y_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
                    else:
                        images, y_a, y_b, lam = cutmix_data(images, labels, CUTMIX_ALPHA)
 
                with torch.autocast(device_type="cuda", enabled=USE_AMP):
                    # HuggingFace: pixel_values kwarg kullan
                    outputs = model(pixel_values=images)
                    logits  = outputs.logits
                    if use_aug and aug_mode != "none":
                        loss = mixed_criterion(criterion, logits, y_a, y_b, lam)
                    else:
                        loss = criterion(logits, labels)
                    loss = loss / ACCUM_STEPS
 
                scaler.scale(loss).backward()
 
                if (batch_idx + 1) % ACCUM_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
 
                total_loss += loss.item() * ACCUM_STEPS
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                pbar.set_postfix({'loss': f'{loss.item() * ACCUM_STEPS:.4f}'})
                del images, labels, outputs, logits, loss
 
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n   ⚠️  OOM! Batch {batch_idx} atlanıyor...")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    skipped_batches += 1
                    continue
                else:
                    raise e
            except Exception:
                skipped_batches += 1
                continue
 
        if skipped_batches > 0:
            print(f"   {skipped_batches} batch atlandı")
 
        if scheduler:
            scheduler.step()
 
        torch.cuda.empty_cache()
 
        tr_loss = total_loss / len(train_loader)
        tr_acc  = accuracy_score(all_labels, all_preds)
        tr_f1   = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        tr_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        tr_rec  = recall_score(all_labels, all_preds, average='macro', zero_division=0)
 
        val_loss, val_acc, val_prec, val_rec, val_f1, val_preds, val_labels = evaluate(
            model, val_loader, criterion
        )
 
        elapsed = time.time() - start
        print(f"\n📊 [{phase_name}] Epoch {epoch}/{num_epochs} ({elapsed:.0f}s)")
        print(f"   Train → Loss:{tr_loss:.4f}  Acc:{tr_acc:.4f}  F1:{tr_f1:.4f}  Prec:{tr_prec:.4f}  Rec:{tr_rec:.4f}")
        print(f"   Val   → Loss:{val_loss:.4f}  Acc:{val_acc:.4f}  F1:{val_f1:.4f}  Prec:{val_prec:.4f}  Rec:{val_rec:.4f}")
 
        history['train_loss'].append(tr_loss);  history['val_loss'].append(val_loss)
        history['train_acc'].append(tr_acc);    history['val_acc'].append(val_acc)
        history['train_f1'].append(tr_f1);      history['val_f1'].append(val_f1)
        history['train_prec'].append(tr_prec);  history['val_prec'].append(val_prec)
        history['train_rec'].append(tr_rec);    history['val_rec'].append(val_rec)
 
        # Train CM
        save_confusion_matrix(
            all_labels, all_preds, class_names,
            os.path.join(cm_dir, f"epoch{epoch:02d}_train_cm.png"),
            title=f"[{phase_name}] Train CM — Epoch {epoch}/{num_epochs} | F1:{tr_f1:.4f}"
        )
        # Val CM
        save_confusion_matrix(
            val_labels, val_preds, class_names,
            os.path.join(cm_dir, f"epoch{epoch:02d}_val_cm.png"),
            title=f"[{phase_name}] Val CM — Epoch {epoch}/{num_epochs} | F1:{val_f1:.4f}"
        )
 
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"   ✅ En iyi model kaydedildi (Val F1: {best_val_f1:.4f})")
 
    return history, best_model_path

# ─────────────────────────────────────────────
# TEST DEĞERLENDİRME
# ─────────────────────────────────────────────
def test_evaluation(model, test_loader, criterion, class_names, phase_label, aug_label):
    print(f"\n{'='*60}")
    print(f"🧪 TEST DEĞERLENDİRMESİ - {phase_label} | Aug: {aug_label}")
    print(f"{'='*60}")
    _, acc, prec, rec, f1, preds, labels = evaluate(model, test_loader, criterion, "test")
    print(f"   Accuracy  : {acc:.4f}")
    print(f"   Precision : {prec:.4f}")
    print(f"   Recall    : {rec:.4f}")
    print(f"   F1 Score  : {f1:.4f}")
 
    cm_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{phase_label}_{aug_label}.png")
    save_confusion_matrix(labels, preds, class_names, cm_path,
                          title=f"Test CM - {phase_label} | {aug_label}")
 
    report = classification_report(labels, preds, target_names=class_names, zero_division=0)
    report_path = os.path.join(OUTPUT_DIR, f"classification_report_{phase_label}_{aug_label}.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Phase: {phase_label} | Augmentation: {aug_label}\n\n")
        f.write(report)
    print(f"   💾 Classification report: {report_path}")
 
    return acc, prec, rec, f1

# ─────────────────────────────────────────────
# GRAD-CAM
#
# CvT-21 Transformer tabanlı bir model olduğu için
# pytorch_grad_cam'ın standart GradCAM'i yerine
# GradCAMPlusPlus + reshape_transform kullanıyoruz.
#
# CvT son stage çıktısı (stage_2) shape:
#   (batch, H*W, embed_dim) → reshape ile (batch, embed_dim, H, W)
# ─────────────────────────────────────────────
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
 
GRADCAM_DIR      = os.path.join(OUTPUT_DIR, "gradcam")
SAMPLES_PER_CLASS = 2
os.makedirs(GRADCAM_DIR, exist_ok=True)
 
vis_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])
norm_transform = transforms.Normalize([0.485, 0.456, 0.406],
                                       [0.229, 0.224, 0.225])
 
def cvt_reshape_transform(tensor, height=7, width=7):
    """
    CvT son katman çıktısı: (batch, height*width, channels)
    GradCAM için: (batch, channels, height, width) gerekli
    """
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result
 
def get_target_layer_cvt(model):
    """
    CvT-21 için Grad-CAM hedef katmanı:
    model.cvt.encoder.stages[-1] → son transformer stage
    """
    return [model.cvt.encoder.stages[-1]]
 
def run_gradcam(model, model_name, dataset, class_names):
    model.eval()
    target_layers = get_target_layer_cvt(model)
    cam = GradCAMPlusPlus(
        model=model,
        target_layers=target_layers,
        reshape_transform=cvt_reshape_transform
    )
 
    save_dir = os.path.join(GRADCAM_DIR, model_name)
    os.makedirs(save_dir, exist_ok=True)
 
    class_samples = {i: [] for i in range(len(class_names))}
    for path, label in dataset.samples:
        if len(class_samples[label]) < SAMPLES_PER_CLASS:
            class_samples[label].append(path)
        if all(len(v) >= SAMPLES_PER_CLASS for v in class_samples.values()):
            break
 
    print(f"\n🎨 [{model_name}] Grad-CAM oluşturuluyor...")
    for class_idx, paths in tqdm(class_samples.items(), desc=model_name):
        class_name = class_names[class_idx]
        fig, axes = plt.subplots(len(paths), 3, figsize=(12, 4 * len(paths)))
        if len(paths) == 1:
            axes = [axes]
        fig.suptitle(f"{model_name} | Sınıf: {class_name}", fontsize=11)
 
        for row, path in enumerate(paths):
            try:
                from PIL import Image as PILImage
                pil_img = PILImage.open(path).convert("RGB")
                pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE))
                rgb_img = np.array(pil_img).astype(np.float32) / 255.0
 
                input_tensor = norm_transform(vis_transform(pil_img)).unsqueeze(0).to(device)
 
                with torch.no_grad():
                    output = model(pixel_values=input_tensor)
                pred_idx   = output.logits.argmax(dim=1).item()
                pred_name  = class_names[pred_idx]
                confidence = torch.softmax(output.logits, dim=1)[0][pred_idx].item()
 
                targets = [ClassifierOutputTarget(class_idx)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                grayscale_cam = grayscale_cam[0]
                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
 
                is_correct   = (pred_idx == class_idx)
                border_color = "green" if is_correct else "red"
                status       = "✅ Doğru" if is_correct else f"❌ Yanlış → {pred_name}"
 
                axes[row][0].imshow(pil_img)
                axes[row][0].set_title("Orijinal", fontsize=9)
                axes[row][0].axis("off")
 
                axes[row][1].imshow(grayscale_cam, cmap="jet")
                axes[row][1].set_title("Grad-CAM++ Isı Haritası", fontsize=9)
                axes[row][1].axis("off")
 
                axes[row][2].imshow(cam_image)
                axes[row][2].set_title(f"{status}\nGüven: {confidence:.2%}",
                                       fontsize=9, color=border_color)
                axes[row][2].axis("off")
                for spine in axes[row][2].spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(3)
 
            except Exception as e:
                print(f"   ⚠️  {path} atlandı: {e}")
                continue
 
        plt.tight_layout()
        safe_name = class_name.replace("/", "_").replace("\\", "_")
        plt.savefig(os.path.join(save_dir, f"{safe_name}.png"), dpi=100, bbox_inches="tight")
        plt.close()
 
    print(f"   💾 Kaydedildi: {save_dir}")

# ─────────────────────────────────────────────
# ANA AKIŞ
# ─────────────────────────────────────────────
criterion   = nn.CrossEntropyLoss()
class_names = train_dataset.classes
 
print(f"\n{'='*60}")
print(f"🚀 EĞİTİM BAŞLIYOR")
print(f"   Model          : CvT-21 (microsoft/cvt-21)")
print(f"   Görsel boyutu  : {IMG_SIZE}x{IMG_SIZE}")
print(f"   Augmentation   : {AUGMENTATION_MODE}")
print(f"   Freeze Epochs  : {FREEZE_EPOCHS}")
print(f"   Finetune Epochs: {FINETUNE_EPOCHS}")
print(f"   Batch Size     : {BATCH_SIZE}  (efektif: {BATCH_SIZE*ACCUM_STEPS})")
print(f"   Mixed Precision: {USE_AMP}")
print(f"{'='*60}\n")

# ──────────────────────────────
# AŞAMA 1: BACKBONE FREEZE
# ──────────────────────────────
print("\n" + "="*60)
print("⚡ AŞAMA 1: BACKBONE FREEZE")
print("="*60)
freeze_backbone(model)
 
optimizer_freeze = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=FREEZE_LR, weight_decay=1e-4
)
scheduler_freeze = optim.lr_scheduler.CosineAnnealingLR(optimizer_freeze, T_max=FREEZE_EPOCHS)
 
history_freeze, best_freeze_path = train_phase(
    model, train_loader, val_loader,
    optimizer_freeze, scheduler_freeze,
    criterion, FREEZE_EPOCHS,
    phase_name="freeze",
    aug_mode=AUGMENTATION_MODE,
    class_names=class_names
)
 
save_metrics_plot(history_freeze,
                  os.path.join(OUTPUT_DIR, "grafik_freeze.png"),
                  title=f"Aşama 1: Backbone Freeze | Aug: {AUGMENTATION_MODE}")
 
print("\n📌 Aşama 1 tamamlandı → Test seti üzerinde değerlendiriliyor...")
model.load_state_dict(torch.load(best_freeze_path, map_location=device))
acc1, prec1, rec1, f1_1 = test_evaluation(
    model, test_loader, criterion, class_names,
    phase_label="Phase1_Freeze", aug_label=AUGMENTATION_MODE
)

# ──────────────────────────────
# AŞAMA 2: FINE-TUNING
# ──────────────────────────────
print("\n" + "="*60)
print("🔥 AŞAMA 2: FINE-TUNING (Backbone açılıyor)")
print("="*60)
unfreeze_all(model)
 
optimizer_finetune = optim.AdamW(
    model.parameters(),
    lr=FINETUNE_LR, weight_decay=1e-4
)
scheduler_finetune = optim.lr_scheduler.CosineAnnealingLR(optimizer_finetune, T_max=FINETUNE_EPOCHS)
 
history_finetune, best_finetune_path = train_phase(
    model, train_loader, val_loader,
    optimizer_finetune, scheduler_finetune,
    criterion, FINETUNE_EPOCHS,
    phase_name="finetune",
    aug_mode=AUGMENTATION_MODE,
    class_names=class_names
)
 
save_metrics_plot(history_finetune,
                  os.path.join(OUTPUT_DIR, "grafik_finetune.png"),
                  title=f"Aşama 2: Fine-Tuning | Aug: {AUGMENTATION_MODE}")
 
print("\n📌 Aşama 2 tamamlandı → Test seti üzerinde değerlendiriliyor...")
model.load_state_dict(torch.load(best_finetune_path, map_location=device))
acc2, prec2, rec2, f1_2 = test_evaluation(
    model, test_loader, criterion, class_names,
    phase_label="Phase2_FineTune", aug_label=AUGMENTATION_MODE
)

# Tüm eski hook'ları temizle
for stage in model.cvt.encoder.stages:
    stage._forward_hooks.clear()

# Stage 2 çıktısını yakala
captured = {}

def hook_fn(module, input, output):
    if isinstance(output, tuple):
        captured['output'] = output[0]  # ilk eleman
        print(f"tuple içi [0] shape: {output[0].shape}")
        print(f"tuple eleman sayısı: {len(output)}")
        for i, o in enumerate(output):
            if hasattr(o, 'shape'):
                print(f"  [{i}]: {o.shape}")
    else:
        captured['output'] = output
        print(f"output shape: {output.shape}")

hook = model.cvt.encoder.stages[-1].register_forward_hook(hook_fn)

dummy = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    model(pixel_values=dummy)

hook.remove()
print(f"\n✅ Yakalanan shape: {captured['output'].shape}")
# Tüm hook'ları temizle
for stage in model.cvt.encoder.stages:
    stage._forward_hooks.clear()

# Wrapper'ları tanımla
class CvTWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(pixel_values=x).logits

class StageOutputWrapper(torch.nn.Module):
    def __init__(self, stage):
        super().__init__()
        self.stage = stage
    def forward(self, x):
        out = self.stage(x)
        return out[0] if isinstance(out, tuple) else out

# Test
test_path, test_label = gradcam_dataset.samples[0]
from PIL import Image as PILImage
import numpy as np

pil_img = PILImage.open(test_path).convert("RGB").resize((224, 224))
vis_t   = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
norm_t  = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
input_tensor = norm_t(vis_t(pil_img)).unsqueeze(0).to(device)

model.eval()
wrapped_model = CvTWrapper(model)
wrapped_stage = StageOutputWrapper(model.cvt.encoder.stages[-1])

# Wrapped model çıktısını kontrol et
print("1. Wrapped model çıktısı:")
with torch.no_grad():
    out = wrapped_model(input_tensor)
print(f"   shape: {out.shape}, type: {type(out)}")

# Wrapped stage çıktısını kontrol et
print("\n2. Wrapped stage çıktısı:")
hook_out = {}
def h(m, i, o):
    hook_out['val'] = o
    print(f"   shape: {o.shape}, type: {type(o)}")
hk = wrapped_stage.stage.register_forward_hook(h)
with torch.no_grad():
    wrapped_model(input_tensor)
hk.remove()

# GradCAM manuel adım adım
print("\n3. GradCAM gradyan testi:")
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

cam = GradCAMPlusPlus(
    model=wrapped_model,
    target_layers=[wrapped_stage],
    reshape_transform=None
)

try:
    result = cam(input_tensor=input_tensor, 
                 targets=[ClassifierOutputTarget(test_label)])
    print(f"   shape: {result.shape}")
    print(f"   min: {result.min():.6f}")
    print(f"   max: {result.max():.6f}")
    print(f"   sıfır mı: {(result==0).all()}")
    print(f"   NaN mı: {np.isnan(result).any()}")
except Exception as e:
    print(f"   HATA: {e}")
    import traceback
    traceback.print_exc()

# ──────────────────────────────
# KARŞILAŞTIRMA TABLOSU
# (kernel restart sonrası kullan)
# ──────────────────────────────

best_freeze_path   = os.path.join(OUTPUT_DIR, "best_freeze.pth")
best_finetune_path = os.path.join(OUTPUT_DIR, "best_finetune.pth")
criterion = nn.CrossEntropyLoss()

# Phase 1 skorları
print("⚡ Phase 1 modeli değerlendiriliyor...")
model.load_state_dict(torch.load(best_freeze_path, map_location=device))
acc1, prec1, rec1, f1_1 = test_evaluation(
    model, test_loader, criterion, class_names,
    phase_label="Phase1_Freeze", aug_label=AUGMENTATION_MODE
)

torch.cuda.empty_cache()

# Phase 2 skorları
print("🔥 Phase 2 modeli değerlendiriliyor...")
model.load_state_dict(torch.load(best_finetune_path, map_location=device))
acc2, prec2, rec2, f1_2 = test_evaluation(
    model, test_loader, criterion, class_names,
    phase_label="Phase2_FineTune", aug_label=AUGMENTATION_MODE
)

torch.cuda.empty_cache()

# Tablo
print("\n" + "="*60)
print("📈 SONUÇ KARŞILAŞTIRMASI")
print("="*60)
print(f"{'Aşama':<30} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
print("-"*60)
print(f"{'Phase 1 (Freeze)':<30} {acc1:>8.4f} {prec1:>8.4f} {rec1:>8.4f} {f1_1:>8.4f}")
print(f"{'Phase 2 (Fine-Tune)':<30} {acc2:>8.4f} {prec2:>8.4f} {rec2:>8.4f} {f1_2:>8.4f}")
print("="*60)

# Özet dosyası
summary_path = os.path.join(OUTPUT_DIR, f"summary_{AUGMENTATION_MODE}.txt")
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(f"CvT-21 Eğitim Özeti\n")
    f.write(f"Augmentation: {AUGMENTATION_MODE}\n")
    f.write(f"Görsel boyutu: {IMG_SIZE}x{IMG_SIZE}\n")
    f.write(f"{'='*60}\n")
    f.write(f"{'Aşama':<30} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}\n")
    f.write(f"{'-'*60}\n")
    f.write(f"{'Phase 1 (Freeze)':<30} {acc1:>8.4f} {prec1:>8.4f} {rec1:>8.4f} {f1_1:>8.4f}\n")
    f.write(f"{'Phase 2 (Fine-Tune)':<30} {acc2:>8.4f} {prec2:>8.4f} {rec2:>8.4f} {f1_2:>8.4f}\n")

print(f"\n✅ Tüm sonuçlar kaydedildi: {OUTPUT_DIR}")
print(f"   Özet: {summary_path}")
print("\n🎉 EĞİTİM TAMAMLANDI!")