import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from transformers import ConvNextV2Model
from collections import Counter
from PIL import Image

# ═══════════════════════════════════════════════════════════
# 1. DATASET
# ═══════════════════════════════════════════════════════════
class AnimalDiseaseDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.transform = transform
        self.samples   = []

        # mode → split klasörü ve suffix'ler
        split_dir = os.path.join(root_dir, mode)

        if mode == 'train':
            suffixes = ('_train_augmented', '_train')  # önce augmented, yoksa normal train
        elif mode == 'val':
            suffixes = ('_validation',)
        elif mode == 'test':
            suffixes = ('_test',)

        # Sınıf isimlerini suffix'i soyarak topla
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

        # Resimleri topla
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

        print(f"  [{mode}] {len(self.classes)} sınıf | {len(self.samples)} resim yüklendi")

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

    def get_class_weights(self):
        label_counts = Counter([s[1] for s in self.samples])
        weights = [1.0 / label_counts[label] for _, label in self.samples]
        return torch.tensor(weights, dtype=torch.float)


# ═══════════════════════════════════════════════════════════
# 2. TRANSFORM
# ═══════════════════════════════════════════════════════════
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ═══════════════════════════════════════════════════════════
# 3. DATALOADER
# ═══════════════════════════════════════════════════════════
def get_dataloaders(root_dir, batch_size=8):
    print("\n📂 Dataset yükleniyor...")
    train_dataset = AnimalDiseaseDataset(root_dir, mode='train', transform=train_transform)
    val_dataset   = AnimalDiseaseDataset(root_dir, mode='val',   transform=val_test_transform)
    test_dataset  = AnimalDiseaseDataset(root_dir, mode='test',  transform=val_test_transform)

    weights = train_dataset.get_class_weights()
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,   num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,   num_workers=2, pin_memory=True)

    print(f"✅ DataLoader hazır! Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, train_dataset.classes


# ═══════════════════════════════════════════════════════════
# 4. MİXUP & CUTMİX
# ═══════════════════════════════════════════════════════════
def mixup(images, labels, num_classes, alpha=0.4):
    lam           = np.random.beta(alpha, alpha)
    batch_size    = images.size(0)
    idx           = torch.randperm(batch_size)
    mixed_images  = lam * images + (1 - lam) * images[idx]
    labels_onehot = torch.zeros(batch_size, num_classes).to(images.device)
    labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
    mixed_labels  = lam * labels_onehot + (1 - lam) * labels_onehot[idx]
    return mixed_images, mixed_labels

def cutmix(images, labels, num_classes, alpha=1.0):
    lam                  = np.random.beta(alpha, alpha)
    batch_size, _, H, W  = images.size()
    idx                  = torch.randperm(batch_size)
    cut_ratio            = np.sqrt(1.0 - lam)
    cut_h, cut_w         = int(H * cut_ratio), int(W * cut_ratio)
    cx, cy               = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    mixed_images         = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
    lam                  = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    labels_onehot        = torch.zeros(batch_size, num_classes).to(images.device)
    labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
    mixed_labels         = lam * labels_onehot + (1 - lam) * labels_onehot[idx]
    return mixed_images, mixed_labels

def apply_mixup_or_cutmix(images, labels, num_classes, prob=0.5):
    if np.random.rand() < prob:
        return mixup(images, labels, num_classes)
    else:
        return cutmix(images, labels, num_classes)


# ═══════════════════════════════════════════════════════════
# 5. MODEL
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

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("🔒 Backbone donduruldu")

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("🔓 Backbone açıldı")

    def count_params(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"   Toplam: {total:,} | Eğitilebilir: {trainable:,}")


# ═══════════════════════════════════════════════════════════
# 6. EARLY STOPPING
# ═══════════════════════════════════════════════════════════
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience    = patience
        self.min_delta   = min_delta
        self.counter     = 0
        self.best_loss   = float('inf')
        self.should_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            print(f"   ⚠️  EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True


# ═══════════════════════════════════════════════════════════
# 7. LOSS
# ═══════════════════════════════════════════════════════════
def soft_cross_entropy(predictions, soft_labels):
    log_probs = torch.nn.functional.log_softmax(predictions, dim=-1)
    return -(soft_labels * log_probs).sum(dim=-1).mean()


# ═══════════════════════════════════════════════════════════
# 8. TRAIN & VALIDATE
# ═══════════════════════════════════════════════════════════
scaler = None

def train_one_epoch(model, loader, optimizer, device, num_classes, use_mixup=True):
    torch.cuda.empty_cache()
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        if use_mixup:
            images, soft_labels = apply_mixup_or_cutmix(images, labels, num_classes)
            soft_labels = soft_labels.to(device)

        optimizer.zero_grad()

        # autocast YOK, normal forward
        outputs = model(images)
        loss    = soft_cross_entropy(outputs, soft_labels) if use_mixup else nn.CrossEntropyLoss()(outputs, labels)

        if torch.isnan(loss):
            print(f"     ⚠️  NaN loss, batch atlanıyor!")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds       = outputs.argmax(dim=1)
        targets     = soft_labels.argmax(dim=1) if use_mixup else labels
        correct    += (preds == targets).sum().item()
        total      += labels.size(0)

        if (batch_idx + 1) % 100 == 0:
            print(f"     Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}")

    return total_loss / max(total, 1), correct / max(total, 1)

def validate(model, loader, device, num_classes):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images  = images.to(device)
            labels  = labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            total_loss += loss.item()
            correct    += (outputs.argmax(dim=1) == labels).sum().item()
            total      += labels.size(0)

    return total_loss / len(loader), correct / total


# ═══════════════════════════════════════════════════════════
# 9. ANA EĞİTİM
# ═══════════════════════════════════════════════════════════
def train(model, train_loader, val_loader, device, num_classes, classes):
    PHASE1_EPOCHS = 1
    PHASE2_EPOCHS = 2
    PHASE1_LR     = 1e-4
    PHASE2_LR     = 1e-5
    SAVE_DIR      = "checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)

    best_val_acc = 0.0
    history      = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # ── AŞAMA 1 ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"🔒 AŞAMA 1: FREEZE ({PHASE1_EPOCHS} epoch | LR: {PHASE1_LR})")
    print(f"{'='*60}")

    model.freeze_backbone()
    optimizer1  = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=PHASE1_LR)
    scheduler1  = CosineAnnealingLR(optimizer1, T_max=PHASE1_EPOCHS)
    early_stop1 = EarlyStopping(patience=5)

    for epoch in range(1, PHASE1_EPOCHS + 1):
        start = time.time()
        print(f"\n  📌 Epoch {epoch}/{PHASE1_EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer1, device, num_classes)
        val_loss,   val_acc   = validate(model, val_loader, device, num_classes)
        scheduler1.step()
        print(f"  ⏱️  {time.time()-start:.1f}s | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch'               : epoch,
                'phase'               : 1,
                'model_state_dict'    : model.state_dict(),
                'optimizer_state_dict': optimizer1.state_dict(),
                'val_acc'             : val_acc,
                'val_loss'            : val_loss,
                'classes'             : classes,
                'num_classes'         : num_classes,
            }, os.path.join(SAVE_DIR, "best_model.pth"))
            print(f"  💾 Kaydedildi! Val Acc: {val_acc:.4f}")

        early_stop1(val_loss)
        if early_stop1.should_stop:
            print("  🛑 Early stopping!")
            break

    # ── AŞAMA 2 ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"🔓 AŞAMA 2: UNFREEZE ({PHASE2_EPOCHS} epoch | LR: {PHASE2_LR})")
    print(f"{'='*60}")

    model.unfreeze_backbone()
    optimizer2  = optim.AdamW(model.parameters(), lr=PHASE2_LR, weight_decay=1e-4)
    scheduler2  = CosineAnnealingLR(optimizer2, T_max=PHASE2_EPOCHS)
    early_stop2 = EarlyStopping(patience=7)

    for epoch in range(1, PHASE2_EPOCHS + 1):
        start = time.time()
        print(f"\n  📌 Epoch {epoch}/{PHASE2_EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer2, device, num_classes)
        val_loss,   val_acc   = validate(model, val_loader, device, num_classes)
        scheduler2.step()
        print(f"  ⏱️  {time.time()-start:.1f}s | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch'               : epoch,
                'phase'               : 2,
                'model_state_dict'    : model.state_dict(),
                'optimizer_state_dict': optimizer2.state_dict(),
                'val_acc'             : val_acc,
                'val_loss'            : val_loss,
                'classes'             : classes,
                'num_classes'         : num_classes,
            }, os.path.join(SAVE_DIR, "best_model.pth"))
            print(f"  💾 Kaydedildi! Val Acc: {val_acc:.4f}")

        early_stop2(val_loss)
        if early_stop2.should_stop:
            print("  🛑 Early stopping!")
            break

    print(f"\n{'='*60}")
    print(f"✅ Eğitim tamamlandı! En iyi Val Acc: {best_val_acc:.4f}")
    print(f"   Model kaydedildi: checkpoints/best_model.pth")
    print(f"{'='*60}")
    return history


# ═══════════════════════════════════════════════════════════
# 10. ÇALIŞTIR
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.cuda.empty_cache()  # önce cache temizle
    torch.backends.cudnn.benchmark = True


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Cihaz: {device}")

    root_dir    = "data_split"
    batch_size  = 8
    
    train_loader, val_loader, test_loader, classes = get_dataloaders(root_dir, batch_size)
    NUM_CLASSES = len(classes)
    print(f"📊 Sınıf sayısı: {NUM_CLASSES}")

    model = ConvNextV2Classifier(num_classes=NUM_CLASSES, dropout=0.3).to(device)
    model.count_params()

    history = train(model, train_loader, val_loader, device, NUM_CLASSES, classes)