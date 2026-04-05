import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from transformers import ConvNextV2Model
from PIL import Image
import random

# ═══════════════════════════════════════════════════════════
# 1. MODEL TANIMI
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
# 2. GRAD-CAM
# ═══════════════════════════════════════════════════════════
class GradCAM:
    def __init__(self, model):
        self.model = model

    def generate(self, input_tensor, target_class=None):
        self.model.eval()

        input_tensor = input_tensor.unsqueeze(0).to(next(self.model.parameters()).device)
        input_tensor.requires_grad_(True)

        # Forward pass
        backbone_out = self.model.backbone(input_tensor)

        # last_hidden_state → (1, 1024, 7, 7)  channel-first
        last_hidden = backbone_out.last_hidden_state
        last_hidden.retain_grad()

        # Global average pool → (1, 1024)
        features = last_hidden.mean(dim=[-2, -1])
        logits   = self.model.classifier(features)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        confidence = logits.softmax(dim=1)[0, target_class].item()

        # Backward
        self.model.zero_grad()
        logits[0, target_class].backward(retain_graph=True)

        # Grad-CAM
        # last_hidden: (1, 1024, 7, 7)
        grad        = last_hidden.grad          # (1, 1024, 7, 7)
        activations = last_hidden               # (1, 1024, 7, 7)

        # Kanal boyutunda ağırlık → (1, 1024, 1, 1)
        weights = grad.mean(dim=[2, 3], keepdim=True)
        cam     = (activations * weights).sum(dim=1).squeeze(0)  # (7, 7)
        cam     = torch.relu(cam)

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam.cpu().detach().numpy(), target_class, confidence

# ═══════════════════════════════════════════════════════════
# 3. GÖRSELLEŞTİRME
# ═══════════════════════════════════════════════════════════
def visualize_gradcam(img_path, cam, true_label, pred_label, confidence, save_path):
    img    = Image.open(img_path).convert('RGB').resize((224, 224))
    img_np = np.array(img)

    # CAM → 224x224
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    ) / 255.0

    # Isı haritası
    heatmap = plt.cm.jet(cam_resized)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    # Overlay
    overlay = (0.5 * img_np + 0.5 * heatmap).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_np)
    axes[0].set_title('Orijinal Resim', fontsize=12)
    axes[0].axis('off')

    im = axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Isı Haritası', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    correct = true_label == pred_label
    color   = 'green' if correct else 'red'
    status  = '✓ DOĞRU' if correct else '✗ YANLIŞ'
    axes[2].set_title(
        f'Overlay\nGerçek  : {true_label}\nTahmin : {pred_label} ({confidence:.1%}) {status}',
        fontsize=9, color=color
    )
    axes[2].axis('off')

    plt.suptitle(f'{true_label}', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════
# 4. GRAD-CAM ÇALIŞTIR
# ═══════════════════════════════════════════════════════════
def run_gradcam(model, classes, root_dir, save_dir, num_samples=20, device='cpu'):
    os.makedirs(save_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Test setinden resimleri topla
    test_dir    = os.path.join(root_dir, 'test')
    extensions  = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    all_samples = []

    for folder in sorted(os.listdir(test_dir)):
        folder_path = os.path.join(test_dir, folder)
        if not os.path.isdir(folder_path) or not folder.endswith('_test'):
            continue
        class_name = folder.replace('_test', '')
        if class_name not in classes:
            continue
        class_idx = classes.index(class_name)
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith(extensions):
                all_samples.append((os.path.join(folder_path, img_file), class_idx, class_name))

    # Her sınıftan 1 örnek seç
    selected = {}
    random.shuffle(all_samples)
    for img_path, class_idx, class_name in all_samples:
        if class_name not in selected:
            selected[class_name] = (img_path, class_idx, class_name)
        if len(selected) >= num_samples:
            break

    samples = list(selected.values())
    print(f"📸 {len(samples)} resim için Grad-CAM uygulanıyor...\n")

    gradcam       = GradCAM(model)
    correct_count = 0

    for i, (img_path, true_idx, class_name) in enumerate(samples):
        try:
            img    = Image.open(img_path).convert('RGB')
            tensor = transform(img).to(device)

            cam, pred_idx, confidence = gradcam.generate(tensor)

            true_label = class_name
            pred_label = classes[pred_idx]

            if true_idx == pred_idx:
                correct_count += 1

            save_path = os.path.join(save_dir, f"gradcam_{i+1:02d}_{class_name[:30]}.png")
            visualize_gradcam(img_path, cam, true_label, pred_label, confidence, save_path)

            status = "✓" if true_idx == pred_idx else "✗"
            print(f"  {status} [{i+1:02d}/{len(samples)}] {class_name[:35]:<35} → {pred_label[:30]} ({confidence:.1%})")

        except Exception as e:
            print(f"  ⚠️  Hata [{i+1}] {class_name}: {e}")

    print(f"\n{'='*60}")
    print(f"✅ Grad-CAM tamamlandı!")
    print(f"   Doğru : {correct_count}/{len(samples)} (%{correct_count/len(samples)*100:.1f})")
    print(f"   Kayıt : {save_dir}/")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════
# 5. ÇALIŞTIR
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    CHECKPOINT  = "checkpoints/best_model.pth"
    ROOT_DIR    = "data_split"
    SAVE_DIR    = "gradcam_results"
    NUM_SAMPLES = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Cihaz: {device}")

    # Model yükle
    print(f"\n📦 Model yükleniyor: {CHECKPOINT}")
    checkpoint  = torch.load(CHECKPOINT, map_location=device)
    classes     = checkpoint['classes']
    num_classes = checkpoint['num_classes']

    model = ConvNextV2Classifier(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model yüklendi! ({num_classes} sınıf)\n")

    run_gradcam(model, classes, ROOT_DIR, SAVE_DIR,
                num_samples=NUM_SAMPLES, device=device)