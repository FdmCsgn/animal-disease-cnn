from pathlib import Path
from PIL import Image
import shutil
from collections import defaultdict

SOURCE_DIR = Path(r"C:\Users\ardac\OneDrive\Masaüstü\YL_project\data\hastalıklar")
TARGET_DIR = Path("cleandataset")

IMG_EXT = [".jpg", ".jpeg", ".png", ".bmp"]
IMG_SIZE = (224, 224)

TARGET_DIR.mkdir(parents=True, exist_ok=True)

class_counts = defaultdict(int)
total_kept = 0
total_skipped = 0

for animal_dir in SOURCE_DIR.iterdir():
    if not animal_dir.is_dir():
        continue

    print(f"\n🐾 Hayvan: {animal_dir.name}")

    for disease_dir in animal_dir.iterdir():
        if not disease_dir.is_dir():
            continue

        print(f"   📁 Hastalık: {disease_dir.name}")

        target_class_dir = TARGET_DIR / animal_dir.name / disease_dir.name
        target_class_dir.mkdir(parents=True, exist_ok=True)

        kept = 0
        skipped = 0

        for img_path in disease_dir.iterdir():
            if img_path.suffix.lower() not in IMG_EXT:
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(IMG_SIZE)

                save_path = target_class_dir / img_path.name
                img.save(save_path)

                kept += 1
                total_kept += 1

            except Exception as e:
                print("      ❌ Okunamıyor:", img_path.name)
                skipped += 1
                total_skipped += 1

        class_counts[f"{animal_dir.name}/{disease_dir.name}"] += kept
        print(f"      ✅ Aktarılan: {kept}")
        print(f"      ❌ Atlanan: {skipped}")

print("\n===================================")
print("📊 TEMİZ DATASET SINIF SAYILARI")
print("===================================")

for cls, count in class_counts.items():
    print(f"{cls}: {count} görsel")

print("===================================")
print(f"🧮 Toplam temiz görsel: {total_kept}")
print(f"🗑️ Toplam atlanan görsel: {total_skipped}")
print("✅ Temiz dataset 'cleandataset' klasöründe.")