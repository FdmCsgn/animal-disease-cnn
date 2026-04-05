import os

dataset_path = r"C:\Users\ardac\OneDrive\Masaüstü\YL_project\data_split"

image_ext = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

for split in ["train", "val", "test"]:

    split_path = os.path.join(dataset_path, split)

    print(f"\n📂 {split} klasörü")

    for class_name in os.listdir(split_path):

        class_path = os.path.join(split_path, class_name)

        if not os.path.isdir(class_path):
            continue

        files = [f for f in os.listdir(class_path) if f.lower().endswith(image_ext)]

        print(f"   ➜ {class_name} ({len(files)} görsel)")

        for i, file in enumerate(files):

            old_path = os.path.join(class_path, file)

            ext = os.path.splitext(file)[1]

            new_name = f"{class_name}_{i+1:04d}{ext}"

            new_path = os.path.join(class_path, new_name)

            try:
                os.rename(old_path, new_path)
            except Exception:
                print("Hata atlandı:", file)

print("\n✅ Tüm dataset yeniden adlandırıldı.")