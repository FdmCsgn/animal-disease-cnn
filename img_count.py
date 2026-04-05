import os

ROOT_DIR = r"C:\Users\ardac\OneDrive\Masaüstü\YL_project\data\hastalıklar"   # kendi yolunu yaz
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

for hayvan in os.listdir(ROOT_DIR):
    hayvan_path = os.path.join(ROOT_DIR, hayvan)

    if os.path.isdir(hayvan_path):
        print(f"\n🐾 {hayvan.upper()}")

        for hastalik in os.listdir(hayvan_path):
            hastalik_path = os.path.join(hayvan_path, hastalik)

            if os.path.isdir(hastalik_path):
                count = 0
                for root, dirs, files in os.walk(hastalik_path):
                    for file in files:
                        if file.lower().endswith(image_extensions):
                            count += 1

                print(f"   ➜ {hastalik}: {count} görsel")