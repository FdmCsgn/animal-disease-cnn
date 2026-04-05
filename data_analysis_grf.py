import os
import matplotlib.pyplot as plt

dataset_path = "C:\\Users\\ardac\\OneDrive\\Masaüstü\\YL_project\\data_split\\train"  # dataset klasörün

class_counts = {}

for root, dirs, files in os.walk(dataset_path):
    
    images = [f for f in files if f.lower().endswith((".jpg",".jpeg",".png"))]
    
    if len(images) > 0:
        class_name = os.path.basename(root)  # en son klasör adı
        class_counts[class_name] = len(images)

# grafik verileri
classes = list(class_counts.keys())
counts = list(class_counts.values())

plt.figure(figsize=(14,6))
plt.bar(classes, counts)

plt.title("Dataset Class Distribution")
plt.xlabel("Classes")
plt.ylabel("Image Count")
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()