# 🐾 Hayvan Hastalıklarının Görüntü Tabanlı Derin Öğrenme ile Sınıflandırılması

Bu proje, hayvanlarında görülen hastalıkların fotoğraf görüntüleri kullanılarak otomatik olarak teşhis edilmesini amaçlamaktadır. Bir veteriner ya da çiftçi, hasta görünen hayvanın fotoğrafını sisteme verdiğinde, model hangi hastalık olduğunu tahmin etmektedir.

6 farklı hayvan türüne ait **47 hastalık sınıfı** üzerinde üç farklı derin öğrenme modeli eğitilmiş ve karşılaştırılmıştır.

---

## 🎯 Projenin Amacı

Geleneksel hastalık teşhisi için deneyimli bir veteriner gerekmektedir ve bu her zaman mümkün olmayabilir. Bu proje, görüntü tabanlı yapay zeka kullanarak:

- Hastalık teşhisini hızlandırmayı
- Veteriner erişiminin kısıtlı olduğu bölgelerde destek sağlamayı
- Farklı model mimarilerinin bu görevdeki başarısını karşılaştırmayı

hedeflemektedir.

---

## 📂 Veri Seti

Veri seti 3 parçaya ayrılmıştır:

| Bölüm | Açıklama |
|-------|----------|
| **Train** | Modelin öğrendiği görüntüler |
| **Validation** | Eğitim sırasında modelin izlendiği görüntüler |
| **Test** | Modelin hiç görmediği, nihai değerlendirme görüntüleri |

Veri setinde bazı hastalık sınıfları diğerlerinden çok daha az örneğe sahiptir. Bu **sınıf dengesizliği** problemi, modelin her zaman çoğunluk sınıfını tahmin etmesine yol açabilir. Bu sorunu çözmek için `WeightedRandomSampler` kullanılmıştır. Bu yöntem, az örnekli sınıfları eğitim sırasında daha sık göstererek modelin tüm sınıfları eşit öğrenmesini sağlar.

---

## 🧠 Kullanılan Modeller

Beş farklı derin öğrenme modeli seçilmiş ve aynı koşullar altında eğitilmiştir. Her model farklı bir mimari yaklaşımı temsil etmektedir.

### 1. ResNeSt101
Geleneksel evrişimli sinir ağı (CNN) mimarisinin geliştirilmiş versiyonudur. **Split-Attention** mekanizması sayesinde görüntünün farklı bölgelerini aynı anda farklı dikkat ağırlıklarıyla işleyebilir. `timm` kütüphanesi üzerinden yüklenmiştir.

### 2. EfficientNetV2-M
Google tarafından geliştirilen, **bileşik ölçekleme** (compound scaling) kullanan verimli bir CNN modelidir. Derinlik, genişlik ve çözünürlüğü dengeli şekilde artırarak hem yüksek başarı hem de düşük hesaplama maliyeti elde eder. `torchvision` kütüphanesi üzerinden yüklenmiştir.

### 3. CvT-21 (Convolutional Vision Transformer)
Microsoft tarafından geliştirilen bu model, CNN ve Transformer mimarilerini birleştiren hibrit bir yaklaşım sunar. Transformer'ların uzun menzilli bağımlılık kurma gücünü, CNN'lerin yerel özellik çıkarma avantajıyla birleştirir. HuggingFace `microsoft/cvt-21` üzerinden yüklenmiştir.

### 4. ResNeXt101-32x8d
ResNet mimarisinin geliştirilmiş versiyonudur. **Grouped Convolution** 
kullanarak aynı parametre sayısıyla daha güçlü özellik öğrenir. 
"32x8d" ifadesi 32 grup ve grup başına 8 genişlik anlamına gelir. 
`torchvision` kütüphanesinden yüklenmiş, son `fc` katmanı 47 
sınıf için değiştirilmiştir.

### 5. ConvNeXtV2-Base
Facebook/Meta tarafından geliştirilen bu model, Transformer 
mimarisinden ilham alarak tasarlanmış modern bir CNN'dir. 
**Global Response Normalization (GRN)** katmanı sayesinde özellikler 
arasındaki rekabeti artırır ve daha ayırt edici özellikler öğrenir. 
22K ImageNet sınıfı üzerinde önceden eğitilmiştir. HuggingFace 
`facebook/convnextv2-base-22k-224` üzerinden yüklenmiş, özel bir 
sınıflandırıcı başlığı (LayerNorm → Dropout → Linear → GELU → 
Linear) eklenmiştir.

| Model | Kaynak | Parametre Sayısı | Mimari Türü |
|-------|--------|-----------------|-------------|
| ResNeSt101e | timm | ~48M | CNN (Split-Attention) |
| EfficientNetV2-M | torchvision | ~54M | CNN (Compound Scaling) |
| CvT-21 | HuggingFace | ~32M | Hibrit (CNN + Transformer) |
| ResNeXt101-32x8d | torchvision | ~88M | CNN (Grouped Convolution) |
| ConvNeXtV2-Base | HuggingFace (facebook/convnextv2-base-22k-224) | ~89M | Modern CNN (Transformer-inspired) |

---

## 🔄 Eğitim Süreci

### İki Aşamalı Transfer Learning

Modeller ImageNet veri seti üzerinde milyonlarca görüntüyle önceden eğitilmiştir. Bu öğrenilen genel özellikleri (kenar, doku, şekil) kullanmak için **transfer learning** uygulanmıştır. Eğitim iki aşamada gerçekleştirilmiştir:

**Aşama 1 — Backbone Freeze (Dondurma)**

Modelin tüm katmanları dondurulur, yalnızca son sınıflandırıcı katman eğitilir. Model bu aşamada "ImageNet'ten öğrendiği genel özellikleri kullanarak hangi sınıflandırıcıyı eklemeliyim?" sorusunu cevaplar. Yüksek learning rate (1e-3) kullanılarak hızlı bir başlangıç yapılır.

**Aşama 2 — Fine-Tuning (İnce Ayar)**

Tüm model katmanları açılır ve çok düşük bir learning rate (1e-5) ile eğitime devam edilir. Bu aşamada model, hastalık görüntülerine özgü ince detayları öğrenir. Düşük learning rate kullanılmasının sebebi, yüksek değerlerin önceden öğrenilen değerli ağırlıkları bozmasını engellemektir.

### Eğitim Parametreleri

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| Görsel boyutu | 224×224 | Tüm görseller bu boyuta getirilir |
| Batch size | 8 | Aynı anda işlenen görsel sayısı |
| Efektif batch | 32 | Gradient accumulation ile elde edilen etki |
| Optimizer | AdamW | Ağırlık güncellemesi için |
| Scheduler | CosineAnnealingLR | Learning rate'i kademeli düşürür |
| Gradient clipping | 1.0 | Gradyan patlamasını önler |

---
# 🖥️ Donanım Gereksinimleri

Bu proje aşağıdaki donanımlar üzerinde geliştirilmiştir:

---------------------------Fadime----------------------------------------------
| Bileşen | Detay |
|---------|-------|
| GPU | NVIDIA RTX 3050 4GB VRAM |
| İşletim Sistemi | Windows |
| Python | 3.10 |
| PyTorch | 2.x |

-------------------------Serhan--------------------------------------------------
## 🖥️ Donanım

| Bileşen        | Detay                                      |
|----------------|--------------------------------------------|
| GPU (Yerel)    | NVIDIA GeForce GTX 1650 Ti (4GB VRAM)      |
| GPU (Cloud)    | Google Colab Tesla T4 (15.6GB VRAM)        |
| CUDA           | 12.1                                       |
| Python         | 3.12                                       |
| PyTorch        | 2.x                                        |

4GB VRAM ile çalışabilmek için gradient accumulation, mixed precision (AMP) ve subset training gibi bellek optimizasyon teknikleri kullanılmıştır.

## 🎨 Veri Artırma (Augmentation)

Modelin sadece eğitim verilerini ezberlemek yerine gerçek özellikleri öğrenmesi için gelişmiş veri artırma teknikleri kullanılmıştır.

### MixUp
İki farklı görüntü piksel düzeyinde belirli bir oranda karıştırılır. Etiketler de aynı oranda karıştırılır. Örneğin %60 meme iltihabı + %40 ayak çürümesi görüntüsü oluşturulur ve model bu karışıma göre tahmin yapmayı öğrenir. Model daha yumuşak ve genelleştirilebilir karar sınırları oluşturur.

### CutMix
Bir görüntünün rastgele seçilen dikdörtgen bölgesi, başka bir görüntünün aynı bölgesiyle değiştirilir. MixUp'tan farklı olarak bölgeler net sınırlara sahiptir. Model görüntünün farklı bölümlerine odaklanmayı öğrenir.

### Both Modu
Her batch'te %50 olasılıkla MixUp, %50 olasılıkla CutMix seçilir. Her iki yöntemin avantajlarından aynı anda yararlanılır.

Farklı augmentation modlarının (mixup / cutmix / both / none) model başarısına etkisi de bu proje kapsamında karşılaştırılmıştır.

---

## 📊 Değerlendirme

Model eğitildikten sonra hiç görmediği test görselleri üzerinde değerlendirilmiştir. Değerlendirmede beş farklı yöntem kullanılmıştır.

### Accuracy (Doğruluk)
Modelin tüm tahminleri içinde doğru olanların oranıdır. Anlaşılması kolay ancak sınıf dengesizliğinde yanıltıcı olabilir. 95 görsel sağlıklıysa model hepsine "sağlıklı" dese bile %95 accuracy alır.

### Precision (Kesinlik)
Model bir sınıfı tahmin ettiğinde ne kadar haklıdır sorusunu cevaplar. Yanlış alarm oranını ölçer. Düşük precision modelin gereksiz yere çok fazla "hasta" dediği anlamına gelir.

### Recall (Duyarlılık)
Gerçek hastaların kaçının doğru tespit edildiğini ölçer. Tıbbi teşhiste kritik öneme sahiptir çünkü hasta bir hayvanı kaçırmak ciddi sonuçlar doğurabilir.

### F1 Score
Precision ve Recall'un dengeli ortalamasıdır. Dengesiz veri setlerinde en güvenilir metriktir. Proje boyunca model seçimi bu metriğe göre yapılmıştır.

### Confusion Matrix (Karışıklık Matrisi)
47×47 boyutunda bir tablodur. Her satır gerçek sınıfı, her sütun tahmin edilen sınıfı gösterir. Köşegen üzerindeki değerler doğru tahminleri, köşegen dışındakiler ise hangi sınıfın hangi sınıfla karıştırıldığını gösterir. Hangi hastalıkların birbirine benzediğini analiz etmek için kullanılır.

---

## 🔍 Grad-CAM — Model Neye Bakıyor?

Metrikler modelin ne kadar doğru tahmin yaptığını söyler ancak neden bu kararı verdiğini söylemez. Grad-CAM bu soruyu görsel olarak cevaplar.

Model bir görüntüyü sınıflandırırken hangi piksellerin kararı etkilediği ısı haritası olarak gösterilir. Kırmızı bölgeler kararı en çok etkileyen alanlardır, mavi bölgeler ise kararı çok az etkiler.

Bu görselleştirme sayesinde modelin gerçekten hastalıklı bölgeye mi odaklandığı, yoksa arka plandaki alakasız bir örüntüyü mü öğrendiği anlaşılabilir. Her sınıftan 2 örnek görsel için orijinal görüntü, ısı haritası ve üst üste bindirilmiş görüntü olmak üzere 3 panel üretilmiştir. Doğru tahminler yeşil, yanlış tahminler kırmızı çerçeve ile gösterilmiştir.

---

## 📦 Kurulum
```bash
pip install torch torchvision
pip install transformers huggingface_hub hf_xet
pip install timm
pip install pytorch-grad-cam
pip install scikit-learn matplotlib seaborn tqdm
```

Augmentation modunu değiştirmek için ilgili dosyada şu satırı düzenleyin:
```python
AUGMENTATION_MODE = "both"  # "both" | "mixup" | "cutmix" | "none"
```

## 📄 Lisans

Bu proje akademik amaçlı geliştirilmiştir.
