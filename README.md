# ✏️ ANN Models That Recognize Drawing

Bu proje, kullanıcı tarafından çizilen (hand-drawn) görüntüleri tanıyabilen bir **Artificial Neural Network (ANN)** modelinin geliştirilmesini içermektedir.

Amaç, basit çizimlerden oluşan verileri kullanarak bir modelin **görsel pattern tanıma yeteneğini öğrenmesini sağlamaktır.**

---

## 🚀 Proje Özeti

El çizimleri (sketch/drawing), klasik görüntülere göre:

- Renk ve doku içermez
- Daha az bilgi barındırır
- Daha soyut ve zor sınıflandırılır  

Bu nedenle drawing recognition, bilgisayarla görmede özel bir problemdir. :contentReference[oaicite:0]{index=0}  

Bu projede:

- ✏️ Çizim verileri kullanılmıştır  
- 🧠 ANN (yapay sinir ağı) modeli eğitilmiştir  
- 📊 Model performansı analiz edilmiştir  

---

## 🧠 Kullanılan Yöntem: Artificial Neural Network (ANN)

Artificial Neural Networks:

- Girdi → Gizli katmanlar → Çıktı yapısına sahiptir  
- Görüntü sınıflandırma ve pattern recognition problemlerinde yaygın olarak kullanılır :contentReference[oaicite:1]{index=1}  
- Özellikle el yazısı ve çizim tanıma gibi görevlerde etkilidir :contentReference[oaicite:2]{index=2}  

Bu projede ANN modeli:

- Piksel değerlerini giriş olarak alır  
- Feature extraction işlemini öğrenir  
- Sınıflandırma yapar  

---

## 🏗️ Proje Yapısı

```

ANN-models-that-recognizes-drawing/
│
├── notebook.ipynb        # Model geliştirme ve eğitim süreci
├── dataset/              # Çizim verileri (varsa)
├── README.md             # Proje dokümantasyonu

````

---

## ⚙️ Kurulum

### 1. Repoyu klonla
```bash
git clone https://github.com/beratolmez/ANN-models-that-recognizes-drawing.git
cd ANN-models-that-recognizes-drawing
````

### 2. Gerekli kütüphaneleri yükle

```bash
pip install numpy matplotlib tensorflow keras
```

---

## ▶️ Kullanım

Jupyter Notebook’u başlat:

```bash
jupyter notebook
```

Ardından:

```
notebook.ipynb
```

dosyasını aç ve hücreleri sırayla çalıştır.

---

## 📊 Çıktılar

Bu proje aşağıdaki çıktıları üretir:

* 📈 Eğitim / doğrulama loss grafikleri
* 🎯 Model accuracy sonuçları
* 🤖 Çizim tahminleri
* 🔍 Model performans analizi

---

## 🔍 Öğrenilenler

Bu proje ile:

* Yapay sinir ağlarının temel çalışma prensibi
* Görüntü verisinin vektörleştirilmesi
* Pattern recognition mantığı
* Model eğitimi ve değerlendirme süreçleri

uygulamalı olarak öğrenilmiştir.

---

## 📌 Geliştirme Fikirleri

* [ ] CNN ile karşılaştırma (çok önemli)
* [ ] Daha büyük dataset kullanımı (QuickDraw gibi)
* [ ] Data augmentation eklenmesi
* [ ] Web tabanlı çizim uygulaması (canvas + model)
* [ ] Gerçek zamanlı tahmin sistemi

---

## 📊 Örnek Hyperparameters

| Parametre     | Değer |
| ------------- | ----- |
| Learning Rate | 0.001 |
| Batch Size    | 32    |
| Epoch         | 10    |
| Optimizer     | Adam  |

---

## ⚠️ Not

ANN modelleri, görüntü verisinde CNN’lere göre daha sınırlı performans gösterebilir.
Çünkü CNN’ler spatial (uzamsal) bilgiyi daha iyi yakalar.

---

## 🤝 Katkı

Katkılara açıktır. Pull request gönderebilirsiniz.

---

## 📜 Lisans

MIT License

---

## 👤 Yazar

**Berat Ölmez**

* 💻 AI & Machine Learning Developer
* 🔗 GitHub: [https://github.com/beratolmez](https://github.com/beratolmez)
İstersen bir sonraki adımda:
👉 Bu projeyi **web app + gerçek zamanlı çizim tanıma sistemi** haline getirelim (çok güçlü portföy projesi olur).
