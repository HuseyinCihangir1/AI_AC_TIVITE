# Proje: Akıllı Telefon Sensörleri ile İnsan Aktivitesi Tanıma

Bu proje, **BIM 322 - Makine Öğrenmesi ve Uygulamaları** dersi final ödevi kapsamında geliştirilmiştir.
Akıllı telefon sensör verileri işlenmiş, optimize edilmiş ve **LIME (XAI)** ile **Streamlit** entegrasyonu sağlanarak canlı bir web uygulamasına dönüştürülmüştür. 


# Ekip ve Görev Dağılımı

| Ekip Üyesi | Rolü / Sorumlulukları | Tamamlanan Akademik Görevler |
| :--- | :--- | :--- |
| **Hüseyin Cihangir** | Veri Bilimci (Data Scientist) | **Görev 1:** Z-Score ile Aykırı Değer Temizliği<br>**Görev 2:** MinMaxScaler Normalizasyonu & 3 Farklı Özellik Seçimi<br>**Görev 3:** 3 Farklı Model Eğitimi (SVM, Random Forest, KNN) |
| **Arda Aktürk** | ML & Dağıtım Mühendisi | **Görev 4:** Eğitim-Doğrulama-Test Metrik Kıyaslamaları<br>**Görev 4:** ROC Eğrisi ve Sınıf Bazlı Recall Grafik Analizleri<br>**Görev 4:** LIME Protokol Entegrasyonu & Streamlit Dağıtımı |


# Model Performans Sonuçları (Test Verisi)

Veri seti akademik standartlara uygun olarak **%70 Eğitim, %15 Doğrulama ve %15 Test** seti olacak şekilde bölünmüştür. Modellerin genel test sonuçları:

| Algoritma | Doğruluk (Accuracy) | Hassasiyet (Precision) | Duyarlılık (Recall) | F1-Skor |
| :--- | :---: | :---: | :---: | :---: |
| **Support Vector Machine (SVM)** | **%96.2** | **%96.5** | **%96.1** | **%96.3** |
| **Random Forest (RF)** | %93.8 | %94.1 | %93.5 | %93.8 |
| **K-Nearest Neighbors (KNN)** | %90.5 | %91.0 | %90.2 | %90.6 |

# Neden SVM En İyi Çıktı?: Projedeki sensör verisi 561 gibi çok yüksek bir özellik boyutuna sahipti. KNN bu boyutta mesafeleri ölçemedi, Random Forest ise kurallarda boğuldu. SVM algoritması ise bu yüksek boyutlu uzayda sınıflar arasına en kararlı çizgiyi (hiper-düzlem) çekerek ezberleme (overfitting) yapmadan %96.2 doğrulukla en iyi performansı verdi.



# Projeyi Çalıştırma Adımları

1. Gerekli kütüphaneleri yükleyin:
   
   pip install pandas numpy scikit-learn matplotlib seaborn streamlit lime

2. Model değerlendirme metriklerini ve analiz grafiklerini çalıştırmak için:

    python scripts/4_degerlendirme.py

3. Canlı Streamlit web uygulamasını başlatmak için: 

    streamlit run app.py
