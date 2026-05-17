import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

# Dosya yollarını garantiye alma
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '../data/train_cleaned.csv')
model_dir = os.path.join(current_dir, '../models')

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 1. İşlenmiş Veriyi Yükleme
train = pd.read_csv(data_path)
X = train.drop(['Activity', 'subject'], axis=1)
y = train['Activity']

# --- GÖREV 2: ÖZELLİK SEÇİMİ (25 PUAN) ---

# Adım 1: Normalizasyon (Zorunlu)
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Arda'nın(diger uye) verileri aynı şekilde ölçekleyebilmesi için scaler'ı kaydediyoruz
with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

# YÖNTEM 1: İstatistiksel Seçim (SelectKBest)
# En yüksek f-skoruna sahip 50 özelliği seçiyoruz
selector = SelectKBest(score_func=f_classif, k=50)
X_best = selector.fit_transform(X_scaled, y)

# Arda'nın hangi 50 özelliğin seçildiğini bilmesi için selector'ı kaydediyoruz
with open(os.path.join(model_dir, 'feature_selector.pkl'), 'wb') as f:
    pickle.dump(selector, f)

# YÖNTEM 2: Model Tabanlı Seçim (Random Forest Feature Importance)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

# YÖNTEM 3: Korelasyon Tabanlı Seçim (Isı Haritası)
plt.figure(figsize=(12,10))
cor = X_scaled.iloc[:, :20].corr() # İlk 20 özellik için örnekleme
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.title("Özellikler Arası Korelasyon Isı Haritası")
plt.show() # Bu grafiğin ekran görüntüsü sunum (Görev 5) için alınmalı!

print("Görev 2 Tamamlandı: Scaler ve Selector araçları 'models/' klasörüne kaydedildi.")