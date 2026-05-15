import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

# 1. processed Veriyi Yükleme
train = pd.read_csv('../data/train_cleaned.csv')
X = train.drop(['Activity', 'subject'], axis=1)
y = train['Activity']

# ÖZELLİK SEÇİMİ:

# Adım 1: Normalizasyon 
scaler = MinMaxScaler() #veriyi 0-1 arasına kısıtlıyoruz
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# YÖNTEM 1: İstatistiksel Seçim(KBest)
selector = SelectKBest(score_func=f_classif, k=50) #En yüksek f-skoruna sahip 50 özelliği seçicez
X_best = selector.fit_transform(X_scaled, y)

# YÖNTEM 2: Model Tabanlı Seçim(Random Forest)
rf = RandomForestClassifier(n_estimators=100, random_state=42)#Tüm Özelliklerin sınıflandırmadaki etkisini bulur
rf.fit(X_scaled, y)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)


# 3-YÖNTEM : Korelasyon Tabanlı Seçim
plt.figure(figsize=(12,10))# Birbiriyle aşırı benzer  olan özellikleri tespit edicek kısım
cor = X_scaled.iloc[:, :20].corr() # İlk 20 özellik için ısı haritasi cikaricaz
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.title("Özellikler Arası Korelasyon Isı Haritası")
plt.show() # Bu grafiğin ekran görüntüsü(sunum için ss alınacak*)

print("Görev 2 Tamamlandı: Normalizasyon ve 3 farklı özellik seçimi uygulandı.")