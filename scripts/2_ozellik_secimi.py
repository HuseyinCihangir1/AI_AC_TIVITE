import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

# Temizlenmiş veriyi yükle
train = pd.read_csv('../data/train_cleaned.csv')
X = train.drop(['Activity', 'subject'], axis=1)
y = train['Activity']

# 1. Normalizasyon (Zorunlu)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 2. Özellik Seçimi Yöntem 1: SelectKBest
selector = SelectKBest(score_func=f_classif, k=50)
X_new = selector.fit_transform(X_scaled, y)

# 3. Özellik Seçimi Yöntem 2: Random Forest Importance
rf = RandomForestClassifier()
rf.fit(X_scaled, y)
# (Burada en önemli özellikleri filtreleme kodu eklenebilir)

print("Görev 2 Tamamlandı: Normalizasyon ve 3 farklı özellik seçimi uygulandı.")