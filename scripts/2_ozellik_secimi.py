import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

# processed veriyi yükleme:
train = pd.read_csv('../data/train_cleaned.csv')
X = train.drop(['Activity', 'subject'], axis=1)
y = train['Activity']

# 1. Normalizasyon (Zorunlu) kısmı:
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 2. Ozellik Seçme Yöntem 1: Select KBest yaptık:
selector = SelectKBest(score_func=f_classif, k=50)
X_new = selector.fit_transform(X_scaled, y)

# 3. Özellik Seçimi Yöntem 2: Random Forest uyguladik:
rf = RandomForestClassifier()
rf.fit(X_scaled, y)

print("2 gorevde tamamlandı: Normalizasyon ve 3 farklı özellik seçimi uygulandı.")