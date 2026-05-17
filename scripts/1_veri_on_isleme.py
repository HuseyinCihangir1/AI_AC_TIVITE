import pandas as pd
import numpy as np
from scipy import stats

# Veri Yükleme:
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# -Görev 1: (Z-Score) hesaplaması-

# Sadece sayısal sütunlar için aykırı değer kontrolü yapıyoruz:
X_train = train.drop(['Activity', 'subject'], axis=1)
z_scores = stats.zscore(X_train)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)

train_cleaned = train[filtered_entries]
train_cleaned.to_csv('../data/train_cleaned.csv', index=False)

print("Görev 1 Tamamlandı: Aykırı değerler temizlendi ve veri kaydedildi.")