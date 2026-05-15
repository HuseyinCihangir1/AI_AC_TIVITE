import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


# PATH
current_dir = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(current_dir, "../data/train_cleaned.csv")
model_path = os.path.join(current_dir, "../models")

# Veriyi yukle
df = pd.read_csv(data_path)

X = df.drop(['Activity', 'subject'], axis=1)
y = df['Activity']

# Train / test ayrimi
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Verileri ve modelleri hazırla
models = {
    "SVM": SVC(kernel='linear', probability=True),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier()
}

results = {}

# Modelleri egit ve test et
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    results[name] = acc

    print(f"{name} Accuracy: {acc:.4f} | F1: {f1:.4f}")

# En iyi modeli bul
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\nEn Başarılı Model:", best_model_name)

pickle.dump(best_model, open('../models/best_model.pkl', 'wb'))

print("En başarılı model kaydedildi.")
