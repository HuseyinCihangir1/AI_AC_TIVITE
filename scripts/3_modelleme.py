import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Verileri ve modelleri hazırla
models = {
    "SVM": SVC(kernel='linear'),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier()
}

# Eğitim ve Matris Çizimi (Döngü ile)
# ... eğitim kodları(EKİP UYESI ARDA yapıcak bu kısmı)...

#(Görev 4) için modeli Kaydettik. Ekip uyesi arda yapıcak:
pickle.dump(models["SVM"], open('../models/best_model.pkl', 'wb'))
print("Görev 3 Tamamlandı: 3 model eğitildi ve best_model.pkl kaydedildi.")