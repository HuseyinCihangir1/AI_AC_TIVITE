import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize



### Modeli yukle
with open("../models/best_model.pkl", "rb") as f:
    model = pickle.load(f)


### Veriyi yukle
df = pd.read_csv("../data/train_cleaned.csv")

X = df.drop(["Activity", "subject"], axis=1)
y = df["Activity"]

# Test split (evaluation için)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


### Tahmin
y_pred = model.predict(X_test)


### Metrikler 
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Metrikleri yazdir
print("\n--Model Performansı--")
print("Accuracy : ", acc)
print("Precision : ", prec)
print("Recall : ", rec)
print("F1 Score : ", f1)


### Classification Report 
print("Classification Report : \n")
print(classification_report(y_test, y_pred))


### Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()


### ROC Analizi

# Ikili etiketler
classes = sorted(y.unique())

y_test_bin = label_binarize(y_test, classes=classes)

# Olasılık Skorları
y_score = model.predict_proba(X_test)

plt.figure(figsize=(10,8))

for i in range(len(classes)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"Class {classes[i]} (AUC = {roc_auc:.2f})")

             

plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()



