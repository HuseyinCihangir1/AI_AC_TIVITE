import pandas as pd
import shap
import lime
import pickle
import numpy as np
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split

# En iyi modeli yukle
with open("../models/best_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]

# Veriyi yukle
df = pd.read_csv("../data/train_cleaned.csv")

X = df.drop(["Activity", "subject"], axis=1)
y = df["Activity"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

### LIME EXPLAINER
explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns,
    class_names=np.unique(y),
    mode="classification"
)

# Bir ornek acikla
i = 0  # aciklanacak ornek

exp = explainer.explain_instance(
    data_row=X_test.iloc[i].values,
    predict_fn=model.predict_proba,
    num_features=10
)

# Ciktiyi goster
print("Gerçek sınıf:", y_test.iloc[i])
print("Tahmin:", model.predict(X_test.iloc[i:i+1])[0])

# Plot
fig = exp.as_pyplot_figure()
plt.title("LIME Explanation (Local Interpretability)")
plt.show()

# HTML olarak kaydet
exp.save_to_file("lime_explanation.html")


### SHAP
# background data
# background = shap.sample(X_train, 50)

# explainer = shap.KernelExplainer(model.predict_proba, background)

# hizlandirmak icin kucuk sample size kullan
# X_sample = X_test.sample(20)

# shap_values = explainer.shap_values(X_sample)

# shap.summary_plot(shap_values, X_sample)
# plt.show()