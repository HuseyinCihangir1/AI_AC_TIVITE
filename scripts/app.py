import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split

# Path
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.normpath(os.path.join(current_dir, '../models'))

model_path = os.path.join(model_dir, 'best_model.pkl')
scaler_path = os.path.join(model_dir, 'scaler.pkl')
selector_path = os.path.join(model_dir, 'feature_selector.pkl')


# Modeli yukle
try:
    with open(model_path, "rb") as f:
        data = pickle.load(f)

    model = data["model"]
    model_name = data["name"]

except Exception as e:
    st.error(f"Model yüklenemedi: {e}")
    st.stop()


# Veriyi yukle
df = pd.read_csv("../data/train_cleaned.csv")

X = df.drop(["Activity", "subject"], axis=1)
y = df["Activity"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


### STREAMLIT UI
st.title("Human Activity Recognition (HAR)")

st.write(f"Model: {model_name} + LIME Explainability")

# Ornek secici
index = st.slider("Bir örnek seç", 0, len(X_test)-1, 0)

sample = X_test.iloc[index]

st.subheader("Seçilen Veri")
st.write(sample)

# prediction
pred = model.predict(sample.values.reshape(1, -1))[0]

st.subheader("Tahmin")
st.success(pred)

### LIME EXPLANATION
st.subheader("Model Açıklaması (LIME)")

explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns,
    class_names=np.unique(y),
    mode="classification"
)

exp = explainer.explain_instance(
    sample.values,
    model.predict_proba,
    num_features=10
)

fig = exp.as_pyplot_figure()
st.pyplot(fig)

# HTML download
st.markdown("### Detaylı açıklama")
html = exp.as_html()
st.components.v1.html(html, height=500, scrolling=True)