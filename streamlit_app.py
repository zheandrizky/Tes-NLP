import streamlit as st
import joblib
from preprocessing import preprocess_text  # Mengimpor preprocessing dari file preprocessing.py

# Load model dan TF-IDF yang sudah dilatih
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

# Setup Streamlit app
st.title("Klasifikasi Berita dengan NLP")

# Input dari user
user_input = st.text_area("Masukkan teks berita:")

if st.button("Klasifikasi"):
    # Proses teks input
    processed_text = preprocess_text(user_input)
    
    # Transformasi teks ke format TF-IDF
    tfidf_vector = tfidf.transform([processed_text])
    
    # Prediksi kategori
    pred_label = model.predict(tfidf_vector)[0]
    pred_category = model.classes_[pred_label]  # Menampilkan kategori hasil prediksi
    st.write(f"Prediksi Kategori: {pred_category}")
