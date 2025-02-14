import streamlit as st
import joblib
import pandas as pd
from preprocessing import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer

# Memuat model dan TF-IDF vectorizer
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

# Judul aplikasi
st.title("Klasifikasi Berita dengan NLP")

# Input dari pengguna
user_input = st.text_area("Masukkan teks berita:")

# Ketika tombol diklik
if st.button("Klasifikasi"):
    # Preprocessing teks
    processed_text = preprocess_text(user_input)
    
    # Mengubah teks yang diproses menjadi representasi TF-IDF
    tfidf_vector = tfidf.transform([processed_text])
    
    # Prediksi kategori berita
    pred_label = model.predict(tfidf_vector)[0]
    
    # Decode label ke kategori yang sesuai
    pred_category = model.classes_[pred_label]
    
    # Tampilkan hasil prediksi
    st.write(f"Prediksi Kategori: {pred_category}")
