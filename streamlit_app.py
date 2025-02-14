import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import preprocess_text  # Mengimpor fungsi preprocessing dari file preprocessing.py

# Memuat model dan TF-IDF vectorizer yang sudah disimpan
model = joblib.load('model.pkl')
tfidf = joblib.load('tfidf.pkl')

# Judul Aplikasi
st.title('Prediksi Kategori Berita')

# Upload file dataset atau pilih file yang sudah ada
st.sidebar.header('Upload File CSV')
uploaded_file = st.sidebar.file_uploader("Pilih file dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    # Jika file tidak diupload, menggunakan dataset yang sudah ada
    df = pd.read_csv('dataset_berita_nlp_100.csv')

# Menampilkan beberapa baris data
st.write("Data yang dimuat:", df.head())

# Preprocessing Text
st.sidebar.header('Preprocessing')
text_input = st.text_area("Masukkan Berita untuk Prediksi")

if text_input:
    # Melakukan preprocessing pada teks input
    clean_text = preprocess_text(text_input)
    
    # Mengubah teks input menjadi vektor dengan TF-IDF
    tfidf_vector = tfidf.transform([clean_text])
    
    # Prediksi kategori dengan model
    prediction = model.predict(tfidf_vector)
    predicted_category = model.classes_[prediction[0]]
    
    st.write(f"Kategori yang diprediksi: {predicted_category}")
