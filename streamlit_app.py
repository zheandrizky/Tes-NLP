import streamlit as st
import joblib
#from preprocessing import preprocess_text  # Import fungsi preprocessing

# Muat model dan TF-IDF vectorizer
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

# Menampilkan judul aplikasi
st.title("Klasifikasi Berita dengan NLP")

# Input teks dari pengguna
user_input = st.text_area("Masukkan teks berita:")

# Ketika tombol diklik, proses klasifikasi
if st.button("Klasifikasi"):
    processed_text = preprocess_text(user_input)  # Proses teks
    tfidf_vector = tfidf.transform([processed_text])  # Vektorize teks
    pred_label = model.predict(tfidf_vector)[0]  # Prediksi kategori
    pred_category = df["kategori"].astype("category").cat.categories[pred_label]  # Mengonversi label ke kategori
    st.write(f"Prediksi Kategori: {pred_category}")  # Menampilkan hasil prediksi
