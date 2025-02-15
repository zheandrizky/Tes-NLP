import streamlit as st
import joblib
from preprocessing import preprocess_text  # Pastikan file preprocessing.py ada di folder yang sama

# Langkah 1: Memuat model dan TF-IDF
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

# Judul aplikasi
st.title("Prediksi Kategori Berita")

# Langkah 2: Input teks berita
user_input = st.text_area("Masukkan Teks Berita")

# Langkah 3: Button untuk memulai prediksi
if st.button("Prediksi Kategori"):
    if user_input:
        # Langkah 4: Preprocessing teks
        processed_text = preprocess_text(user_input)

        # Langkah 5: Transformasi teks ke dalam format TF-IDF
        tfidf_vector = tfidf.transform([processed_text])

        # Langkah 6: Prediksi kategori
        pred_label = model.predict(tfidf_vector)[0]
        pred_category = model.classes_[pred_label]

        # Output prediksi kategori
        st.write(f"Prediksi Kategori: {pred_category}")
    else:
        st.write("Masukkan teks berita terlebih dahulu.")
