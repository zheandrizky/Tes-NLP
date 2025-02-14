# -*- coding: utf-8 -*-
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# Unduh sumber daya NLTK yang diperlukan
nltk.download('punkt')
nltk.download('stopwords')

# Membuat stemmer menggunakan Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Mengambil daftar stopwords bahasa Indonesia
stop_words = set(stopwords.words("indonesian"))

# Fungsi preprocessing teks
def preprocess_text(text):
    # Mengubah teks menjadi huruf kecil
    text = text.lower()

    # Menghapus angka dan tanda baca
    text = re.sub(r'[^\w\s]', '', text)  # Menghapus tanda baca
    text = re.sub(r'\d+', '', text)      # Menghapus angka

    # Tokenisasi kata
    tokens = word_tokenize(text)

    # Menghapus stopwords dan melakukan stemming
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]

    # Menggabungkan kembali kata-kata yang sudah diproses menjadi sebuah teks
    return " ".join(tokens)

# Membaca dataset
file_path = 'dataset_berita_nlp_100.csv'  # Sesuaikan path jika perlu
df = pd.read_csv(file_path)

# Memeriksa data yang hilang
df.isna().sum()

# Menghapus baris yang memiliki nilai yang hilang pada kolom 'isi_berita' dan 'kategori'
df = df.dropna(subset=['isi_berita', 'kategori'])

# Melakukan preprocessing pada teks
df['clean_text'] = df['isi_berita'].apply(preprocess_text)

# Membuat model TF-IDF
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(df["clean_text"])  # Kolom teks yang sudah diproses

# Encoding kategori menjadi numerik
encoder = LabelEncoder()
df['kategori_encoded'] = encoder.fit_transform(df['kategori'])

# Membagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df["kategori_encoded"], test_size=0.2)

# Membuat model Naïve Bayes
model = MultinomialNB()

# Melatih model
model.fit(X_train, y_train)

# Prediksi pada data test
y_pred = model.predict(X_test)

# Evaluasi model
print("Akurasi Model: ", accuracy_score(y_test, y_pred))

# Simpan model yang telah dilatih
joblib.dump(model, "model.pkl")

# Simpan TF-IDF vectorizer
joblib.dump(tfidf, "tfidf.pkl")
