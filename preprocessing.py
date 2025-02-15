import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Membuat stemmer menggunakan Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Mengambil daftar stopwords bahasa Indonesia
stop_words = set(stopwords.words("indonesian"))

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
