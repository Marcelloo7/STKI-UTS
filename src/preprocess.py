# src/preprocess.py

import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter

# Download resource NLTK jika belum ada
nltk.download('stopwords')

# === 1. CASE FOLDING & CLEANING ===
def clean(text):
    text = text.lower()  # case folding
    text = re.sub(r'\d+', '', text)  # hapus angka
    text = text.translate(str.maketrans("", "", string.punctuation))  # hapus tanda baca
    text = text.strip()
    return text

# === 2. TOKENIZATION ===
def tokenize(text):
    tokens = text.split()
    return tokens

# === 3. STOPWORD REMOVAL ===
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('indonesian'))
    filtered = [t for t in tokens if t not in stop_words]
    return filtered

# === 4. STEMMING ===
def stem(tokens):
    stemmer = PorterStemmer()  # ganti dengan Sastrawi untuk Bahasa Indonesia jika ingin lebih akurat
    stemmed = [stemmer.stem(t) for t in tokens]
    return stemmed

# === 5. PIPELINE UTAMA ===
def preprocess_document(text):
    text = clean(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stem(tokens)
    return tokens

# === 6. PROSES SEMUA FILE DI FOLDER RAW ===
def preprocess_all(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(input_folder, filename), "r", encoding="utf-8") as f:
                text = f.read()
            
            tokens = preprocess_document(text)

            # Simpan hasil ke data/processed
            output_path = os.path.join(output_folder, filename)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(" ".join(tokens))
            
            # Hitung token paling sering
            counter = Counter(tokens)
            print(f"\nDokumen: {filename}")
            print("10 token paling sering:", counter.most_common(10))
            print("Jumlah token total:", len(tokens))

# === 7. MAIN ===
if __name__ == "__main__":
    # Pastikan path selalu benar, tidak tergantung dari lokasi eksekusi
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed")

    preprocess_all(raw_dir, processed_dir)
