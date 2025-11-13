# src/vector_space_model.py

import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === 1. LOAD DOKUMEN YANG SUDAH DIPROSES ===
def load_documents(folder):
    docs = {}
    if not os.path.exists(folder):
        print(f"[ERROR] Folder '{folder}' tidak ditemukan.")
        return docs

    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            path = os.path.join(folder, filename)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                # Hapus karakter non-huruf agar tidak error
                text = re.sub(r'[^a-zA-Z\s]', ' ', text)
                if text:  # hanya tambahkan kalau tidak kosong
                    docs[filename] = text
    return docs


# === 2. BANGUN TF-IDF MATRIX ===
def build_tfidf_matrix(docs):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs.values())
    feature_names = vectorizer.get_feature_names_out()
    return vectorizer, tfidf_matrix, feature_names


# === 3. REPRESENTASIKAN QUERY SEBAGAI VEKTOR TF-IDF ===
def vectorize_query(query, vectorizer):
    return vectorizer.transform([query])


# === 4. HITUNG COSINE SIMILARITY DAN RANKING ===
def rank_documents(tfidf_matrix, query_vector, doc_names, k=3):
    cosine_sim = cosine_similarity(query_vector, tfidf_matrix).flatten()
    ranked_indices = cosine_sim.argsort()[::-1]
    ranked_docs = [(doc_names[i], cosine_sim[i]) for i in ranked_indices[:k]]
    return ranked_docs


# === 5. PRECISION@K ===
def precision_at_k(predicted, gold):
    relevant = sum(1 for doc, _ in predicted if doc in gold)
    return relevant / len(predicted) if predicted else 0


# === 6. MAIN PROGRAM ===
if __name__ == "__main__":
    # ✅ Pastikan path ke folder processed benar
    data_folder = "D:/TUUUUUGGGGGGAAAAASSSSSSS/stki-uts-A11.2023.15390-AtanasiusMarcello/data/processed"
    docs = load_documents(data_folder)
    doc_names = list(docs.keys())

    print(f"Total dokumen: {len(docs)}\n")
    if len(docs) == 0:
        print("⚠️ Tidak ada dokumen yang terbaca! Periksa isi folder processed.")
        exit()

    # Bangun TF-IDF matrix
    vectorizer, tfidf_matrix, features = build_tfidf_matrix(docs)

    # === 7. UJI DENGAN BEBERAPA QUERY ===
    tests = [
        {"query": "pedang hutan", "gold": ["buku_fantasi.txt"]},
        {"query": "cinta motivasi", "gold": ["buku_romansa.txt", "buku_motivasi.txt"]},
        {"query": "ilmu sains pengetahuan", "gold": ["buku_sains.txt"]},
    ]

    for test in tests:
        query = test["query"]
        gold = test["gold"]
        print("=" * 60)
        print(f"Query: {query}")
        query_vec = vectorize_query(query, vectorizer)

        top_docs = rank_documents(tfidf_matrix, query_vec, doc_names, k=3)

        print(f"\nTop-3 Hasil Ranking:")
        for rank, (doc, score) in enumerate(top_docs, 1):
            snippet = docs[doc][:120].replace("\n", " ")
            print(f"{rank}. {doc:<25} | cosine={score:.4f} | {snippet}")

        precision = precision_at_k(top_docs, gold)
        print(f"\nPrecision@3: {precision:.2f}")
