# app/chat.py
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === 1. LOAD DOKUMEN (otomatis ubah list → string) ===
def load_documents(folder):
    docs = {}
    if not os.path.exists(folder):
        print(f"⚠️ Folder {folder} tidak ditemukan!")
        return docs

    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                text = f.read().strip()
                # Kalau file berisi token yang dipisah spasi atau koma, ubah jadi string
                if text.startswith("[") and text.endswith("]"):  # tanda list
                    text = text.replace("[", "").replace("]", "").replace("'", "").replace(",", "")
                docs[filename] = text
    return docs


# === 2. VECTOR SPACE MODEL SEARCH ===
def vsm_search(query, docs, k=3):
    """Cari top-k dokumen paling relevan dengan cosine similarity"""
    if len(docs) == 0:
        print("⚠️ Tidak ada dokumen untuk diproses.")
        return []

    # pastikan semua dokumen adalah string
    clean_docs = [str(v) for v in docs.values()]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(clean_docs)
    query_vec = vectorizer.transform([query])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    ranked_indices = cosine_sim.argsort()[::-1]

    doc_names = list(docs.keys())
    results = [(doc_names[i], cosine_sim[i]) for i in ranked_indices[:k]]
    return results


# === 3. TEMPLATE JAWABAN ===
def generate_response(query, results, docs):
    if not results:
        return f"❌ Tidak ditemukan dokumen relevan untuk '{query}'."

    response = f"🔍 Berdasarkan pencarian untuk '{query}', berikut {len(results)} dokumen teratas:\n\n"
    for i, (doc, score) in enumerate(results, 1):
        snippet = docs[doc][:120].replace("\n", " ")
        response += f"{i}. {doc:<25} (cosine: {score:.3f}) — {snippet}\n"
    response += "\n🧠 Sistem menampilkan hasil paling relevan berdasarkan kesamaan deskripsi teks."
    return response


# === 4. ANTARMUKA CHAT ===
def chat_interface():
    data_folder = "data/processed"
    docs = load_documents(data_folder)

    print("=" * 60)
    print("🤖 Mini Search Assistant (VSM-based)")
    print("Ketik pertanyaan atau kata kunci Anda (ketik 'exit' untuk keluar)")
    print("=" * 60)

    while True:
        query = input("\n🗨️  Query: ").strip()
        if query.lower() == "exit":
            print("👋 Terima kasih! Program selesai.")
            break

        results = vsm_search(query, docs, k=3)
        response = generate_response(query, results, docs)
        print("\n" + response)
        print("-" * 60)


# === 5. MAIN ===
if __name__ == "__main__":
    chat_interface()
