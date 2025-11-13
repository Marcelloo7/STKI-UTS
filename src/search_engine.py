import argparse
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from boolean_retrieval import build_inverted_index
import numpy as np

# === 1. LOAD DOKUMEN ===
def load_documents(folder):
    docs = {}
    if not os.path.exists(folder):
        print(f"⚠️ Folder {folder} tidak ditemukan!")
        return docs
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                # 🟢 ubah ke list token (sesuai format hasil preprocessing)
                docs[filename] = f.read().lower().split()
    return docs


# === 2. BOOLEAN SEARCH (versi final fix) ===
def boolean_search(query, inverted_index):
    """
    Boolean Search:
    - Mendukung AND, OR, NOT
    - Mencari token mirip (substring match)
    """
    tokens = query.lower().split()
    result = set()
    operator = None

    for token in tokens:
        token = token.strip().lower()

        if token in ["and", "or", "not"]:
            operator = token
            continue

        # cari token yang mirip (misal 'pedang' cocok dengan 'pedang' atau 'pedangnya')
        matched_terms = [t for t in inverted_index.keys() if token in t]

        docs = set()
        for term in matched_terms:
            docs |= set(inverted_index.get(term, []))

        if not result:
            result = docs
        else:
            if operator == "and":
                result &= docs
            elif operator == "or":
                result |= docs
            elif operator == "not":
                result -= docs

    return list(result)


# === 3. VECTOR SPACE MODEL ===
def vsm_search(query, docs, k=3):
    if len(docs) == 0:
        print("⚠️ Tidak ada dokumen untuk diproses.")
        return [], None

    # ubah list token jadi string supaya bisa diolah TF-IDF
    doc_texts = {name: " ".join(tokens) for name, tokens in docs.items()}

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(doc_texts.values())
    query_vec = vectorizer.transform([query])

    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    ranked_indices = cosine_sim.argsort()[::-1]
    doc_names = list(doc_texts.keys())
    results = [(doc_names[i], cosine_sim[i]) for i in ranked_indices[:k]]
    return results, vectorizer


# === 4. MAIN ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini Search Engine CLI")
    parser.add_argument("--model", choices=["boolean", "vsm"], required=True, help="Pilih model pencarian")
    parser.add_argument("--k", type=int, default=3, help="Jumlah dokumen hasil (untuk VSM)")
    parser.add_argument("--query", type=str, required=True, help="Masukkan query pencarian")
    args = parser.parse_args()

    data_folder = "data/processed"
    docs = load_documents(data_folder)

    if len(docs) == 0:
        print("⚠️ Folder data/processed kosong atau belum ada hasil preprocessing.")
        exit()

    # === BOOLEAN ===
    if args.model == "boolean":
        inverted_index = build_inverted_index(docs)
        hasil = boolean_search(args.query, inverted_index)

        print(f"\nModel: BOOLEAN RETRIEVAL")
        print(f"Query: {args.query}")
        print("=" * 60)
        if len(hasil) == 0:
            print("Tidak ada dokumen ditemukan.")
        else:
            for i, doc in enumerate(hasil, 1):
                snippet = " ".join(docs[doc])[:120]
                print(f"{i}. {doc:<25} | {snippet}")
        print(f"\nTotal hasil: {len(hasil)} dokumen.")

    # === VSM ===
    elif args.model == "vsm":
        results, vectorizer = vsm_search(args.query, docs, args.k)
        print(f"\nModel: VECTOR SPACE MODEL")
        print(f"Query: {args.query}")
        print("=" * 60)
        for rank, (doc, score) in enumerate(results, 1):
            snippet = " ".join(docs[doc])[:120]
            feature_array = np.array(vectorizer.get_feature_names_out())
            query_terms = args.query.lower().split()
            top_terms = [term for term in query_terms if term in feature_array]
            print(f"{rank}. {doc:<25} | cosine={score:.4f} | {snippet}")
            print(f"   → Top terms match: {', '.join(top_terms) if top_terms else '-'}")
