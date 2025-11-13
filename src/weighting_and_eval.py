import os
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
                content = f.read().strip()
                if content:
                    docs[filename] = content
    return docs


# === 2. TF-IDF normal ===
def build_tfidf(docs):
    vectorizer = TfidfVectorizer(norm="l2")
    tfidf_matrix = vectorizer.fit_transform(docs.values())
    return vectorizer, tfidf_matrix


# === 3. TF-IDF Sublinear (log scaling) ===
def build_tfidf_sublinear(docs):
    vectorizer = TfidfVectorizer(norm="l2", sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(docs.values())
    return vectorizer, tfidf_matrix


# === 4. Ranking dokumen berdasarkan cosine similarity ===
def rank_documents(vectorizer, tfidf_matrix, docs, query, k=3):
    query_vec = vectorizer.transform([query])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    doc_names = list(docs.keys())
    ranked_indices = cosine_sim.argsort()[::-1]
    top_docs = [(doc_names[i], cosine_sim[i]) for i in ranked_indices[:k]]
    return top_docs


# === 5. Evaluation metrics ===
def precision_at_k(predicted, gold):
    relevant = sum(1 for doc, _ in predicted if doc in gold)
    return relevant / len(predicted) if predicted else 0.0


def map_at_k(predicted_lists, gold_lists):
    ap_scores = []
    for predicted, gold in zip(predicted_lists, gold_lists):
        hits = 0
        sum_precisions = 0
        for i, (doc, _) in enumerate(predicted):
            if doc in gold:
                hits += 1
                sum_precisions += hits / (i + 1)
        ap = sum_precisions / len(gold) if gold else 0
        ap_scores.append(ap)
    return np.mean(ap_scores)


# === 6. MAIN PROGRAM ===
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(base_dir, "../data/processed")

    docs = load_documents(data_folder)
    print(f"Total dokumen: {len(docs)}\n")

    if not docs:
        print("[!] Tidak ada dokumen yang ditemukan di data/processed/.")
        exit()

    # Dataset pengujian (query dan gold relevance)
    tests = [
        {"query": "pedang hutan", "gold": ["buku_fantasi.txt"]},
        {"query": "cinta motivasi", "gold": ["buku_romansa.txt", "buku_motivasi.txt"]},
        {"query": "ilmu sains pengetahuan", "gold": ["buku_sains.txt"]},
    ]

    models = {
        "TF-IDF Normal": build_tfidf,
        "TF-IDF Sublinear": build_tfidf_sublinear,
    }

    eval_results = {}

    for model_name, builder in models.items():
        print("=" * 70)
        print(f"🔹 Evaluasi Model: {model_name}")
        vectorizer, tfidf_matrix = builder(docs)

        precisions = []
        predicted_lists = []
        gold_lists = []

        for test in tests:
            query = test["query"]
            gold = test["gold"]

            top_docs = rank_documents(vectorizer, tfidf_matrix, docs, query, k=3)
            prec = precision_at_k(top_docs, gold)
            precisions.append(prec)
            predicted_lists.append(top_docs)
            gold_lists.append(gold)

            print(f"\nQuery: {query}")
            for rank, (doc, score) in enumerate(top_docs, 1):
                snippet = docs[doc][:100].replace("\n", " ")
                print(f"{rank}. {doc:<25} | cosine={score:.4f} | {snippet}")
            print(f"Precision@3: {prec:.2f}")

        mean_prec = np.mean(precisions)
        mapk = map_at_k(predicted_lists, gold_lists)
        eval_results[model_name] = {"Precision@3": mean_prec, "MAP@3": mapk}

    print("\n" + "=" * 70)
    print("📊 Ringkasan Evaluasi Model:")
    for model, scores in eval_results.items():
        print(f"{model:<20} | Precision@3={scores['Precision@3']:.2f} | MAP@3={scores['MAP@3']:.2f}")
