import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

# === 1. LOAD DOKUMEN ===
def load_documents(folder):
    docs = {}
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                docs[filename] = f.read()
    return docs


# === 2. TF-IDF SEARCH ===
def tfidf_search(query, docs, k=3):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs.values())
    query_vec = vectorizer.transform([query])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    ranked_indices = cosine_sim.argsort()[::-1]
    doc_names = list(docs.keys())
    results = [doc_names[i] for i in ranked_indices[:k]]
    return results


# === 3. BM25 SEARCH ===
def bm25_search(query, docs, k=3):
    tokenized_corpus = [doc.split() for doc in docs.values()]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = np.argsort(scores)[::-1]
    doc_names = list(docs.keys())
    results = [doc_names[i] for i in ranked_indices[:k]]
    return results


# === 4. METRIK EVALUASI ===
def precision_recall_f1(retrieved, relevant):
    retrieved_set = set(retrieved)
    relevant_set = set(relevant)
    tp = len(retrieved_set & relevant_set)
    precision = tp / len(retrieved) if retrieved else 0
    recall = tp / len(relevant) if relevant else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1


def map_at_k(retrieved, relevant, k=3):
    relevant_set = set(relevant)
    hits = 0
    sum_precisions = 0
    for i, doc in enumerate(retrieved[:k], start=1):
        if doc in relevant_set:
            hits += 1
            sum_precisions += hits / i
    return sum_precisions / min(len(relevant_set), k) if relevant_set else 0


def ndcg_at_k(retrieved, relevant, k=3):
    relevant_set = set(relevant)
    dcg = sum([1 / np.log2(i + 2) for i, doc in enumerate(retrieved[:k]) if doc in relevant_set])
    ideal_dcg = sum([1 / np.log2(i + 2) for i in range(min(len(relevant_set), k))])
    return dcg / ideal_dcg if ideal_dcg > 0 else 0


# === 5. VISUALISASI HASIL ===
def plot_comparison(metrics_summary):
    labels = list(metrics_summary.keys())
    metrics = ["Precision", "Recall", "F1", "MAP@3", "nDCG@3"]
    values = np.array([metrics_summary[m] for m in labels])

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, values[0], width, label=labels[0])
    plt.bar(x + width / 2, values[1], width, label=labels[1])

    plt.xticks(x, metrics)
    plt.ylim(0, 1.1)
    plt.ylabel("Score")
    plt.title("Perbandingan Metrik Evaluasi: TF-IDF vs BM25")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


# === 6. MAIN ===
if __name__ == "__main__":
    folder = "data/processed"
    docs = load_documents(folder)
    print(f"Total dokumen: {len(docs)}")

    if not docs:
        print("⚠️ Tidak ada dokumen! Pastikan sudah ada hasil preprocessing.")
        exit()

    # === Gold Set ===
    queries = {
        "pedang hutan": ["buku_fantasi.txt"],
        "cinta motivasi": ["buku_romansa.txt", "buku_motivasi.txt"],
        "ilmu sains pengetahuan": ["buku_sains.txt"],
    }

    models = {
        "TF-IDF": tfidf_search,
        "BM25": bm25_search,
    }

    metrics_summary = {}

    # === Evaluasi Tiap Model ===
    for model_name, search_func in models.items():
        print("\n" + "=" * 70)
        print(f"🔹 Evaluasi skema: {model_name}")
        all_prec, all_rec, all_f1, all_map, all_ndcg = [], [], [], [], []

        for query, gold in queries.items():
            results = search_func(query, docs, k=3)
            precision, recall, f1 = precision_recall_f1(results, gold)
            mapk = map_at_k(results, gold)
            ndcgk = ndcg_at_k(results, gold)

            all_prec.append(precision)
            all_rec.append(recall)
            all_f1.append(f1)
            all_map.append(mapk)
            all_ndcg.append(ndcgk)

            print(f"\nQuery: {query}")
            print(f"Top-3: {results}")
            print(f"Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}, MAP@3={mapk:.2f}, nDCG@3={ndcgk:.2f}")

        avg_metrics = [
            np.mean(all_prec),
            np.mean(all_rec),
            np.mean(all_f1),
            np.mean(all_map),
            np.mean(all_ndcg),
        ]
        metrics_summary[model_name] = avg_metrics

        print("-" * 70)
        print(f"Rata-rata: Precision={avg_metrics[0]:.2f}, Recall={avg_metrics[1]:.2f}, "
              f"F1={avg_metrics[2]:.2f}, MAP@3={avg_metrics[3]:.2f}, nDCG@3={avg_metrics[4]:.2f}")

    # === Plot Perbandingan TF-IDF vs BM25 ===
    print("\n📊 Menampilkan grafik perbandingan metrik ...")
    plot_comparison(metrics_summary)
