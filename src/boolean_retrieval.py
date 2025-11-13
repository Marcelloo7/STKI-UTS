import os
import re
from collections import defaultdict

# === 1. LOAD DOKUMEN YANG SUDAH DIPROSES ===
def load_processed_docs(folder):
    docs = {}
    if not os.path.exists(folder):
        print(f"Folder {folder} tidak ditemukan!")
        return docs

    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                docs[filename] = f.read().split()
    return docs


# === 2. BANGUN VOCABULARY & INVERTED INDEX ===
def build_inverted_index(docs):
    inverted_index = defaultdict(set)
    for doc_id, tokens in docs.items():
        for token in tokens:
            inverted_index[token].add(doc_id)
    return inverted_index


# === 3. INCIDENCE MATRIX (OPSIONAL, UNTUK DOKUMENTASI) ===
def build_incidence_matrix(docs, vocab):
    matrix = {}
    for term in vocab:
        matrix[term] = [1 if term in tokens else 0 for tokens in docs.values()]
    return matrix


# === 4. EVALUASI BOOLEAN QUERY ===
def boolean_query(query, index, all_docs):
    terms = re.findall(r'\w+|AND|OR|NOT', query.upper())

    result_stack = []
    operator_stack = []

    def apply_operator(op):
        if op == 'AND':
            b = result_stack.pop()
            a = result_stack.pop()
            result_stack.append(a & b)
        elif op == 'OR':
            b = result_stack.pop()
            a = result_stack.pop()
            result_stack.append(a | b)
        elif op == 'NOT':
            a = result_stack.pop()
            result_stack.append(all_docs - a)

    for term in terms:
        if term in ('AND', 'OR', 'NOT'):
            operator_stack.append(term)
        else:
            term = term.lower()
            result_stack.append(set(index.get(term, set())))

            # Jika sudah ada operator, langsung terapkan
            if operator_stack:
                op = operator_stack.pop()
                apply_operator(op)

    return result_stack.pop() if result_stack else set()


# === 5. HITUNG PRECISION & RECALL ===
def evaluate(query, retrieved, relevant):
    tp = len(retrieved & relevant)
    fp = len(retrieved - relevant)
    fn = len(relevant - retrieved)

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    return precision, recall


# === 6. MAIN PROGRAM ===
if __name__ == "__main__":
    folder = "data/processed"  # pastikan ini benar
    docs = load_processed_docs(folder)

    print(f"Total dokumen: {len(docs)}")

    if not docs:
        print("⚠️ Tidak ada dokumen yang ditemukan. Jalankan preprocess.py dulu.")
        exit()

    inverted_index = build_inverted_index(docs)
    all_docs = set(docs.keys())

    # tampilkan contoh index
    print("\n=== Contoh Inverted Index ===")
    for term in list(inverted_index.keys())[:5]:
        print(f"{term} -> {list(inverted_index[term])}")

    # === 7. QUERY UJI COBA (DISUSUN ULANG AGAR KATA ADA DI KORPUS) ===
    queries = [
        ("pedang AND hutan", {"buku_fantasi.txt"}),
        ("cinta OR motivasi", {"buku_romansa.txt", "buku_motivasi.txt"}),
        ("NOT horor", set(all_docs) - {"buku_horor.txt"}),
    ]

    print("\n=== HASIL PENGUJIAN QUERY ===")
    for q, gold in queries:
        result = boolean_query(q, inverted_index, all_docs)
        precision, recall = evaluate(q, result, gold)

        print(f"\nQuery: {q}")
        print(f"→ Ditemukan: {list(result)}")
        print(f"→ Relevan (gold): {list(gold)}")
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")

        print("Penjelasan logika Boolean:")
        if "AND" in q:
            print("Operator AND → irisan (∩) antara dua set dokumen.")
        elif "OR" in q:
            print("Operator OR → gabungan (∪) dua set dokumen.")
        elif "NOT" in q:
            print("Operator NOT → komplemen dari dokumen yang mengandung kata tersebut.")
