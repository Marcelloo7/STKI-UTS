Mini Search Engine - UTS STKI

Nama: Atanasius Marcello
NIM: A11.2023.15390
Mata Kuliah: Sistem Temu Kembali Informasi (STKI)

Tujuan Proyek
Proyek ini bertujuan untuk membangun sebuah mini search engine sederhana yang menerapkan konsep utama dalam Information Retrieval (IR):
Boolean Retrieval Model — pencarian berbasis logika menggunakan operator AND, OR, NOT.
Vector Space Model (VSM) — pencarian berbasis cosine similarity dengan pembobotan TF-IDF.
Term Weighting Comparison — membandingkan performa antara TF-IDF, TF-IDF Sublinear, dan BM25.
Chat Interface (VSM-based) — sistem percakapan berbasis VSM.
Sistem ini juga dievaluasi menggunakan metrik Precision, Recall, F1-score, MAP@k, dan nDCG@k.

Struktur Direktori

stki-uts-A11.2023.15390-AtanasiusMarcello/
│

├── data/

│ ├── raw/ → Dokumen asli sebelum diproses

|

│ └── processed/ → Dokumen hasil preprocessing (tokenized & cleaned)

│

├── src/

│ 
   ├── preprocess.py → Membersihkan dan memproses teks
│ 
   ├── boolean_retrieval.py → Implementasi Boolean IR dan Inverted Index
│ 
   ├── vector_space_model.py → Model VSM menggunakan TF-IDF
│ 
   ├── weighting_and_eval.py → Perbandingan TF-IDF, Sublinear TF, dan BM25
│ 
   ├── search_engine.py → Search Engine CLI (Boolean & VSM)
│ 
   └── eval.py → Evaluasi metrik IR (Precision, Recall, MAP, nDCG)
│

├── app/

│ 
   └── chat.py → Chat Interface (interaktif)
│

├── reports/

│ 
   ├── laporan.pdf → Laporan akhir (6–10 halaman)
│ 
   └── readme.md → File README ini
   
│

└── requirements.txt → Daftar dependensi Python

Cara Menjalankan Proyek
1. Instalasi

Pastikan Python 3.9+ sudah terpasang, lalu jalankan:
pip install -r requirements.txt

Jika file requirements.txt belum ada, bisa gunakan:
pip install numpy pandas scikit-learn rank-bm25 matplotlib

2. Preprocessing Data

Jalankan:
python src/preprocess.py
File hasil bersih akan tersimpan di data/processed.

3. Boolean Retrieval

Jalankan perintah:
python src/search_engine.py --model boolean --query "pedang AND hutan"
Contoh hasil:
Model: BOOLEAN RETRIEVAL
Query: pedang AND hutan
============================================================
1. buku_fantasi.txt | anak lakilaki bernama arka menemukan pedang ajaib tersembunyi hutan terlarang...
Total hasil: 1 dokumen.

4. Vector Space Model (VSM)

python src/search_engine.py --model vsm --query "cinta motivasi" --k 3
Contoh hasil:
Model: VECTOR SPACE MODEL
Query: cinta motivasi
============================================================
1. buku_romansa.txt | cosine=0.3015 | kisah cinta insan terhalang jarak...
   → Top terms match: cinta
2. buku_motivasi.txt | cosine=0.2123 | buku mengajarkan berpikir positif...

5. Chat Interface

python app/chat.py
Contoh interaksi:
🤖 Mini Search Assistant (VSM-based)
🗨️  Query: pedang hutan
🔍 Berdasarkan pencarian untuk 'pedang hutan', berikut 3 dokumen teratas:
1. buku_fantasi.txt (cosine: 0.349)
2. buku_petualangan.txt (cosine: 0.212)
3. buku_sains.txt (cosine: 0.000)
🧠 Sistem menampilkan hasil paling relevan berdasarkan kesamaan deskripsi teks.

Evaluasi dan Analisis
Evaluasi dilakukan pada tiga model pembobotan:

TF-IDF Normal
TF-IDF Sublinear
BM25 (opsional bonus)

Metrik yang digunakan:
Precision, Recall, F1-score, MAP@k, nDCG@k

Analisis:
TF-IDF Sublinear dan BM25 memberikan hasil ranking yang lebih stabil.
Boolean lebih presisi untuk pencarian kata exact, namun kurang fleksibel.
VSM unggul untuk query alami dan fleksibel.

Kesimpulan
Boolean IR dan VSM berhasil diimplementasikan dengan baik.
Evaluasi sistem menunjukkan hasil yang konsisten antar skema pembobotan.
Mini search engine ini dapat dikembangkan menjadi aplikasi berbasis web (Streamlit/Flask).
