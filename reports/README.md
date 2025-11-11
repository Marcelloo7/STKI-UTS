# STKI UTS Project Template

Struktur repository untuk Ujian Tengah Semester (UTS) mata kuliah **Sistem Temu Kembali Informasi (STKI)**.

## 📁 Struktur Folder
```
stki-uts-<nim>-<nama>/
├── data/
│   ├── raw/          -> Dataset asli (.txt)
│   └── processed/    -> Hasil preprocessing
│
├── src/              -> Kode sumber Python
│   ├── preprocess.py -> Soal 02
│   ├── boolean_ir.py -> Soal 03
│   ├── vsm_ir.py     -> Soal 04
│   ├── search.py     -> Soal 05
│   └── eval.py       -> Evaluasi hasil
│
├── notebooks/        -> Notebook uji setiap soal
│   ├── Soal_01_Konsep_STKI.ipynb
│   ├── Soal_02_Preprocessing.ipynb
│   ├── Soal_03_Indexing.ipynb
│   ├── Soal_04_VSM.ipynb
│   └── Soal_05_Retrieval.ipynb
│
├── reports/          -> Laporan PDF dan README
│   ├── Soal_01_Konsep_STKI.pdf
│   ├── laporan_akhir_STKI.pdf
│   └── readme.md
│
└── requirements.txt  -> Daftar library Python
```

## 📌 Petunjuk
- Simpan semua file hasil preprocessing di `data/processed/`.
- Simpan semua kode Python tiap soal di `src/`.
- Simpan laporan PDF atau esai di `reports/`.
- Jalankan proyek di `notebooks/` untuk dokumentasi uji.

---
🧩 **Nama Template:** STKI UTS Project
