# Vision Transformer Comparison - Cara Menjalankan

Repository resmi tugas Deep Learning - Prodi Teknik Informatika ITERA

## ⚙️ Instalasi

```bash
# 1. Clone repository (PASTIKAN PUBLIC)
git clone https://github.com/gabriellarumapea/VisionTransformer-Comparison.git
cd VisionTransformer-Comparison

# 2. Buat virtual environment (opsional tapi direkomendasikan)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt
```
## 🚀 Cara Menjalankan

Jalankan :
```bash
python main.py --epochs 10 --batch-size 32
```
## Metode Manual (Step-by-Step)

### 1. Download Dataset
Dataset dapat diunduh dengan menjalankan script 'download_dataset.py', atau dapat juga mengakses langsung di Google Drive (https://drive.google.com/file/d/1o3rl6Ap4QjxM5-C9WtiJMIgIwWVFK_ZO/view?usp=drive_link)
```bash
python download_dataset.py
```

### 2. Analisis Dataset (opsional, untuk cek distribusi data):
```bash
python dataset_analysis.py
```

### 3. Training Model:
```bash
python main.py
```
