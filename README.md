# VisionReader AI

AI pembaca gambar ringan untuk Windows dengan CPU dan RAM terbatas. Versi awal menggabungkan:
- **Klasifikasi gambar** dengan MobileNetV3-Small (ImageNet, 1000 kategori).
- **OCR opsional** untuk membaca tulisan pada gambar melalui Tesseract.
- **Fine-tuning** MobileNetV3-Small dengan folder dataset milik sendiri.

> Catatan kemampuan: ini adalah fondasi pembaca gambar, bukan model yang memahami semua isi gambar seperti manusia. Klasifikasi ImageNet hanya memberi label objek dominan; OCR membaca teks yang terlihat. Pemahaman adegan, relasi antarbenda, dan deskripsi bebas membutuhkan model vision-language yang lebih besar.

## Spesifikasi yang disarankan
- Windows 10/11, Python 3.10 atau 3.11 64-bit
- CPU saja; RAM 4 GB: tutup aplikasi lain saat menjalankan
- Ruang disk beberapa GB untuk PyTorch dan bobot model
- Koneksi internet pertama kali untuk mengunduh bobot MobileNetV3

## Instalasi (PowerShell)
Buka PowerShell di folder repository:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Jika PowerShell memblokir aktivasi, jalankan:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## OCR (opsional)
1. Instal Tesseract OCR untuk Windows dari https://github.com/UB-Mannheim/tesseract/wiki
2. Pastikan bahasa Indonesia dan Inggris tersedia saat instalasi.
3. Jika Tesseract tidak terdeteksi otomatis, atur variabel lingkungan `TESSERACT_CMD` ke lokasi `tesseract.exe`, contoh:
   ```powershell
   $env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```

## Menjalankan aplikasi
```powershell
streamlit run app.py
```
Browser akan membuka aplikasi lokal. Pilih JPG, JPEG, atau PNG. Hasil klasifikasi menampilkan lima prediksi teratas. OCR ditampilkan jika Tesseract terpasang.

## Melatih dengan dataset sendiri
Susun dataset seperti ini (nama folder menjadi nama kelas):

```text
data/
  train/
    kucing/
      gambar1.jpg
    anjing/
      gambar2.jpg
```

Jalankan:
```powershell
python train.py --data_dir data/train --epochs 3 --batch_size 4
```

Hasil tersimpan di `models/custom_mobilenetv3.pth`. Untuk komputer RAM 4 GB, mulai dari 2–3 epoch dan batch size 2 atau 4. Dataset harus memiliki minimal dua folder kelas dan beberapa gambar per kelas.

Setelah model khusus tersedia, aplikasi otomatis memakainya. Hapus atau pindahkan file model tersebut untuk kembali ke model ImageNet bawaan.

## Struktur proyek
```text
visionreader-ai/
  app.py
  train.py
  requirements.txt
  visionreader/
    __init__.py
    inference.py
  data/train/       # dataset lokal (tidak diunggah)
  models/           # bobot hasil training (tidak diunggah)
```

## Privasi dan batasan
Gambar diproses secara lokal di komputer Anda oleh aplikasi ini. Jangan mengunggah foto sensitif ke repository. Dataset dan bobot model dikecualikan melalui `.gitignore`. Akurasi bergantung pada kualitas gambar dan kemiripan kelas dataset dengan data yang dilatih.

## Lisensi
MIT


## Deteksi dan menghitung objek

Aplikasi juga memakai MobileNet-SSD melalui OpenCV DNN untuk mendeteksi objek dan menghitung jumlah tiap kategori. Pada penggunaan pertama, file model diunduh otomatis ke `models/detection/` (memerlukan internet). Model ini mengenali 20 kategori Pascal VOC, bukan semua benda di dunia. Objek kecil, tertutup, atau kategori di luar daftar dapat terlewat. Model deteksi berasal dari [chuanqi305/MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD), dilatih pada VOC0712.

Instal dependensi tambahan jika memperbarui proyek lama:

```powershell
python -m pip install opencv-python-headless==4.10.0.84
```
