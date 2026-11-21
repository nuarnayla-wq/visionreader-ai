import streamlit as st
from PIL import Image
from visionreader.inference import ImageReader, ObjectCounter, read_text

st.set_page_config(page_title="VisionReader AI", page_icon="🖼️")
st.title("🖼️ VisionReader AI")
st.write("Klasifikasi gambar, deteksi dan hitung objek, serta pembacaan tulisan.")
st.caption("Berjalan di CPU. Penghitung objek memakai MobileNet-SSD dengan 20 kategori Pascal VOC.")

@st.cache_resource
def load_reader():
    return ImageReader()

@st.cache_resource
def load_counter():
    return ObjectCounter()

uploaded = st.file_uploader("Pilih gambar", type=["png", "jpg", "jpeg"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Gambar pilihan", use_column_width=True)
    try:
        with st.spinner("Menganalisis dan menghitung objek..."):
            counter = load_counter()
            detections = counter.detect(image)
            summary = counter.summarize(detections)
            st.subheader("Deteksi dan jumlah objek")
            st.metric("Total objek terdeteksi", summary["total"])
            if detections:
                st.image(counter.draw(image, detections), caption="Kotak menunjukkan lokasi objek", use_column_width=True)
                st.write("**Jumlah per jenis objek**")
                for label, count in sorted(summary["counts"].items()):
                    st.write(f"- {label.capitalize()}: **{count}**")
                with st.expander("Detail deteksi"):
                    for item in detections:
                        st.write(f"{item['label_id']} — keyakinan {item['confidence']:.1%}")
            else:
                st.info("Belum ada objek dari 20 kategori model yang terdeteksi.")
        with st.spinner("Menjalankan klasifikasi gambar..."):
            reader = load_reader()
            st.subheader("Prediksi kategori gambar")
            st.caption("Model khusus" if reader.custom else "Model ImageNet bawaan")
            for result in reader.predict(image):
                st.write(f"**{result['label']}** — {result['confidence']:.1%}")
                st.progress(result["confidence"])
        st.subheader("Tulisan dalam gambar (OCR)")
        text = read_text(image)
        if text.startswith("OCR belum tersedia:"):
            st.info(text)
        else:
            st.text_area("Hasil OCR", text or "Tidak ada tulisan terdeteksi.", height=160)
    except Exception as exc:
        st.error(f"Gagal memproses gambar: {exc}")
        st.info("Periksa koneksi internet saat unduhan model deteksi pertama dan pastikan semua dependensi sudah terpasang.")
else:
    st.info("Unggah gambar untuk memulai.")
