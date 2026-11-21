import streamlit as st
from PIL import Image
from visionreader.inference import ImageReader, read_text

st.set_page_config(page_title="VisionReader AI", page_icon="🖼️")
st.title("🖼️ VisionReader AI")
st.write("Klasifikasi objek dan pembacaan tulisan dalam gambar.")
st.caption("Berjalan lokal di CPU. Model awal mengenali kategori objek dominan, bukan seluruh detail adegan.")

@st.cache_resource
def load_reader():
    return ImageReader()

uploaded = st.file_uploader("Pilih gambar", type=["png", "jpg", "jpeg"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Gambar pilihan", use_container_width=True)
    with st.spinner("Menganalisis gambar..."):
        try:
            reader = load_reader()
            st.subheader("Prediksi objek")
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
            st.info("Pastikan instalasi dependensi selesai dan internet tersedia saat unduhan bobot pertama.")
else:
    st.info("Unggah gambar untuk memulai.")
