import numpy as np
import tensorflow as tf
from pathlib import Path
import streamlit as st

# Judul aplikasi
st.title("Klasifikasi Citra Anjing")

# Fungsi prediksi
def predict(uploaded_image):
    # Daftar kelas
    class_names = ["Boxer", "Dachshund", "Golden_Retriever", "Poodle", "Rottweiler"]

    # Muat dan preprocess citra
    img = tf.keras.utils.load_img(uploaded_image, target_size=(224, 224))  # Pastikan ukuran sesuai dengan model
    img = tf.keras.utils.img_to_array(img) / 255.0  # Normalisasi
    img = np.expand_dims(img, axis=0)  # Tambahkan dimensi batch

    # Muat model
    model_path = Path(__file__).parent / "model.h5"
    model = tf.keras.models.load_model(model_path)

    # Prediksi
    output = model.predict(img)
    score = tf.nn.softmax(output[0])  # Hitung probabilitas
    return class_names[np.argmax(score)], 100 * np.max(score)  # Prediksi label dan confidence

# Komponen file uploader untuk banyak file
uploads = st.file_uploader("Unggah citra untuk mendapatkan hasil prediksi", type=["png", "jpg"], accept_multiple_files=True)

# Tombol prediksi
if st.button("Predict", type="primary"):
    if uploads:
        st.subheader("Hasil prediksi:")

        for upload in uploads:
            # Tampilkan setiap citra yang diunggah
            st.image(upload, caption=f"Citra yang diunggah: {upload.name}", use_column_width=True)
            
            with st.spinner(f"Memproses citra {upload.name} untuk prediksi..."):
                # Panggil fungsi prediksi
                try:
                    label, confidence = predict(upload)
                    st.write(f"**{upload.name}** - Label: **{label}**, Confidence: **{confidence:.5f}%**")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses {upload.name}: {e}")
    else:
        st.error("Unggah setidaknya satu citra terlebih dahulu!")
