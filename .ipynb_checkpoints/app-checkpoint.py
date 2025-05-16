import streamlit as st
from ultralytics import YOLO
import os
from PIL import Image

# Título de la app
st.title("Detección de Objetos con YOLO")

# Subida de imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Mostrar imagen original
    st.image(uploaded_file, caption="Imagen original", use_container_width =True)

    # Guardar temporalmente la imagen
    with open("input_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Cargar modelo YOLO
    model = YOLO("yolo11n.pt")  # Asegúrate de tener este archivo en la misma carpeta

    # Ejecutar predicción
    results = model(source="input_image.png", conf=0.4, save=True)

    # Obtener ruta del archivo resultado
    save_dir = results[0].save_dir

    # Buscar la imagen resultante en esa carpeta (la primera .jpg o .png encontrada)
    result_img_path = None
    for file in os.listdir(save_dir):
        if file.endswith((".jpg", ".jpeg", ".png")):
            result_img_path = os.path.join(save_dir, file)
            break

    if result_img_path and os.path.exists(result_img_path):
        st.image(Image.open(result_img_path), caption="Resultado con objetos detectados", use_container_width =True)
    else:
        st.error("No se encontró la imagen resultante.")
