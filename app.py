import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Cargar el modelo entrenado
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
model = tf.keras.models.load_model(MODEL_PATH)

# Cargar etiquetas
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

def preprocess_frame(frame):
    """Preprocesa el frame para que sea compatible con el modelo."""
    img = cv2.resize(frame, (224, 224))  # Redimensionar a tamaño compatible
    img = img.astype(np.float32) / 255.0  # Normalización
    img = np.expand_dims(img, axis=0)  # Agregar batch dimension
    return img

def predict(image):
    """Realiza la predicción sobre un frame."""
    processed_img = preprocess_frame(image)
    predictions = model.predict(processed_img)
    class_index = np.argmax(predictions)
    return labels[class_index], predictions[0][class_index]

# Configuración de la app
st.title("Detección de Gafas en Tiempo Real")
st.text("Usa la webcam para detectar si llevas gafas o no")

# Captura de video con OpenCV
video_capture = cv2.VideoCapture(0)
frame_placeholder = st.empty()

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        st.error("No se pudo acceder a la cámara")
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    prediction_label, confidence = predict(frame)
    
    # Mostrar la predicción en el frame
    cv2.putText(frame, f"{prediction_label} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Mostrar el frame en Streamlit
    frame_placeholder.image(frame, channels="RGB")