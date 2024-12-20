import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gdown
import os
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.io as io

# Mostrar un mensaje mientras se carga el modelo
placeholder = st.empty()
placeholder.text("Cargando modelo, por favor espere...")

# Diccionario de enlaces de Google Drive para modelos
model_drive_links = {
    "VGG": "<URL_DEL_MODELO_VGG>",  # Reemplaza con el enlace de Google Drive
    "ResNet50": "<URL_DEL_MODELO_RESNET50>",  # Reemplaza con el enlace de Google Drive
    "Inception": "https://drive.google.com/uc?id=1iHJK3UA1gHDfMXB_EvhxDwOjbluUsMFk"
}

# Función para descargar y cargar el modelo desde Google Drive
def load_model_from_drive(model_name):
    model_link = model_drive_links.get(model_name)
    if not model_link:
        st.error("Enlace del modelo no encontrado.")
        return None

    model_filename = f"{model_name}.keras"
    model_path = os.path.join("/tmp", model_filename)

    if not os.path.exists(model_path):
        st.write(f"Descargando el modelo {model_name} desde Google Drive...")
        gdown.download(model_link, model_path, quiet=False)

    st.write(f"Cargando el modelo {model_name}...")
    return tf.keras.models.load_model(model_path)

# Placeholder para el modelo
model = None
placeholder.empty()

#Función mostrar img

root = './real_vs_fake/real-vs-fake/'
train_dir = root + 'train'
val_dir = root + 'valid'
test_dir = root + 'test'
class_names = ['fake', 'real']
num_images = 5

# Función para preprocesar la imagen
def preprocess_image(image):
    image = image.resize((128, 128))  # Ajustar el tamaño como en el entrenamiento
    image_array = np.array(image) / 255.0  # Normalizar los píxeles
    return image_array

# Función para aplicar LIME
def apply_lime(image_for_lime, model):
    explainer = lime_image.LimeImageExplainer()

    def predict_fn(images):
        return model.predict(np.array(images))

    explanation = explainer.explain_instance(
        image_for_lime,  # (128, 128, 3)
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        label, positive_only=True, hide_rest=False, num_features=5
    )
    return temp, mask

# Crear menú en el sidebar
with st.sidebar:
    selected2 = option_menu("Menú", ["Explicación", "Clasificador", "Modelos"], 
        icons=['house', 'cloud-upload', "list-task", 'gear', 'camera'], 
        menu_icon="cast", default_index=0)

if selected2 == "Clasificador":
    st.title("Clasificación de Imágenes: Real vs Deepfake")
    
    # Seleccionar modelo
    option = st.sidebar.selectbox(
        "Selecciona el modelo a utilizar:",
        ("VGG", "ResNet50", "Inception")
    )

    # Mostrar descripción del modelo seleccionado
    st.write(f"### Descripción detallada del modelo {option}")
    st.write(model_details[option]["detailed_description"])
    st.image(model_details[option]["image"], caption=option, use_column_width=True)

    # Cargar el modelo correspondiente desde Google Drive
    model = load_model_from_drive(option)

    # Sidebar para seleccionar modo de entrada
    st.sidebar.title("Opciones de entrada")
    st.sidebar.write("Seleccione cómo cargar la imagen")
    input_mode = st.sidebar.radio("Selecciona el modo de entrada:", ("Subir imagen", "Tomar foto"))

    image = None

    if input_mode == "Subir imagen":
        # Cargar imagen del usuario
        uploaded_file = st.sidebar.file_uploader("Sube tu imagen aquí", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Mostrar la imagen cargada
            image = Image.open(uploaded_file)
            st.sidebar.image(image, caption="Imagen cargada", use_column_width=True)

    elif input_mode == "Tomar foto":
        if st.sidebar.button("Tomar foto"):
            st.sidebar.write("Esta función estará disponible próximamente")

    if image is not None:
        # Preprocesar la imagen
        image_array = preprocess_image(image)
        preprocessed_image = np.expand_dims(image_array, axis=0)  # Expandir dimensiones para el modelo

        # Realizar la predicción
        predictions = model.predict(preprocessed_image)

        # Obtener el resultado
        class_names = ["Real", "Deepfake"]
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)
        st.write("### Clasificación de la imagen")
        # Mostrar el resultado
        st.write(f"Predicción: **{predicted_class}**")

        # Aplicar LIME
        st.write("Generando explicaciones con LIME, por favor espere...")
        image_for_lime = image_array  # Imagen procesada para LIME
        temp, mask = apply_lime(image_for_lime, model)

        # Visualizar la explicación
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            ax.imshow(mark_boundaries(temp, mask))
            ax.set_title("LIME: Regiones Importantes")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(8, 8))
            heatmap = mask.astype(float)
            ax.imshow(image_array)
            ax.imshow(heatmap, cmap='jet', alpha=0.5)
            cbar = plt.colorbar(ax.imshow(heatmap, cmap='jet', alpha=0.5), ax=ax)
            cbar.set_label('Importancia de las regiones', rotation=270, labelpad=20)
            ax.set_title("LIME Heatmap: Regiones Importantes")
            ax.axis('off')
            st.pyplot(fig)
    
elif selected2 == "Modelos":
    st.title("Modelos y Resultados")
    
    option2 = st.selectbox(
        "Selecciona el modelo para ver su rendimiento:",
        ("VGG", "ResNet50", "Inception")
    )

elif selected2 == "Explicación":
    st.title("DeepFake Face Detection Project")
    st.write("## Dataset:")
    st.write("Este conjunto de datos contiene 70k imágenes de caras reales y 70k imágenes de caras generadas artificialmente mediante GAN. GAN es una red neuronal\
             generativa que permite la creación de datos sintéticos a partir de datos reales basándose en sus dos principales componentes: Generador y Discriminador.\
             El hecho de que las clases del dataset estén balanceadas nos facilitará mucho el trabajo. **¿Un humano puede diferenciar las siguientes imágenes?**")
    st.image("dataset_img.png")
    
    st.write("## Algunas aplicaciones en el mundo real:")
    st.write("""
- **Estos modelos podrían aplicarse tanto en las redes sociales como en las noticias** para evitar la publicación de deepfakes de terceros publicados sin consentimiento y evitar fraudes.
- Hoy en día, se está poniendo muy de moda generar imágenes y videos de personas públicas con las IA, lo cual plantea desafíos éticos y legales. Por ejemplo:
""")
    st.image("Elon_musk.png")

    st.write("## Pasos realizados:")
    st.write("### 1) Análisis EDA")
    st.write("Nuestra principal preocupación es si los datos realmente están balanceados...")
    st.image("eda.png")
    st.write("")
    st.write("""
             ### 2) Preparación del dataset
            Las imágenes originales tenían un tamaño de **256x256** en RGB 0-255. Para facilitar el entrenamiento de los modelos hemos reducido el tamaño de las imágenes\
             a 128x128 y normalizado los 3 canales para que estén entre **0-1**
            """)
    st.image("pixeles.png")
    st.write("""
            Analizando la gráfica podemos concluir que **hay una distribución suave y uniforme en los píxeles** intermedios que facilitarán el aprendizaje\ 
            de nuestros modelos. Sin embargo también hay 2 outliers que serían los píxeles normalizados con un valor de 0 y 1... los cuales en el caso de un\
            mal rendimiento de los modelos deberemos preprocesar.
            """)
    st.write("")
 
    st.write("### 3) Entrenamiento de modelos")
    st.write("""Una vez hecho el análisis EDA y el preprocesamiento de las imágenes pasaremos al entrenamiento de los modelos de clasificación.\
             Usaremos el transfer learning para importar estos modelos ya entrenados con las imágenes del dataset de ImageNet y congelaremos sus pesos. \
             Únicamente cambiaremos la parte del clasificador añadiendo capas densas y cambiando la función de activación final para nuestra clasificación.
            """)
    st.write("- ResNet50")
    st.write("- Inception v3")