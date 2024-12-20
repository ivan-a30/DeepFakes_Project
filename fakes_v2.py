import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import skimage.io as io

# Mostrar un mensaje mientras se carga el modelo
placeholder = st.empty()
placeholder.text("Cargando modelo, por favor espere...")

# Diccionario de rutas de modelos
model_paths = {
    "VGG": "model.keras",
    "ResNet50": "model.keras",
    "Inception": "model_inception.keras"
}

# Diccionario de descripciones de modelos
model_details = {
    "VGG": {
        "description": "El modelo VGG es conocido por su arquitectura profunda y su uso en tareas de clasificación de imágenes.",
        "image": "vgg16_image.png",
        "detailed_description": "El modelo VGG (Visual Geometry Group) utiliza una arquitectura de red neuronal convolucional con múltiples capas profundas, ideal para aplicaciones donde la precisión es fundamental. Fue introducido en 2014 y es ampliamente utilizado en tareas de visión por computadora."
    },
    "ResNet50": {
        "description": "ResNet50 es una red residual con 50 capas, ideal para evitar el problema del desvanecimiento del gradiente.",
        "image": "resnet_image.png",
        "detailed_description": "La principal característica de estos es que solucionan el problema del desvanecimiento del gradiente cuando als redes se hacen realmente profundas. Es decir, que cuando la red aprende y actualiza los pesos con backpropagation, una vez el gradiente llega a las primeras capas ya es tan pequeño que los pesos no se actualizan. Por eso el modelo deja de aprender. ResNet en pocas palabras genera un puente 'skip conections' para el gradiente que le permite llegar a las caoas mas internas.\
Intetamos llevarnos la imagen original para que esta no se olvide en las capas más profundas. Le añadimos unos ciertos residuos que la red considerar importante (diferencias entre la img pixelada y con buena resolución). De esta forma intentamos modelar esta diferencia.",
        "arquitectura": "arquitectura_resnet.png"
    },
    "Inception": {
        "description": "Inception utiliza bloques convolucionales modulares para lograr una gran precisión con menos parámetros.",
        "image": "inception_image.png",
        "detailed_description": "La arquitectura Inception optimiza el rendimiento sin incrementar excesivamente la profundidad de las redes, lo que podría causar sobreajuste y alta carga computacional. Para lograrlo, combina convoluciones de diferentes tamaños (1x1, 3x3, 5x5) y max-pooling en una misma capa, capturando información local y global de la imagen simultáneamente.\
Inicialmente, este diseño aumentaba significativamente los parámetros del modelo. Para mitigar esto, se introdujeron convoluciones 1x1 como preprocesamiento, reduciendo la dimensionalidad antes de aplicar filtros más grandes. Este enfoque reduce redundancias y mejora la eficiencia al disminuir los parámetros necesarios, manteniendo el ancho y alto de las matrices de características.",
        "arquitectura": "arquitectura_inception.png"
    }
}

# Función para cargar el modelo seleccionado
@st.cache_resource
def load_model(model_name):
    model_path = model_paths.get(model_name)
    if model_path:
        return tf.keras.models.load_model(model_path)
    else:
        st.error("Modelo no encontrado.")
        return None

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
@st.cache_data
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
    #st.sidebar.write(f"**Descripción del modelo seleccionado:**")
    #st.sidebar.write(model_details[option]["description"])

    # Mostrar imagen y descripción detallada
    st.write(f"### Descripción detallada del modelo {option}")
    st.write(model_details[option]["detailed_description"])
    st.image(model_details[option]["image"], caption=option, use_column_width=True)


    # Cargar el modelo correspondiente
    model = load_model(option)

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
            st.sidebar.write("Esta función estará disponible proximamente")

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
        st.write("### Calsificación de la imagen")
        # Mostrar el resultado
        st.write(f"Predicción: **{predicted_class}**")
        #st.write(f"Confianza: **{confidence * 100:.2f}%**")

        # Aplicar LIME
        st.write("Generando explicaciones con LIME, por favor espere...")
        image_for_lime = image_array  # Imagen procesada para LIME
        temp, mask = apply_lime(image_for_lime, model)

        # Visualizar la explicación
        c = st.container()
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
    st.write("Este conjunto de datos contiene 70k imagenes de caras reales y 70k imagenes de caras generadas artificialemnte mediante GAN. GAN es una red neuronal\
             generativa que permite la creación de datos sintéticos a partir de datos reales basandose en un sus dos principales componentes: Generador y Discriminador.\
             El hecho de que las clases del dataset esten balanceadas nos faciltará mucho el trabajo. **¿Un humano puede diferencias las siguientes imagenes?**"
             )
    st.image("dataset_img.png")
    
    st.write("## Algunas aplicaciones en el mundo real:")
    st.write("""
- **Estos modelos podrían aplicarse tanto en las redes sociales como en las noticias** para evitar la publicación de deepfakes de terceros publicados sin consentimiento y evitar fraudes.
- Hoy en día, se está poniendo muy de moda generar imágenes y videos de personas públicas con las IA, lo cual plantea desafíos éticos y legales. Por ejemplo:
""")
    st.image("Elon_musk.png")

    st.write("## Pasos realizados:")
    st.write("### 1) Análisis EDA")
    st.write("Nuestra principal preocupación es si los datos realmente estan balanceados...")
    st.image("eda.png")
    st.write("")
    st.write("""
             ### 2) Preparación del dataset
            Las imagenes originales tenían un tamaño de **256x256** en RGB 0-255. Para facilitar el entrenamiento de los modelos hemos reducido el tamaño de las imagenes\
             a 128x128 y normalizado los 3 canales para que esten entre **0-1**
            """)
    st.image("pixeles.png")
    st.write("""
            Anañizando la gráfica podemos concluir que **hay una distribución suave y uniforme en los pixeles** intermedios que facilitaran el aprendizaje\ 
            de nuestros modelos. Sin embargo también hay 2 outliers que serian los pixeles normalizados con un valor de 0 y 1... los cuales en el caso de un\
            mal renimiento de los modelos deberemos preporocesar.
            """)
    st.write("")
 
    st.write("### 3) Entreanmiento de modelos")
    st.write("""Una vez hecho el analisis EDA y el preporocesameinto de las imagenes pasaremos al entrnamiento de los modelos de clasificación.\
             Usaremos el trasnfer learning para importar estos modelos ya entrenados con las imagenes del dataset de ImageNet y congelaremos sus pesos. \
             Unicamente cambiaremos la parte del calsificador anñadiendo capas densas y cambaindo la función de activación final para nuestra clasificación.
            """)
    st.write("- ResNet50")
    st.write("- Inception v3")
    








