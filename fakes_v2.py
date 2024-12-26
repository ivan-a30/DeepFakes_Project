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
    "VGG": "https://drive.google.com/uc?id=1F2PWzz968SgDgF_iXKhpcIWBUme1IkNv",  # Reemplaza con el enlace de Google Drive
    "ResNet50": "https://drive.google.com/uc?id=1F2PWzz968SgDgF_iXKhpcIWBUme1IkNv",  # Reemplaza con el enlace de Google Drive
    "Inception": "https://drive.google.com/uc?id=1iHJK3UA1gHDfMXB_EvhxDwOjbluUsMFk"
}

# Diccionario de descripciones de modelos
model_details = {
    "VGG": {
        "description": "El modelo VGG es conocido por su arquitectura profunda y su uso en tareas de clasificación de imágenes.",
        "image": "vgg16_image.png",
        "detailed_description": "El modelo VGG (Visual Geometry Group) utiliza una arquitectura de red neuronal convolucional con múltiples capas profundas, ideal para aplicaciones donde la precisión es fundamental. Fue introducido en 2014 y es ampliamente utilizado en tareas de visión por computadora.",
        "accuracy": "https://drive.google.com/uc?id=1HiLVunasCR23YGyPVlgdDnMDWk3-akoN",
        "Matriz": "matriz_resnet.png",
        "metricas": "metricas_resnet.png",
        "csvs": "layers_resnet_info.csv",
        "params": "params_resnet_info.csv"
    },
    "ResNet50": {
        "description": "ResNet50 es una red residual con 50 capas, ideal para evitar el problema del desvanecimiento del gradiente.",
        "image": "resnet_ss.png",
        "detailed_description": "La principal característica de estos es que solucionan el problema del desvanecimiento del gradiente cuando als redes se hacen realmente profundas. Es decir, que cuando la red aprende y actualiza los pesos con backpropagation, una vez el gradiente llega a las primeras capas ya es tan pequeño que los pesos no se actualizan. Por eso el modelo deja de aprender. ResNet en pocas palabras genera un puente 'skip conections' para el gradiente que le permite llegar a las caoas mas internas.\nIntetamos llevarnos la imagen original para que esta no se olvide en las capas más profundas. Le añadimos unos ciertos residuos que la red considerar importante (diferencias entre la img pixelada y con buena resolución). De esta forma intentamos modelar esta diferencia.",
        "arquitectura": "arquitectura_resnet.png",
        "accuracy": "https://drive.google.com/uc?id=1HiLVunasCR23YGyPVlgdDnMDWk3-akoN",
        "Matriz": "matriz_resnet.png",
        "metricas": "metricas_resnet.png",
        "csvs": "layers_resnet_info.csv",
        "params": "params_resnet_info.csv"
    },
    "Inception": {
        "description": "Inception utiliza bloques convolucionales modulares para lograr una gran precisión con menos parámetros.",
        "image": "inception_image.png",
        "detailed_description": "La arquitectura Inception optimiza el rendimiento sin incrementar excesivamente la profundidad de las redes, lo que podría causar sobreajuste y alta carga computacional. Para lograrlo, combina convoluciones de diferentes tamaños (1x1, 3x3, 5x5) y max-pooling en una misma capa, capturando información local y global de la imagen simultáneamente.\nInicialmente, este diseño aumentaba significativamente los parámetros del modelo. Para mitigar esto, se introdujeron convoluciones 1x1 como preprocesamiento, reduciendo la dimensionalidad antes de aplicar filtros más grandes. Este enfoque reduce redundancias y mejora la eficiencia al disminuir los parámetros necesarios, manteniendo el ancho y alto de las matrices de características.",
        "arquitectura": "arquitectura_inception.png",
        "accuracy": "https://drive.google.com/uc?id=1HiLVunasCR23YGyPVlgdDnMDWk3-akoN",
        "Matriz": "matriz_resnet.png",
        "metricas": "metricas_resnet.png",
        "csvs": "layers_resnet_info.csv",
        "params": "params_resnet_info.csv"
    }
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

    #st.write(f"Cargando el modelo {model_name}...")
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
        num_features=5
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
    #st.title("Clasificación de Imágenes: Real vs Deepfake")
    
    # Seleccionar modelo
    option = st.sidebar.selectbox(
        "Selecciona el modelo a utilizar:",
        ("VGG", "ResNet50", "Inception")
    )

    st.title(f"Clasificación con el modelo {option}")
    # Cargar el modelo correspondiente desde Google Drive
    status_placeholder = st.empty()
    status_placeholder.error(f"Cargando el modelo **{option}**, por favor espere...")
    model = load_model_from_drive(option)
    status_placeholder.success(f"El modelo **{option}** se ha cargado correctamente. Ahora puede cargar una imagen para clasificar.")

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
            st.sidebar.image(image, caption="Imagen cargada", use_container_width=True)

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
        class_names = ["Deepfake", "Real"]
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)
        st.write("### Clasificación de la imagen")
        # Mostrar el resultado
        #st.write(f"Predicción: **{predicted_class}**")

        # Mostrar un mensaje emergente con la clasificación
        if predicted_class == "Deepfake":
            st.error(f"⚠️ ¡Cuidado! La imagen es **Deepfake**. Confianza: {confidence:.2%}")
        else:
            st.success(f"✅ No te preocupes, la imagen es verdadera. Confianza: {confidence:.2%}")

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
        # Mostrar descripción del modelo seleccionado
    st.write(f"### Descripción detallada del modelo {option2}")
    st.write(model_details[option2]["detailed_description"])
    st.image(model_details[option2]["image"], caption=option2, use_container_width=True, width=100)

    col1, col2 = st.columns(2)
    df = pd.read_csv(model_details[option2]["csvs"])
    df2 = pd.read_csv(model_details[option2]["params"])
    with col1:
        st.dataframe(df)
    with col2:
        st.dataframe(df2)
    st.write(f"### Métricas y resultados")
    st.image(model_details[option2]["metricas"], caption="Matriz de confusion", use_container_width=True)
    st.image(model_details[option2]["Matriz"], caption="Matriz de confusion", use_container_width=True)
   # st.image(model_details[option2]["accuracy"], caption="Gráfica de Accuracy para Inception", use_container_width=True)
   # st.image("https://drive.google.com/uc?id=1HiLVunasCR23YGyPVlgdDnMDWk3-akoN", caption="Gráfica de Accuracy para Inception", use_container_width=True)


elif selected2 == "Explicación":
    st.title("DeepFake Face Detection Project")
    st.write("## Dataset:")
    st.write("Este conjunto de datos contiene imágenes de caras reales e imágenes de caras generadas artificialmente mediante GAN. GAN es una red neuronal\
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
    st.write("### 1. Análisis EDA")
    st.write("Nuestra principal preocupación es si los datos realmente están balanceados...")
    st.image("eda.png")
    st.write("Podemos comprobar que efectivamente las clases estan balanceadas tanto en el conjunto train como en el validación.")
    st.write("")
    st.write("""
    ### 2. Preparación del dataset
    Las imágenes originales tenían un tamaño de **256x256** píxeles en formato RGB (valores entre 0 y 255). 
    Para optimizar el entrenamiento de los modelos, hemos reducido el tamaño de las imágenes a **128x128** y 
    normalizado los valores de los 3 canales RGB para que estén en el rango **0-1**.
    """)
    st.image("pixeles.png")
    st.write("""
    Al analizar la gráfica, observamos que **existe una distribución uniforme y suave en los valores de los píxeles intermedios**, 
    lo cual facilita el aprendizaje de nuestros modelos. No obstante, también se identifican **2 valores atípicos (outliers)** 
    correspondientes a los píxeles normalizados con valores de 0 y 1. En caso de observar un bajo rendimiento en los modelos, 
    sería necesario ajustar el preprocesamiento para abordar este problema.
    """)

    st.write("")
 
    st.write("### 3. Entrenamiento de modelos")
    st.write("""
            Una vez hecho el análisis EDA y el preprocesamiento de las imágenes pasaremos al entrenamiento de los modelos de clasificación.\
             Usaremos el transfer learning para importar estos modelos ya entrenados con las imágenes del dataset de ImageNet y congelaremos sus pesos. \
             Únicamente cambiaremos la parte del clasificador añadiendo capas densas y cambiando la función de activación final para nuestra clasificación.
             Entrenaremos 10 épocas para cada modelo sobre el conjunto de train e iremos ajustando los parametros con el conjunto de validación.
            """)
    st.write("### 4. Predicciones sobre el conjunto test")
    st.write("""
    En esta sección evaluaremos el rendimiento del modelo entrenado realizando predicciones sobre el conjunto de prueba. 
    Analizaremos los resultados obtenidos para medir la precisión, sensibilidad, especificidad y otras métricas relevantes. 
    Además, presentaremos ejemplos visuales de las predicciones realizadas para observar el desempeño del modelo frente a 
    casos correctos y errores de clasificación.
    """)

