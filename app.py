# 📦 Imports
import os
import pathlib
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import model_from_json
from PIL import Image, UnidentifiedImageError
from google.colab import files
import csv

# 📥 Descargar y preparar datos
url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_path = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', origin=url)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(os.path.dirname(zip_path))

base_dir = os.path.join(pathlib.Path(zip_path).parent, 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# 📊 Preparar generadores de imágenes
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(100, 100), batch_size=10, class_mode='binary'
)
validation_generator = val_datagen.flow_from_directory(
    validation_dir, target_size=(100, 100), batch_size=10, class_mode='binary'
)

# 🧠 Definir y entrenar el modelo
model = models.Sequential([
    layers.Conv2D(8, (3,3), activation='relu', input_shape=(100, 100, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(16, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, epochs=5, validation_data=validation_generator)

# 💾 Guardar el modelo
model_json = model.to_json()
with open("model_gats_gossos.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model_gats_gossos.weights.h5")
files.download("model_gats_gossos.json")
files.download("model_gats_gossos.weights.h5")

# 📈 Mostrar gráfica de precisión
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Entrenamiento')
plt.plot(epochs, val_acc, 'b', label='Validación')
plt.title('Precisión')
plt.legend()
plt.show()

# 📂 Crear historial si no existe
historial_file = 'prediccions_historial.csv'
if not os.path.exists(historial_file):
    with open(historial_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['imatge', 'prediccio', 'confiança', 'correcte', 'resultat'])

# 🔄 Cargar modelo guardado
with open("model_gats_gossos.json", "r") as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_gats_gossos.weights.h5")
loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 📸 Bucle de predicción manual
continuar = True
while continuar:
    uploaded = files.upload()
    uploaded_filename = list(uploaded.keys())[0]
    try:
        image = Image.open(uploaded_filename).convert("RGB").resize((100, 100))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = loaded_model.predict(img_array)
        prob = float(prediction[0])

        if prob > 0.5:
            prediccio_text = 'gos'
            conf = prob
        else:
            prediccio_text = 'gat'
            conf = 1 - prob

        print(f"✅ Predicción: {prediccio_text.upper()} ({conf*100:.2f}% confianza)")

        # 📊 Comparar con historial
        similars = []
        with open(historial_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['prediccio'] == prediccio_text:
                    similars.append(float(row['confiança']))
        if similars:
            avg_conf = np.mean(similars)
            print(f"📈 Confianza media anterior para '{prediccio_text}': {avg_conf*100:.2f}%")
            if conf >= avg_conf:
                print("🔍 Esta predicción es tan buena o mejor que la media.")
            else:
                print("⚠️ Esta predicción tiene menos confianza que la media de casos similares.")
        else:
            print("ℹ️ No hay datos similares suficientes para comparar.")

        # 📸 Visualizar imagen
        plt.figure(figsize=(6,6))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"{prediccio_text.upper()} ({conf*100:.2f}%)", fontsize=14)
        plt.show()

        # 📢 Confirmación usuario
        resposta = input("¿Es correcta? (s/n): ").strip().lower()
        if resposta == 'n':
            correcte = input("¿Cuál es la respuesta correcta? (gat/gos): ").strip().lower()
            if correcte in ['gat', 'gos']:
                os.makedirs(f"correccions/{correcte}", exist_ok=True)
                nueva_ruta = f"correccions/{correcte}/{uploaded_filename}"
                os.rename(uploaded_filename, nueva_ruta)
                print(f"✅ Imagen guardada en {nueva_ruta}.")
                resultat = 'error'
            else:
                print("❌ Valor no válido.")
                resultat = 'desconegut'
                correcte = 'desconegut'
        else:
            print("🎉 Bien! La predicción ha sido correcta.")
            correcte = prediccio_text
            resultat = 'encert'

        # 📂 Guardar en historial
        with open(historial_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([uploaded_filename, prediccio_text, conf, correcte, resultat])

        # 📋 Opciones finales
        print("\n¿Qué quieres hacer ahora?")
        print("1️⃣ Agregar otra imagen")
        print("2️⃣ Finalizar")
        opcion = input("Elige una opción (1/2): ").strip()
        if opcion != '1':
            continuar = False
            print("👋 Fin del programa.")

    except UnidentifiedImageError:
        print("❌ Imagen no válida.")


