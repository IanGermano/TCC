"""Carregar o drive"""

from google.colab import drive
drive.mount('/content/drive')

"""Imports"""

import numpy as np
import tensorflow as tf
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

"""Carregando o caminho para as imagens de validação e de teste"""

path_teste = '/content/drive/MyDrive/TCC/COVID-19-Dataset/teste-X-ray'

"""Carregando o dataset das imagens de teste"""

ds_teste = image_dataset_from_directory(
    path_teste,
    labels="inferred",
    label_mode="categorical",
    class_names=['class_Non-COVID', 'class_COVID'],
    color_mode="rgb",
    batch_size = 64,
    image_size = (224, 224),
    shuffle = True,
    crop_to_aspect_ratio=True,
)

"""Carregando o modelo treinado"""

model_path = '/content/drive/MyDrive/models/covid19_model'
model = keras.models.load_model(model_path)

"""Obtendo matriz de confusão das imagens do teste"""

lab = []
pred = []
for images, labels in ds_teste:
  lab.append(tf.argmax(labels, axis=1))
  pred.append(tf.argmax(model.predict(images), axis=1))
true_categories = tf.concat(lab, axis=0)
predicted_categories = tf.concat(pred, axis=0)

"""Montando a matriz de confusão"""

lbl = ["COVID", "NON-COVID"]
cm = confusion_matrix(true_categories, predicted_categories)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lbl)
disp.plot(cmap=plt.cm.Blues)
plt.savefig('/content/drive/MyDrive/mc.png')
plt.show()
