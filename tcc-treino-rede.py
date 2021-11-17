"""Carregar o drive"""

from google.colab import drive
drive.mount('/content/drive')

"""Imports"""

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from tensorflow.keras import layers

"""Carregando o caminho para as imagens"""

path = '/content/drive/MyDrive/TCC/COVID-19-Dataset/X-ray'

"""Carregando as imagens de treino para o dataset de treino"""

ds_training = image_dataset_from_directory(
    path,
    labels="inferred",
    label_mode="categorical",
    class_names=['class_Non-COVID', 'class_COVID'],
    color_mode="rgb",
    batch_size = 64,
    image_size = (224, 224),   
    shuffle = True,
    seed = 123,
    validation_split = 0.2,
    subset = "training",
    crop_to_aspect_ratio=True,
)

"""Carregando as imagens de validacao para o dataset de validacao"""

ds_validation = image_dataset_from_directory(
    path,
    labels="inferred",
    label_mode="categorical",
    class_names=['class_Non-COVID', 'class_COVID'],
    color_mode="rgb",
    batch_size = 64,
    image_size = (224, 224),
    shuffle = True,
    seed = 123,
    validation_split = 0.2,
    subset = "validation",
    crop_to_aspect_ratio=True,
)

"""Montando o modelo"""

base_model = ResNet50(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
base_model.trainable = False
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.Flatten()(x)
x = layers.Dense(4096, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(2048, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(1024, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(2, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.summary()

"""Compilando o modelo"""

model.compile(optimizer="adagrad", loss="categorical_crossentropy", metrics=['accuracy'])

"""Treinando o modelo"""

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)
history = model.fit(ds_training, validation_data=ds_validation, epochs=200, callbacks=[callback])
model.save("/content/drive/MyDrive/models/covid19_model")

"""Plot da acurácia do treino e da validação"""
figure(figsize=(9.6, 7.2), dpi=100) #tamanho da figura
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epo = range(1, len(train_acc)+1)
plt.plot(epo, train_acc, label='treino')
plt.plot(epo, val_acc, label='validação')
plt.title('Modelo Acurácia')
plt.xlabel('época')
plt.ylabel('acurácia')
plt.legend()
plt.savefig('/content/drive/MyDrive/accuracy.png')
plt.show()

"""Plot da loss treino e da validação"""
figure(figsize=(9.6, 7.2), dpi=100) #tamanho da figura
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epo = range(1, len(train_loss)+1)
plt.plot(epo, train_loss, label='treino')
plt.plot(epo, val_loss, label='validação')
plt.title('Modelo Perda')
plt.xlabel('época')
plt.ylabel('perda')
plt.legend()
plt.savefig('/content/drive/MyDrive/loss.png')
plt.show()
