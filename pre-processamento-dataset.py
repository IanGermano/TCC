"""Carregar o drive"""

from google.colab import drive
drive.mount('/content/drive')

"""Imports"""
import numpy as np
import os

"""Caminhos para os datasets"""
path_covid_treino = '/content/drive/MyDrive/TCC/COVID-19-Dataset/X-ray/class_COVID'
path_non_covid_treino = '/content/drive/MyDrive/TCC/COVID-19-Dataset/X-ray/class_Non-COVID'
path_covid_teste = '/content/drive/MyDrive/TCC/COVID-19-Dataset/teste-X-ray/class_COVID'
path_non_covid_teste = '/content/drive/MyDrive/TCC/COVID-19-Dataset/teste-X-ray/class_Non-COVID'
count = 1

"""loop para renomear as imagens do treino COVID"""
for file in os.listdir(path_covid_treino):
	nn = "COVID_image_" + str(count) + ".jpg"
	os.rename(path+'/'+file, path+'/'+nn)
	count += 1
count = 1

"""loop para renomear as imagens do treino Non-COVID"""
for file in os.listdir(path_non_covid_treino):
	nn = "Non-COVID_image_" + str(count) + ".jpg"
	os.rename(path+'/'+file, path+'/'+nn)
	count += 1
count = 1

"""loop para renomear as imagens de teste COVID"""
for file in os.listdir(path_covid_teste):
	nn = "COVID_image_" + str(count) + ".png"
	os.rename(path+'/'+file, path+'/'+nn)
	count += 1
count = 1

"""loop para renomear as imagens de teste Non-COVID"""
for file in os.listdir(path_non_covid_teste):
	nn = "Non-COVID_image_" + str(count) + ".png"
	os.rename(path+'/'+file, path+'/'+nn)
	count += 1
