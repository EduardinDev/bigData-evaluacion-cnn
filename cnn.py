#Importar librerías

import glob
import random
import base64
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from PIL import Image
from io import BytesIO
from IPython.display import HTML


from sklearn import preprocessing
from sklearn.model_selection import train_test_split


import numpy as np
import os,cv2,random,time,shutil,csv
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from tqdm import tqdm
import json,os,cv2,keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import BatchNormalization,Dense,GlobalAveragePooling2D,Lambda,Dropout,InputLayer,Input
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras.preprocessing.image import load_img
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

#Configuracion de rutas de acceso a los datos
ruta = r'/content/sample_data/'
ruta_labels_csv = 'labels.csv'
ruta_sample_submission_csv = ruta + 'sample_submission.csv'
ruta_submission_csv = 'sample_submission'
train_ruta = ruta + 'train'
test_ruta = ruta + 'test'

number_of_epochs = 10

labels_df = pd.read_csv(ruta_labels_csv)
print(f'The shape of the labels: {labels_df.shape}')

#mirando las cabeceras de ka etiqueta
labels_df.head()

#Preprocesamiento de las imegenes
def preprocesar_imagen(image, image_size):
    """ Image Preprocessing """

    # Verificar si la imagen necesita ser redimensionada
    if image.shape[0] != image_size or image.shape[1] != image_size:
        # Rescalado de la Imagen
        image_rescale = cv2.resize(image, dsize=(image_size, image_size), interpolation=cv2.INTER_AREA)
    else:
        image_rescale = image
    
    return image_rescale

def comparar_imagenes(image):
    """ Compare original and prepared image """
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    fig.subplots_adjust(hspace = .1, wspace=.1)
    axs = axs.ravel()
    # Plot Original Image
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title('original shape: '+str(image.shape))
    # Image Preprocessing
    image_rescale = preprocesar_imagen(image, image_size)
    # Plot Prepared Image
    axs[1].imshow(image_rescale, cmap='gray')
    axs[1].set_title('rescaled shape: '+str(image_rescale.shape))
    for i in range(2):
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])
    plt.show()

##Establecemos un tamaño general para las imagenes
image_size = 128
row = 0
id_ = labels_df.loc[row, 'id']
breed = labels_df.loc[row, 'breed']
file = id_+'.jpg'
image = cv2.imread('train/0042188c895a2f14ef64a918ed9c7b64.jpg', cv2.IMREAD_GRAYSCALE)
print('Shape:', image.shape)
comparar_imagenes(image)



#Leer imagenes
def preparacion_datos(data, image_size):
    """ Read all images into a numpy array """
    
    X = np.empty((len(data), image_size, image_size), dtype=np.uint8)
    for row in data.index:
        id_ = data.loc[row, 'id']
        file = id_ + '.jpg'
        image = cv2.imread('train/0042188c895a2f14ef64a918ed9c7b64.jpg', cv2.IMREAD_GRAYSCALE)
        image_rescaled = preprocesar_imagen(image, image_size)
        X[row, :, :] = image_rescaled
    X = X.astype('float32')/255
    return X
datos = preparacion_datos(labels_df, image_size)



#Separar en datos de entrenamiento y prueba

X_train = preparacion_datos(labels_df, image_size)
y_train = labels_df['breed']
y_train = pd.get_dummies(y_train)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state=2021)
print('Shape train data:', X_train.shape)
print('Shape val data:', X_val.shape)



# Aplanar las imágenes
X_train_reshaped = X_train.reshape(X_train.shape[0], image_size * image_size)
X_val_reshaped = X_val.reshape(X_val.shape[0], image_size * image_size)

# Normalizar los datos
min_max_scaler = MinMaxScaler()

# Ajustar el escalador a los datos de entrenamiento
min_max_scaler.fit(X_train_reshaped)

# Transformar los datos de entrenamiento y validación
X_train_normalized = min_max_scaler.transform(X_train_reshaped)
X_val_normalized = min_max_scaler.transform(X_val_reshaped)

# Restaurar la forma original de los datos
X_train_normalized = X_train_normalized.reshape(X_train_normalized.shape[0], image_size, image_size, 1)
X_val_normalized = X_val_normalized.reshape(X_val_normalized.shape[0], image_size, image_size, 1)

# Construir el modelo
CNN_model = Sequential()
CNN_model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(image_size, image_size, 1)))
CNN_model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'))
CNN_model.add(MaxPool2D(pool_size=(2, 2)))
CNN_model.add(Dropout(0.15))

CNN_model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
CNN_model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
CNN_model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
CNN_model.add(Dropout(0.15))

CNN_model.add(Flatten())
CNN_model.add(Dense(120, activation='softmax'))

# Entrenamiento del modelo
epochs = 10
batch_size = 128

CNN_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

CNN_history = CNN_model.fit(X_train_normalized, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(X_val_normalized, y_val))

CNN_model.save("modelo1.h5")