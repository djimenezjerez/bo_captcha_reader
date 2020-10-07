import cv2
import pickle
import numpy as np
from imutils import paths
from pathlib import Path
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import load_model, Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from keras.layers import Dropout

class Aprendizaje:
    def __init__(self, servicio = None, epochs = 25):
        self.servicio = servicio
        self.archivo_modelo = 'modelo.hdf5'
        self.archivo_etiquetas = 'etiquetas.dat'
        self.modelo = None
        self.etiquetas = None
        self.archivo_modelo_creado = False
        if self.servicio:
            self.archivo_modelo = '/'.join([self.servicio, self.archivo_modelo])
            self.archivo_etiquetas = '/'.join([self.servicio, self.archivo_etiquetas])
        self.directorios_imagenes = {
            'entrenamiento': 'entrenamiento',
            'validacion': 'validacion',
            'aprendizaje': 'aprendizaje'
        }
        self.__crear_directorios()
        self.leer_modelo()
        self.epochs = epochs

    def directorios(self):
        self.__crear_directorios()
        return self.directorios_imagenes

    def leer_letra(self, imagen):
        if self.archivo_modelo_creado:
            # Añadir ejes para tratamiento de imagenes
            imagen = np.expand_dims(imagen, axis=2)
            imagen = np.expand_dims(imagen, axis=0)
            # Intentar lectura
            prediccion = self.modelo.predict(imagen)
            # Convertir la lectura a cadena de texto
            letra = self.etiquetas.inverse_transform(prediccion)[0]
            return letra
        else:
            self.entrenar()
            self.leer_letra(imagen)

    def entrenar(self):
        datos = []
        etiquetas = []
        cantidad_letras = len(list(Path(self.directorios_imagenes['aprendizaje']).glob('*')))
        for archivo in paths.list_images(self.directorios_imagenes['aprendizaje']):
            imagen = cv2.imread(archivo)
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            imagen = np.expand_dims(imagen, axis=2)
            etiqueta = archivo.split('/')[-2]
            datos.append(imagen)
            etiquetas.append(etiqueta)
        datos = np.array(datos, dtype="float") / 255.0
        etiquetas = np.array(etiquetas)
        (x_entrenamiento, x_validacion, y_entrenamiento, y_validacion) = train_test_split(datos, etiquetas, test_size=0.25, random_state=0)
        etiqueta = LabelBinarizer().fit(y_entrenamiento)
        y_entrenamiento = etiqueta.transform(y_entrenamiento)
        y_validacion = etiqueta.transform(y_validacion)
        with open(self.archivo_etiquetas, "wb") as f:
            pickle.dump(etiqueta, f)
        modelo = Sequential()
        # First convolutional layer with max pooling
        modelo.add(Conv2D(64, (3, 3), padding="same", input_shape=(60, 60, 1), activation="relu"))
        modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        modelo.add(Conv2D(64, (3, 3), padding="same", input_shape=(60, 60, 1), activation="relu"))
        modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # Second convolutional layer with max pooling
        modelo.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
        modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        modelo.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
        modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # Hidden layer with 500 nodes
        modelo.add(Flatten())
        modelo.add(Dropout(0.2, input_shape=(60,)))
        modelo.add(Dense(512, activation="relu"))
        # Output layer with 32 nodes (one for each possible letter/number we predict)
        modelo.add(Dense(cantidad_letras, activation="softmax"))
        # Ask Keras to build the TensorFlow model behind the scenes
        modelo.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        # Train the neural network
        modelo.fit(x_entrenamiento, y_entrenamiento, validation_data=(x_validacion, y_validacion), batch_size=cantidad_letras, epochs=self.epochs, verbose=1)
        # Save the trained model to disk
        modelo.save(self.archivo_modelo)
        self.leer_modelo()

    def modelo_creado(self):
        return self.archivo_modelo_creado

    def leer_modelo(self):
        self.archivo_modelo_creado = Path(self.archivo_modelo).is_file() and Path(self.archivo_etiquetas).is_file()
        if self.archivo_modelo_creado:
            self.modelo = load_model(self.archivo_modelo)
            with open(self.archivo_etiquetas, 'rb') as f:
                self.etiquetas = pickle.load(f)

    def eliminar_directorio(self, directorio):
        for subdirectorio in directorio.iterdir() :
            if subdirectorio.is_dir() :
                self.eliminar_directorio(subdirectorio)
            else :
                subdirectorio.unlink()
        directorio.rmdir()

    def __crear_directorios(self):
        for directorio in self.directorios_imagenes:
            if self.servicio:
                self.directorios_imagenes[directorio] = '/'.join([self.servicio, directorio])
            Path(self.directorios_imagenes[directorio]).mkdir(parents=True, exist_ok=True)