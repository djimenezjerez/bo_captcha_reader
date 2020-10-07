import os,sys,inspect
directorio_actual = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
directorio_padre = os.path.dirname(directorio_actual)
sys.path.insert(0, directorio_padre)

from machine_learning import Aprendizaje
import cv2
import base64
import psutil
import random
import numpy as np
import time
from PIL import Image
from io import BytesIO
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from pathlib import Path

class Mindef:
    def __init__(self):
        self.datos_web = {
            'nombre': 'Servicio Premilitar',
            'entidad': 'Ministerio de Defensa',
            'servicio': 'mindef',
            'lenguaje': 'en-US, en',
            'url': 'https://www.mindef.gob.bo/rupremil/',
            'xpath': '//*[@id="img"]'
        }
        if __name__ != '__main__':
            self.aprendizaje = Aprendizaje(self.datos_web['servicio'])
        else:
            self.aprendizaje = Aprendizaje()
        self.ocr = {
            'x0_corte': 10,
            'y0_corte': 10,
            'cantidad_letras': 6,
            'ancho_letra': 60,
            'alto_letra': 60
        }

    def datos(self):
        return self.datos_web

    def descargar_captcha(self, navegador, refresh=False):
        if not refresh:
            navegador.get(self.datos_web['url'])
        else:
            navegador.find_element_by_xpath('//*[@id="reload"]').click()
            time.sleep(1)
        WebDriverWait(navegador, 30).until(EC.presence_of_element_located((By.XPATH, self.datos_web['xpath'])))
        imagen = navegador.find_element_by_xpath(self.datos_web['xpath']).screenshot_as_png
        return Image.open(BytesIO(imagen))

    def guardar_imagen(self, imagen):
        imagenes_entrenamiento = len(list(Path(self.aprendizaje.directorios()['entrenamiento']).glob('*')))
        if self.aprendizaje.modelo_creado():
            prediccion = self.leer_captcha(imagen)
        while True:
            if self.aprendizaje.modelo_creado():
                print('Captcha: ' + prediccion)
                nombre_archivo = input('Si la lectura es incorrecta escriba el texto del captcha, de lo contrario presione ENTER: ')
            else:
                nombre_archivo = input('Escriba el texto del captcha: ')
            if nombre_archivo == '' and self.aprendizaje.modelo_creado():
                nombre_archivo = prediccion
                break
            elif len(nombre_archivo) != self.ocr['cantidad_letras']:
                print('El texto debe contener ' + str(self.ocr['cantidad_letras']) + ' caracteres vuelva a intentarlo...')
            else:
                break
        for proc in psutil.process_iter():
            if proc.name() == 'display':
                proc.kill()
        if imagenes_entrenamiento < 1000:
            directorio = self.aprendizaje.directorios()['entrenamiento']
        else:
            directorio = self.aprendizaje.directorios()['validacion']
        imagen.save('.'.join(['/'.join([directorio, nombre_archivo]), 'png']))
        self.extraer_letras_captcha(imagen, nombre_archivo)
        return {
            'imagenes_entrenamiento': len(list(Path(self.aprendizaje.directorios()['entrenamiento']).glob('*'))),
            'imagenes_validacion': len(list(Path(self.aprendizaje.directorios()['validacion']).glob('*'))),
            'imagen_descargada': nombre_archivo
        }

    def leer_captcha(self, imagen, abrir_imagen=False):
        imagen_bn = self.__recortar_imagen(imagen)
        contornos = self.__buscar_contornos(imagen_bn)
        predicciones = []
        if abrir_imagen:
            imagen_lectura = cv2.cvtColor(np.array(imagen), cv2.COLOR_RGB2BGR)
        for i in range(self.ocr['cantidad_letras']):
            letra_imagen = self.__recortar_letra(imagen_bn, contornos[i])
            letra = self.aprendizaje.leer_letra(letra_imagen)
            predicciones.append(letra)
            if abrir_imagen:
                cv2.putText(imagen_lectura, letra, ((self.ocr['x0_corte']*i)+5, int((self.ocr['y0_corte']))), cv2.FONT_HERSHEY_DUPLEX, 0.5, (34, 171, 71), 1)
        if abrir_imagen:
            imagen_lectura = Image.fromarray(imagen_lectura)
            imagen_lectura.show()
        return ''.join(predicciones)

    def extraer_letras_captcha(self, imagen, texto_correcto):
        if len(texto_correcto) != self.ocr['cantidad_letras']:
            return False
        imagen_bn = self.__recortar_imagen(imagen)
        contornos = self.__buscar_contornos(imagen_bn)
        for i in range(len(texto_correcto)):
            letra_texto = texto_correcto[i]
            letra_imagen = self.__recortar_letra(imagen_bn, contornos[i])
            directorio_letra = '/'.join([self.aprendizaje.directorios()['aprendizaje'], letra_texto])
            Path(directorio_letra).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(directorio_letra+'/{}.png'.format(str(len(list(Path(directorio_letra).glob('*')))+1).zfill(6)), letra_imagen)

    def extraer_letras(self):
        self.aprendizaje.eliminar_directorio(Path(self.aprendizaje.directorios()['aprendizaje']))
        archivo_cantidad = len(list(Path(self.aprendizaje.directorios()['entrenamiento']).glob('*')))
        for (i, archivo_captcha) in enumerate(Path(self.aprendizaje.directorios()['entrenamiento']).iterdir()):
            print('[INFO] Procesando imagen {}/{}'.format(i+1, archivo_cantidad))
            captcha_correcto = archivo_captcha.stem
            # imagen = cv2.imread(str(archivo_captcha))
            self.extraer_letras_captcha(Image.open(str(archivo_captcha), 'r'), captcha_correcto)
        print('[INFO] Proceso finalizado')

    def imagen_validacion(self, archivo=None):
        if archivo:
            return Image.open(self.aprendizaje.directorios()['entrenamiento']+'/'+archivo+'.png', 'r')
        else:
            imagenes_validacion = list(Path(self.aprendizaje.directorios()['entrenamiento']).glob('*'))
            return Image.open(open(str(random.choice(imagenes_validacion)), 'rb'))

    def llenar_formulario(self, navegador, texto):
        captcha_xpath = '//*[@id="txtcaptcha"]'
        WebDriverWait(navegador, 30).until(EC.presence_of_element_located((By.XPATH, captcha_xpath)))
        captcha_form = navegador.find_element_by_xpath(captcha_xpath).clear()
        captcha_form = navegador.find_element_by_xpath(captcha_xpath).send_keys(texto)

    def __recortar_imagen(self, imagen):
        R, G, B = imagen.convert('RGB').split()
        r = R.load()
        g = G.load()
        b = B.load()
        ancho, alto = imagen.size
        for i in range(ancho):
            for j in range(alto):
                if(r[i, j] != 175 or g[i, j] != 199 or b[i, j] != 200):
                    color = 255
                else:
                    color = 0
                r[i, j] = g[i, j] = b[i, j] = color
        imagen = Image.merge('RGB', (R, G, B))
        return self.__convertir_bn(imagen)

    def __convertir_bn(self, imagen):
        imagen = cv2.cvtColor(np.array(imagen), cv2.COLOR_BGR2GRAY)
        alto, ancho = imagen.shape
        factor = 4
        return cv2.resize(imagen, (int(alto*factor*20), int(ancho*factor)), interpolation=cv2.INTER_CUBIC)

    def __buscar_contornos(self, imagen):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 50))
        imagen = cv2.threshold(imagen, 127, 255, cv2.THRESH_BINARY_INV)[1]
        imagen = cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, kernel)
        contornos = cv2.findContours(imagen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        bordes = []
        for contorno in contornos:
            bordes.append(cv2.boundingRect(contorno))
        conteo = np.asarray(bordes)
        lista = np.lexsort((conteo[:,0], conteo[:,1]))
        return sorted(conteo[lista], key=lambda x: (x[0], x[1]))

    def __recortar_letra(self, imagen, contorno):
        factor = 4
        (x, y, ancho, alto) = contorno
        return self.__rellenar_bordes(cv2.resize(imagen[y:y+alto, x:x+ancho], (0,0), fx=0.20, fy=0.20))

    def __rellenar_bordes(self, imagen):
        alto, ancho = imagen.shape
        x_izquierda = int((self.ocr['ancho_letra'] - ancho) / 2)
        y_superior = int((self.ocr['alto_letra'] - alto) / 2)
        return cv2.copyMakeBorder(imagen, y_superior, self.ocr['alto_letra']-alto-y_superior, x_izquierda, self.ocr['ancho_letra']-ancho-x_izquierda, cv2.BORDER_CONSTANT, value=[255,255,255])