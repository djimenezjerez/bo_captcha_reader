import os,sys,inspect
directorio_actual = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
directorio_padre = os.path.dirname(directorio_actual)
sys.path.insert(0, directorio_padre)

from machine_learning import Aprendizaje
import cv2
import psutil
import random
import numpy as np
import time
from PIL import Image, ImageDraw
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from pathlib import Path

class Gestora:
    def __init__(self):
        self.datos_web = {
            'nombre': 'Bono Universal',
            'entidad': 'Gestora Pública de la Seguridad Social de Largo Plazo',
            'servicio': 'gestora',
            'lenguaje': 'en-US, en',
            'url': 'https://bonouniversal.gestora.bo',
            'xpath': '//*[@id="ui-fieldset-0-content"]/div/div/div[1]/div/div[1]'
        }
        if __name__ != '__main__':
            self.aprendizaje = Aprendizaje(self.datos_web['servicio'])
        else:
            self.aprendizaje = Aprendizaje()
        self.ocr = {
            'x0_corte': 0,
            'alto_letra': 4,
            'cantidad_letras': 5,
            'ancho_letra': 53,
            'alto_imagen': 40
        }

    def datos(self):
        return self.datos_web

    def descargar_captcha(self, navegador, refresh=False):
        if not refresh:
            navegador.get(self.datos_web['url'])
        else:
            navegador.find_element_by_xpath('//*[normalize-space(text())="Cambiar Imagen"]').click()
            time.sleep(1)
        WebDriverWait(navegador, 30).until(EC.presence_of_element_located((By.XPATH, self.datos_web['xpath'])))
        imagen_texto = navegador.find_element_by_xpath(self.datos_web['xpath']).text
        ancho, alto = (self.ocr['ancho_letra']*self.ocr['cantidad_letras'], self.ocr['alto_letra'])
        imagen = Image.new("RGB", (ancho, self.ocr['alto_imagen']), (255, 255, 255))
        y_letra = alto
        for linea in imagen_texto.split('\n'):
            ImageDraw.Draw(imagen).text((0, y_letra), linea, fill=(0, 0, 0))
            y_letra += alto
        return imagen

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
        nombre_archivo = nombre_archivo.upper()
        imagen.save('.'.join(['/'.join([directorio, nombre_archivo]), 'png']))
        self.extraer_letras_captcha(imagen, nombre_archivo)
        return {
            'imagenes_entrenamiento': len(list(Path(self.aprendizaje.directorios()['entrenamiento']).glob('*'))),
            'imagenes_validacion': len(list(Path(self.aprendizaje.directorios()['validacion']).glob('*'))),
            'imagen_descargada': nombre_archivo
        }

    def leer_captcha(self, imagen, abrir_imagen=False):
        imagen_bn = self.__recortar_imagen(imagen)
        predicciones = []
        if abrir_imagen:
            imagen_lectura = cv2.cvtColor(np.array(imagen), cv2.COLOR_RGB2BGR)
        for i in range(self.ocr['cantidad_letras']):
            x = self.ocr['ancho_letra'] * i
            letra_imagen = self.__recortar_letra(imagen_bn, x)
            letra = self.aprendizaje.leer_letra(letra_imagen)
            predicciones.append(letra)
            if abrir_imagen:
                cv2.putText(imagen_lectura, letra, (x+30, 15), cv2.FONT_HERSHEY_DUPLEX, 0.6, (34, 171, 71), 2)
        if abrir_imagen:
            imagen_lectura = Image.fromarray(imagen_lectura)
            imagen_lectura.show()
        return ''.join(predicciones)

    def extraer_letras_captcha(self, imagen, texto_correcto):
        if len(texto_correcto) != self.ocr['cantidad_letras']:
            return False
        imagen_bn = self.__recortar_imagen(imagen)
        for i in range(len(texto_correcto)):
            letra_texto = texto_correcto[i]
            x = self.ocr['ancho_letra'] * i
            letra_imagen = self.__recortar_letra(imagen_bn, x)
            directorio_letra = '/'.join([self.aprendizaje.directorios()['aprendizaje'], letra_texto])
            Path(directorio_letra).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(directorio_letra+'/{}.png'.format(str(len(list(Path(directorio_letra).glob('*')))+1).zfill(6)), letra_imagen)

    def extraer_letras(self):
        self.aprendizaje.eliminar_directorio(Path(self.aprendizaje.directorios()['aprendizaje']))
        archivo_cantidad = len(list(Path(self.aprendizaje.directorios()['entrenamiento']).glob('*')))
        for (i, archivo_captcha) in enumerate(Path(self.aprendizaje.directorios()['entrenamiento']).iterdir()):
            print('[INFO] Procesando imagen {}/{}'.format(i+1, archivo_cantidad))
            captcha_correcto = archivo_captcha.stem
            imagen = cv2.imread(str(archivo_captcha))
            self.extraer_letras_captcha(imagen, captcha_correcto)
        print('[INFO] Proceso finalizado')

    def imagen_validacion(self, archivo=None):
        if archivo:
            return Image.open(self.aprendizaje.directorios()['entrenamiento']+'/'+archivo+'.png', 'r')
        else:
            imagenes_validacion = list(Path(self.aprendizaje.directorios()['entrenamiento']).glob('*'))
            return Image.open(open(str(random.choice(imagenes_validacion)), 'rb'))

    def llenar_formulario(self, navegador, texto):
        captcha_xpath = '//*[@name="captcha"]'
        WebDriverWait(navegador, 30).until(EC.presence_of_element_located((By.XPATH, captcha_xpath)))
        captcha_form = navegador.find_element_by_xpath(captcha_xpath).clear()
        captcha_form = navegador.find_element_by_xpath(captcha_xpath).send_keys(texto)

    def __recortar_imagen(self, imagen):
        imagen = cv2.cvtColor(np.array(imagen), cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    def __recortar_letra(self, imagen, x):
        return self.__rellenar_bordes(imagen[0:self.ocr['alto_imagen'], x:x+self.ocr['ancho_letra']])

    def __rellenar_bordes(self, imagen):
        return cv2.copyMakeBorder(imagen, 10, 10, 4, 3, cv2.BORDER_CONSTANT, value=[255,255,255])