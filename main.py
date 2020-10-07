# Descargar base de datos de imágenes para entrenamiento
from navegador import Navegador
from uninet import Uninet
servicio = Uninet()
web = Navegador(servicio.datos()['lenguaje'])
firefox = web.abrir_navegador()
imagen = servicio.descargar_captcha(firefox)
while True:
    captcha = servicio.guardar_imagen(imagen)['imagen_descargada']
    servicio.llenar_formulario(firefox, captcha)
    continuar = input('Para continuar presione ENTER...')
    if continuar != '':
        web.cerrar_navegador()
        break
    else:
        imagen = servicio.descargar_captcha(firefox, True)
firefox.cerrar_navegador()


# Generar modelo de aprendizaje
from machine_learning import Aprendizaje
aprendizaje = Aprendizaje(servicio.datos()['servicio'])
aprendizaje.entrenar()



# Validar modelo generado
imagen = servicio.imagen_validacion()
print(servicio.leer_captcha(imagen, True))