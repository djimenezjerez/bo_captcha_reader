from selenium import webdriver
from selenium.webdriver.firefox.options import Options

class Navegador:
    def __init__(self, lenguaje):
        self.lenguaje = lenguaje
        self.navegador = self.__firefox()

    def __firefox(self):
        profile = webdriver.FirefoxProfile()
        profile.set_preference('intl.accept_languages', self.lenguaje)
        return webdriver.Firefox(firefox_profile=profile)

    def abrir_navegador(self):
        if not self.navegador:
            self.navegador = self.__firefox()
        return self.navegador

    def cerrar_navegador(self):
        self.navegador.close()
        self.navegador.quit()
        self.navegador = None