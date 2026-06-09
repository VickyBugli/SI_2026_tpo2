class NodoArbol:
    def __init__(self, min_muestras_para_dividir=2, profundidad_maxima=None):

        self.hijos = {}                     # nodos hijos
        self.clase = None          # clase si es hoja

        self.atributo_division = None       # atributo usado para dividir
        self.valor_umbral = None            # valor del corte

        self.min_muestras_para_dividir = min_muestras_para_dividir
        self.profundidad_maxima = profundidad_maxima

    
