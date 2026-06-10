import math
from ucimlrepo import fetch_ucirepo
import random
import csv
from collections import Counter # es una clase que nos permite contar instancias

# Nodo del arbol ---------
# guarda si es hoja
# umbral de corte (porque trabajamos con atributos continuos)
# hijos
  
class Nodo:
    def __init__(self, esHoja, etiqueta, umbral):
        self.esHoja = esHoja
        self.etiqueta = etiqueta
        self.umbral = umbral
        self.hijos = []
    
    def predecir(self, instancia):
        if self.esHoja:
            return self.etiqueta
        else:
            valor = instancia[self.etiqueta]
            if valor <= self.umbral:
                return self.hijos[0].predecir(instancia)
            else:
                return self.hijos[1].predecir(instancia)

# CLASE PRINCIPAL C4_5 -------------------------

class C45:
    """Construye un arbol de decision con el algoritmo C4.5"""
    def __init__(self):
        self.data = []          # lista de instancias (cada una es un dict)
        self.atributos = []    # lista de nombres de atributos
        self.clases = []       # lista de clases posibles (ej: [1, 2, 3])
        self.arbol = None        # raíz del árbol (Node)

# Cargado de datos - dataset Wine id 109

    def fetch_data(self):
      print("Descargando dataset Wine desde UCI...")
      try:
        wine = fetch_ucirepo(id=109) # descarga dataset
        X = wine.data.features # obtiene atributos
        y = wine.data.targets # obtiene la clase
        datos = X.copy() # copia de atributos
        datos["class"] = y["class"] # agrega columna de clase
        self.data = datos.to_dict(orient="records") # convierte a lista de diccionarios
        self.atributos = [col for col in datos.columns if col != "class"] # guardo nombres de atributos
      except Exception:
        # En caso de excepcion: leer desde CSV local
        print("  (sin conexión, usando wine_dataset.csv local)")
        self.data = []
        with open("wine_dataset.csv", newline="") as f:
            reader = csv.DictReader(f) # lee CSV linea por linea
            for row in reader:
                inst = {k: float(v) for k, v in row.items()} # convierte valores a float
                inst["class"] = int(inst["class"]) # la clase se vuelve un int
                self.data.append(inst) # agrega instancia a lista de datos
            self.atributos = [k for k in self.data[0].keys() if k != "class"] # obtiene atributos
 
        self.clases = sorted(set(row["class"] for row in self.data)) # obtiene las clases distintas
 
        print(f"  Instancias  : {len(self.data)}")
        print(f"  Atributos   : {len(self.atributos)}")
        print(f"  Clases      : {self.clases}")

# Division del dataset 
# separa 80% entrenamiento y 20% test
# mantiene la proporcion de cada clase en ambos 

    def dividirDataset(self, test_ratio=0.2, semilla=42):
        random.seed(semilla) # semilla 42 para que siempre salga la misma division
 
        # Agrupar instancias por clase
        por_clase = {}
        for inst in self.data:
            c = inst["class"]
            por_clase.setdefault(c, []).append(inst) # crea lista por clase 
 
        entrenamiento, test = [], []
        for clase, instancias in sorted(por_clase.items()): # recorre cada clase
            random.shuffle(instancias) # mezcla instancias
            n_test = max(1, round(len(instancias) * test_ratio)) # cuantas a test
            # el max(1,..) nos asegura que al menos haya una de una clase
            test.extend(instancias[:n_test]) # toma primeras para el test
            entrenamiento.extend(instancias[n_test:]) # el resto a entrenamiento
 
        # mezclamos para que no queden agrupadas por clase
        random.shuffle(entrenamiento) # con la misma semilla mezclan igual
        random.shuffle(test)
 
        print(f"\n  Entrenamiento: {len(entrenamiento)} instancias")
        print(f"  Test         : {len(test)} instancias")
        dist_entr = Counter(d["class"] for d in entrenamiento) # instancias de cada clase
        dist_test  = Counter(d["class"] for d in test)
        print(f"  Dist. entrenamiento  : {dict(sorted(dist_entr.items()))}")
        print(f"  Dist. test   : {dict(sorted(dist_test.items()))}")
 
        return entrenamiento, test
 
 
    def guardarCSV(self, datos, path):
        """Guarda una lista de dicts en un archivo CSV."""
        if not datos: # si la lista esta vacia
            return 
        with open(path, "w", newline="") as f: # abrimos en modo escritura
            writer = csv.DictWriter(f, fieldnames=datos[0].keys()) # crea csv escritor
            writer.writeheader() # escribe los atributos
            writer.writerows(datos) # escribe las filas
        print(f"  --> Guardado: {path}")

# Construccion del arbol
# recursivo desde la raiz

    def generadorArbol(self,entrData):
      """Construye el arbol usando el set de entrenamiento"""
      self.arbol = self.genArbolRec(entrData,self.atributos[:])

# Funcion recursiva principal

# caso base - se hace hoja
# 1. no hay datos = hoja FAIL
# 2. todos los datos son de la mism clase = hoja con esa clase 
# 3. no quedan atributos = hoja con la clase mas comun 

# caso recursivo - 
# elegir el mejor atributo
# dividir datos
# llamar recursivamente en cada subconjunto

    def genArbolRec(self, dataset, atr):
 
        # CASO BASE 1: no hay datos
        if len(dataset) == 0:
            return Nodo(True, "Fail", None)
 
        # CASO BASE 2: todos son de la misma clase
        mismaClase = self.todosMismaClase(dataset)
        if mismaClase is not False:
            return Nodo(True, mismaClase, None) # no se sigue dividiendo
 
        # CASO BASE 3: no quedan atributos para dividir
        if len(atr) == 0:
            freqClass = self.getClaseFrecuente(dataset) # uso clase mayoritaria
            return Nodo(True, freqClass, None) 
 
        # CASO RECURSIVO: elegir el mejor atributo y dividir
        mejorAtributo, mejorUmbral, subconjuntos = self.selectAtributo(dataset, atr)
 
        # En C4.5 con atributos continuos, el atributo puede
        # reutilizarse en distintas ramas. Por eso NO lo removemos.
        nodo = Nodo(False, mejorAtributo, mejorUmbral)
 
        # Crear un hijo por cada subconjunto (izquierdo y derecho)
        nodo.hijos = [
            self.genArbolRec(subset, atr)
            for subset in subconjuntos
        ]
        return nodo

# Metodos genArbolRec

    def todosMismaClase(self, data):
        """Verifica si todos los datos son de la misma clase.Retorna la clase si son iguales, False si no."""
        primera_clase = data[0]["class"]
        for row in data:
            if row["class"] != primera_clase:
                return False
        return primera_clase
 
    def getClaseFrecuente(self, dataset):
        """Retorna la clase más frecuente en dataset."""
        conteo = Counter(row["class"] for row in dataset)
        return conteo.most_common(1)[0][0]

# Seleccion del mejor atributo
# para cada uno, probamos todos los umbrales posibles - medios entre valores adyacentes ordenados
# se selecciona aquel atributo y umbral que maximizan el gain ratio

    def selectAtributo(self, curData, curAttributes):
        subconjuntos = []
        maxGain = -1 * float("inf")  # inicializar en el mínimo posible - cualquier gain ratio va a ser mejor
        best_attribute = None
        best_umbral = None
 
        for attribute in curAttributes:
            # Ordenar datos por el valor del atributo actual
            curData.sort(key=lambda x: x[attribute]) # ordenamos menor a mayor
 
            # Probar cada punto medio entre valores adyacentes
            for j in range(len(curData) - 1):
                val_j      = curData[j][attribute]
                val_j_next = curData[j + 1][attribute]
 
                # Solo crear umbral si los valores son distintos
                if val_j != val_j_next:
                    threshold = (val_j + val_j_next) / 2 # calcula un medio
 
                    # Dividir en menores/iguales y mayores al umbral
                    menor_igual    = [row for row in curData if row[attribute] <= threshold]
                    mayor = [row for row in curData if row[attribute] >  threshold]
 
                    # Calcular el Gain Ratio para esta división
                    e = self.gainRatio(curData, [menor_igual, mayor])

                    # Guardamos el mejor encontrado
                    if e >= maxGain:
                        maxGain        = e
                        subconjuntos       = [menor_igual, mayor]
                        best_attribute = attribute
                        best_umbral = threshold
 
        return (best_attribute,best_umbral,subconjuntos)
 
# Gain Ratio 

    def gainRatio(self, unionSet, subsets):
        S = len(unionSet)
 
        # Information Gain = entropía antes - entropía después
        entropiaAntes = self.entropia(unionSet)
        weights = [len(subset) / S for subset in subsets]
        entropiaDespues = sum(
            weights[i] * self.entropia(subsets[i])
            for i in range(len(subsets))
        )
        infoGain = entropiaAntes - entropiaDespues
 
        # Split Info: penalización por número y tamaño de ramas
        splitInfo = 0
        for subset in subsets:
            p = len(subset) / S
            splitInfo += p * self.log(1 / p) if p > 0 else 0
 
        # Evitar división por cero
        if splitInfo == 0:
            return 0
 
        return infoGain / splitInfo

# Entropia
# se usa la formula Shannon
#  H(p1,...,ps) = Σ pi * log(1/pi)

    def entropia(self, dataSet):
        S = len(dataSet)
        if S == 0:
            return 0
 
        # Contar instancias por clase
        conteo = Counter(row["class"] for row in dataSet)
 
        ent = 0
        for count in conteo.values():
            p = count / S            # proporción p_i
            ent += p * self.log(1/p) # p_i * log(1/p_i)
        return ent
 
    def log(self, x):
        """Logaritmo base 2. Retorna 0 si x es 0 (evita log(0))."""
        if x == 0:
            return 0
        return math.log(x, 2)

# PREDICCIÓN --------------------------------------
    def clasificar(self, test_data):
        """Evalúa el árbol en el set de test y retorna un reporte con precisión, matriz de confusión, etc."""
        correctas = 0
        total = len(test_data)
        resultados = []
        matriz_confusion = {} # inicializa matriz de confusion
        
        for inst in test_data:
            real = inst["class"]
            pred = self.arbol.predecir(inst)
            resultados.append((pred, real))
            
            if real not in matriz_confusion:
                matriz_confusion[real] = {}


            if pred not in matriz_confusion[real]:
                matriz_confusion[real][pred] = 0
            matriz_confusion[real][pred] += 1

            if pred == inst["class"]:
                correctas += 1

        precision = correctas / len(test_data) if test_data else 0

        return {"precision": precision, "correctas": correctas, "matriz_confusion": matriz_confusion, "resultados": resultados, "total": total}

# -----------------------------------------------------

# Informacion del arbol ===============================

    def printTree(self):
        print("\n  ÁRBOL (primeros 4 niveles):")
        self.printNode(self.arbol, indent="  ", nivel=0, max_nivel=4)
 
    def printNode(self, node, indent="", nivel=0, max_nivel=4):
        if nivel > max_nivel:
            return
        if node.esHoja:
            print(indent + f"[HOJA] --> Clase {int(node.etiqueta)}")
        else:
            print(indent + f"{node.etiqueta} <= {node.umbral:.4f}")
            self.printNode(node.hijos[0], indent + "|   ", nivel+1, max_nivel)
            self.printNode(node.hijos[1], indent + "    ", nivel+1, max_nivel)
 
    def contarNodos(self, node=None):
        """Retorna (total_nodos, hojas)."""
        if node is None:
            node = self.arbol
        if node.esHoja:
            return 1, 1
        t0, h0 = self.contarNodos(node.hijos[0])
        t1, h1 = self.contarNodos(node.hijos[1])
        return 1 + t0 + t1, h0 + h1
 
    def profundidad(self, node=None):
        """Calcula la profundidad máxima del árbol."""
        if node is None:
            node = self.arbol
        if node.esHoja:
            return 0
        return 1 + max(self.profundidad(node.hijos[0]), self.profundidad(node.hijos[1]))
    

# Guardar y mostrar resultados ===============================
    def guardarResultados(self, reporte, archivo):
        with open(archivo, "w", encoding="utf-8") as f:

            f.write("=== Resultados del Test ===\n")
            f.write(f"Instancias evaluadas: {reporte['total']}\n")
            f.write(f"Instancias correctamente clasificadas: {reporte['correctas']}\n")
            f.write(f"Precisión: {reporte['precision'] * 100:.2f}%\n\n")

            f.write("=== Predicciones ===\n")
            for i, (pred, real) in enumerate(reporte["resultados"], start=1):
                estado = "[OK]" if pred == real else "[ERROR]"
                f.write(
                    f"{i:3d}. Predicción: {pred:<15} "
                    f"Real: {real:<15} {estado}\n"
                )

            matriz = reporte["matriz_confusion"]
            clases = sorted(
                set(matriz.keys()) |
                {p for preds in matriz.values() for p in preds}
            )

            f.write("\n\n=== MATRIZ DE CONFUSIÓN ===\n")
            f.write(f"{'Real/Pred':10}")

            for clase in clases:
                f.write(f"{clase:7}")
            f.write("\n")
            for real in clases:
                f.write(f"{real:10}")
                for pred in clases:
                    count = matriz.get(real, {}).get(pred, 0)
                    f.write(f"{count:7}")
                f.write("\n")

        print(f"  --> Guardado: {archivo}")


    def mostrarResultados(self, reporte):
        print("\n  === Resultados del Test ===")
        print(f"  Instancias evaluadas: {reporte['total']}")
        print(f"  Instancias correctamente clasificadas: {reporte['correctas']}")
        print(f"  Precisión: {reporte['precision'] * 100:.2f}%")

        matriz = reporte["matriz_confusion"]
        clases = sorted( set(matriz.keys()) | {p for preds in matriz.values() for p in preds} )

        print("\n\n  === MATRIZ DE CONFUSIÓN ===")
        print(f"  {'Real/Pred':{12}}", end="")

        for clase in clases:
            print(f"{str(clase):{9}}", end="")
        for real in clases:
            print(f"\n  {str(real):<{12}}", end="")
            for pred in clases:
                count = matriz.get(real, {}).get(pred, 0)
                print(f"{count:<{9}}", end="")
             

# MAIN ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def main():
 
    # 1. Crear instancia y cargar datos
      print("\n[1] Cargando datos...")
      modelo = C45()
      modelo.fetch_data()
 
    # 2. Dividir dataset
      print("\n[2] Dividiendo dataset (80% train / 20% test)...")
      train, test = modelo.dividirDataset(test_ratio=0.2, semilla=42)
      print("\n  Guardando datasets divididos en CSV...")
      modelo.guardarCSV(train, "wine_train.csv")
      modelo.guardarCSV(test,  "wine_test.csv")
 
    # 3. Construir árbol
      print("\n[3] Construyendo árbol C4.5...")
      modelo.generadorArbol(train)
      total, hojas = modelo.contarNodos()
      prof = modelo.profundidad()
      print(f"  Nodos totales : {total}")
      print(f"  Hojas         : {hojas}")
      print(f"  Profundidad   : {prof}")
      modelo.printTree()

    # 4. Evaluar en test
      print("\n[4] Evaluando en set de test...")
      reporte = modelo.clasificar(test)
      print("\n  Guardando reporte de resultados...")
      modelo.guardarResultados(reporte, archivo="reporte_test.txt")
      modelo.mostrarResultados(reporte)
    

if __name__ == "__main__":
        main()