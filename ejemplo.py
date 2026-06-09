from ucimlrepo import fetch_ucirepo
import math
import random
from collections import Counter
from NodoArbol import NodoArbol
 
# ─────────────────────────────────────────────────────────────
#  CARGA DEL DATASET
# ─────────────────────────────────────────────────────────────
 
def cargar_wine():
    wine = fetch_ucirepo(id=109) # obtengo el dataset de Internet
    X = wine.data.features # x = atributos
    y = wine.data.targets # y = clase
    datos = X.copy() 
    # Agrega columna de clases / une x e y en una tabla
    datos["class"] = y["class"]
    # Convierte la tabla en una lista de diccionarios
    return datos.to_dict(orient="records")
 
# ─────────────────────────────────────────────────────────────
#  DIVISIÓN DEL DATASET
# ─────────────────────────────────────────────────────────────
 
def dividir_dataset(datos, clase_attr, test_ratio=0.2, semilla=42):
    """ Divide el dataset en entrenamiento (80%) y test (20%)."""
    random.seed(semilla)
    por_clase = {}
    for inst in datos:
        """ separamos las instancias por clase """
        c = inst[clase_attr]
        por_clase.setdefault(c, []).append(inst)
 
    entrenamiento, test = [], []
    for clase, instancias in sorted(por_clase.items()):
        random.shuffle(instancias)
        """ calculamos cuales instancias van a test """
        n_test = max(1, round(len(instancias) * test_ratio))
        test.extend(instancias[:n_test])
        entrenamiento.extend(instancias[n_test:])
 
    random.shuffle(entrenamiento)
    random.shuffle(test)
    return entrenamiento, test
 
def guardar_csv(datos, path):
    """Guarda una lista de dicts en un archivo CSV."""
    import csv
    if not datos:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=datos[0].keys())
        writer.writeheader()
        writer.writerows(datos)
    print(f"  → Guardado: {path} ({len(datos)} instancias)")

# ─────────────────────────────────────────────────────────────
#  FUNCIÓN DE ENTROPIA
# ─────────────────────────────────────────────────────────────

def entropia(datos, clase_attr):
    # datos --> lista de instancias
    # clase_attr --> nombre del atributo de la clase 
    n = len(datos) # cant total
    if n == 0:
        return 0.0 # si no hay nada, retorna 0
    conteo = Counter(ints[clase_attr] for inst in datos) # cuantas veces aparece cada clase 
    # Counter({1: 3, 2: 2, 3: 1})
    ent = 0.0
    for count in conteo.values(): # recorre cantidades de cada clase
        p = count / n_test # probabilidad de cada clase
        if p > 0 # evitamos log 0
           ent += p * math.log2(1/p) # suma la contribucion de cada clase
    return ent

# ─────────────────────────────────────────────────────────────
#  FUNCIÓN DE GAIN RATIO
# ─────────────────────────────────────────────────────────────

def gain_ratio(datos, atributo, clase_attr, umbral):

    n = len(datos)

    # Dividir los datos
    izq = [d for d in datos if d[atributo] <= umbral]
    der = [d for d in datos if d[atributo] > umbral]

    # Entropía del conjunto original H(D)
    ent_padre = entropia(datos, clase_attr)

    # Probabilidades de cada rama
    p_izq = len(izq) / n
    p_der = len(der) / n

    # Entropía ponderada de las ramas
    ent_hijos = (
        p_izq * entropia(izq, clase_attr)
        + p_der * entropia(der, clase_attr)
    )

    # Gain(D,S)
    gain = ent_padre - ent_hijos

    # SplitInfo(D,S)
    split_info = 0

    if p_izq > 0:
        split_info -= p_izq * log2(p_izq)

    if p_der > 0:
        split_info -= p_der * log2(p_der)

    # Evitar división por cero
    if split_info == 0:
        return 0

    # GainRatio(D,S)
    return gain / split_info

# ─────────────────────────────────────────────────────────────
#  FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────
 
def main():
 
    # 1. Cargar datos desde UCI
    print("\n[1] Descargando dataset Wine desde UCI ML Repository...")
    datos = cargar_wine()
    CLASE = "class"
    atributos = [k for k in datos[0].keys() if k != CLASE]
    clases = sorted(set(d[CLASE] for d in datos))
 
    print(f"  Total instancias : {len(datos)}")
    print(f"  Atributos        : {len(atributos)}")
    print(f"  Clases           : {[int(c) for c in clases]}")
    dist = Counter(d[CLASE] for d in datos)
    for c in clases:
        print(f"    Clase {int(c)}: {dist[c]} instancias")
 
    # 2. Dividir dataset (80/20)
    print("\n[2] Dividiendo dataset (80% entrenamiento / 20% test)...")
    entrenamiento, test = dividir_dataset(datos, CLASE, test_ratio=0.2, semilla=42)
    print(f"  Entrenamiento    : {len(entrenamiento)} instancias")
    print(f"  Test             : {len(test)} instancias")
    dist_train = Counter(d[CLASE] for d in entrenamiento)
    dist_test  = Counter(d[CLASE] for d in test)
    print("  Distribución entrenamiento:", {int(k): v for k, v in sorted(dist_train.items())})
    print("  Distribución test         :",{int(k): v for k, v in sorted(dist_test.items())})
 
    guardar_csv(entrenamiento, "wine_train.csv")
    guardar_csv(test,          "wine_test.csv")
 
if __name__ == "__main__":
    main()