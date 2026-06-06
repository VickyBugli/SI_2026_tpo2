from ucimlrepo import fetch_ucirepo
import math
import random
from collections import Counter
 
# ─────────────────────────────────────────────────────────────
#  CARGA DEL DATASET
# ─────────────────────────────────────────────────────────────
 
def cargar_wine():
    wine = fetch_ucirepo(id=109)
    X = wine.data.features
    y = wine.data.targets
    datos = X.copy()
    # Agrega columna de clases
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
    entrenamiento, test = dividir_dataset(datos, CLASE,
                                          test_ratio=0.2, semilla=42)
    print(f"  Entrenamiento    : {len(entrenamiento)} instancias")
    print(f"  Test             : {len(test)} instancias")
    dist_train = Counter(d[CLASE] for d in entrenamiento)
    dist_test  = Counter(d[CLASE] for d in test)
    print("  Distribución entrenamiento:",
          {int(k): v for k, v in sorted(dist_train.items())})
    print("  Distribución test         :",
          {int(k): v for k, v in sorted(dist_test.items())})
 
    guardar_csv(entrenamiento, "wine_train.csv")
    guardar_csv(test,          "wine_test.csv")
 
if __name__ == "__main__":
    main()