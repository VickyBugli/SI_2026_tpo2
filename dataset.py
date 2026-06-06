from ucimlrepo import fetch_ucirepo 
  
def cargar_wine():
    wine = fetch_ucirepo(id=109)

    X = wine.data.features
    y = wine.data.targets

    datos = X.copy()
    #""agrega columna de clases""
    datos["class"] = y["class"]
    
    #""convierte la tabla en una lista""
    return datos.to_dict(orient="records")

