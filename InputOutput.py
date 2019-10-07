'''
    dos funciones la primera verifica que la ruta del archivo exista y se puede abrir
    y la segunda funcion emplea la verificación del archivo y si es verdadero retorna el archivo de lo contrario retorna nada
'''
def verificarArchivo(archivo)->bool:
    puede=False
    try:
        archivo=open(archivo,"r")
        puede=True
        print(puede)
    except:
        print("Imposible leer el archivo verifique la dirección")
    return puede
def leerArchivo(archivo):
    if verificarArchivo(archivo):
        return open(archivo,"r")
    return None

