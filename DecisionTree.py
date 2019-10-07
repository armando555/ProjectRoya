from sklearn import tree
from sklearn.tree.export import export_text
import InputOutput as rg
import numpy as np
from time import time
from LinkedList import LinkedList as Lista
'''
    features es la lista de cada linea con sus respectivos elementos
    labels es la etiqueta que contiene yes o no para determinar si tiene roya o no
'''
features=[]
labels=[]
direccion="/home/armando55/Escritorio/Clases_Datos/Proyecto/data_set.csv"
tiempoInicial=time()
'''
    el ciclo toma cada linea hace un split de ella y la lleva a una lista que se agrega
    a la lista features, pero antes le quita la etiqueta "yes" o "no" para poder
    emplear el machine learning después, y esta etiqueta se guarda en labels
'''
# la función leerArchivo fue creada por mi para verificar que el archivo existe e abrirlo
with rg.leerArchivo(direccion) as archivo:
    for linea in archivo:
        linea=linea.split(",")
        if linea[6]=="yes\n":
            linea[6]=1
        else:
            linea[6]=0
        labels.append(linea.pop(6))
        features.append(linea)
features.pop(0)
labels.pop(0)
print(features)
print(labels)

'''
    aca creo el arbol de decisiones que va clasificar los features a las labels
    empleo la biblioteca sklearn para crear el arbol y preparo la información para aplicarle el machine learning
'''
clf=tree.DecisionTreeClassifier(random_state=0)
clf=clf.fit(features,labels)
print("RESULTADO DEL CLASIFICADOR:")
print(clf.predict([features[0]]))



'''
    se crean unas listas con los respectivos nombres de los features y labels y empleo export_text para escribir el arbol en una variable e imprimirla

'''
classes=["yes","no"]
datos=["ph","soil_temperature","soil_moisture","illuminance","env_temperature","env_humidity"]
r=export_text(clf,feature_names=datos)
print(r)
tiempoFinal=time()
ejecucion=tiempoFinal-tiempoInicial

#--------------------------------------------------------------------------------------MEMORY IN MB--------------------------------
def memoryUsage():
    # return the memory usage in MB
    import psutil
    import os
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / float(2 ** 20)
    return mem




#-------------------------------------------Linked List--------------------------------
lista=Lista()
count=0
timeAddI=time()
for feature in features:
    listaN=feature
    listaN.append(labels[count])
    lista.add(listaN)
    count=count+1
timeAddF=time()
timeAdd=timeAddF-timeAddI
timeImprimirI=time()
lista.print()
timeImprimirF=time()
tiempoImprimir=timeImprimirF-timeImprimirI
print()
print()
timeSearchI=time()
print(lista.search(features[50]).data)
timeSearchF=time()
timeSearch=timeSearchF-timeSearchI

timeDeleteI=time()
lista.delete(features[0])
timeDeleteF=time()
timeDelete=timeDeleteF-timeDeleteI
memoria=memoryUsage()
print("------------------------------------------------ aca hice delete")
lista.print()
#ESTE ES EL TIEMPO DE EJECUCIÓN DEL DECISION TREE
print()
print("TIEMPO DE EJECUCION DEL DECISION TREE")
print(ejecucion)
#ESTE ES EL TIEMPO DE EJECUCIÓN DEL PRINT LIST
print()
print("TIEMPO DE EJECUCION DEL PRINT LIST: ")
print(tiempoImprimir)
#ESTE ES EL TIEMPO DE EJECUCIÓN DEL ADD LIST
print()
print("TIEMPO DE EJECUCION DEL ADD LIST: ")
print(timeAdd)
#ESTE ES EL TIEMPO DE EJECUCIÓN DEL SEARCH LIST
print()
print("TIEMPO DE EJECUCION DEL SEARCH LIST: ")
print(timeSearch)
#ESTE ES EL TIEMPO DE EJECUCIÓN DEL DELETE LIST
print()
print("TIEMPO DE EJECUCION DEL DELETE LIST: ")
print(timeDelete)
#MEMORY IN MB
print()
print("MEMORIA EN MB: \n",memoria)
print()
#PROMEDIO DE LOS TIEMPOS
promedio=(timeDelete+timeSearch+tiempoImprimir+timeAdd)/4
print("EL PROMEDIO DE LOS TIEMPOS ES: \n",promedio)