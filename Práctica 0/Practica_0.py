# Práctica 0
# Aprendizaje Automático - Grupo A2
# Universidad de Granada - Grado en Ingeniería Informática - Curso 2019/2020
# Javier Rodríguez Rodríguez - @doblerodriguez

import matplotlib.pyplot as mpl 
import numpy
import sklearn.datasets

# Parte 1

# Importamos desde scikit-learn la base de datos de iris
iris = sklearn.datasets.load_iris()
print(iris)

# Obtenemos las características (X) y la clase (y)
X = iris.data
y = iris.target

# Nos quedamos con los datos de interés: las últimas dos 
X = X[:, [-2, -1]]

# Separamos las características en distintos arrays según la categoría a la que pertenezcan
# Para ello, encontramos los índices en los que los valores consecutivos dejan de ser iguales
# (cambia la clase), y particionamos el array en estos índices.
splitter, = numpy.where(y[:-1] != y[1:])
X = numpy.split(X, splitter + 1)

# Obtenemos el nombre de las características, las clases y creamos una lista de colores. 
# Estos datos servirán para identificar adecuadamente el gráfico
clases = iris.target_names
caracteristicas = iris.feature_names
colores = ['red', 'green', 'blue']


for color, datos_clase, nombre_clase in zip(colores, X, clases):
    mpl.scatter(datos_clase[:,0], datos_clase[:,1], c=color, label=nombre_clase)
mpl.legend()
mpl.xlabel(caracteristicas[-2])
mpl.ylabel(caracteristicas[-1])
mpl.title("Parte 1")

mpl.show()
    

