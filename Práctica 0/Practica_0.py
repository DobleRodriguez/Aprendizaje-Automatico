# Práctica 0
# Aprendizaje Automático - Grupo A2
# Universidad de Granada - Grado en Ingeniería Informática - Curso 2019/2020
# Javier Rodríguez Rodríguez - @doblerodriguez

import matplotlib.pyplot
import numpy
import sklearn.datasets

# Parte 1

# Importamos desde scikit-learn la base de datos de iris
iris = sklearn.datasets.load_iris()

# Obtenemos las características (X) y la clase (y)
X = iris.data
y = iris.target

# Nos quedamos con los datos de interés: las últimas dos 
X = X[:, [-2, -1]]

# Separamos las características en distintos arrays según la categoría a la que pertenezcan

splitter, = numpy.where(y[:-1] != y[1:])
X = numpy.split(X, splitter + 1)

colores = ['red', 'green', 'blue']

for clase in range(0, len(X)):
    print(colores[clase])
    matplotlib.pyplot.scatter(X[0], X[1], c=colores[clase])
    
matplotlib.pyplot.show()

