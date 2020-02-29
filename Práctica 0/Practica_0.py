# Práctica 0
# Aprendizaje Automático - Grupo A2
# Universidad de Granada - Grado en Ingeniería Informática - Curso 2019/2020
# Javier Rodríguez Rodríguez - @doblerodriguez

import matplotlib.pyplot as mpl 
import numpy
import sklearn.datasets
import sklearn.model_selection

# Parte 1
print("Parte 1:")

# Importamos desde scikit-learn la base de datos de iris
iris = sklearn.datasets.load_iris()

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

# Graficamos cada clase, tomando en cada iteración su color, las características pertenecientes
# a la clase, y el nombre de dicha clase. Para ello utilizamos la función scatter
for color, datos_clase, nombre_clase in zip(colores, X, clases):
    mpl.scatter(datos_clase[:,0], datos_clase[:,1], c=color, label=nombre_clase)

# Le indicamos a matplotlib que grafique la leyenda, así como etiquetas en ambos ejes y título
mpl.legend()
mpl.xlabel(caracteristicas[-2])
mpl.ylabel(caracteristicas[-1])
mpl.title("Parte 1")

# Mostramos en pantalla el gráfico
mpl.show(block=False)


# Parte 2
input("Pulse enter para continuar")
print("Parte 2:")

# Reiniciamos X a los valores originales
X = iris.data

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=.2, stratify=y)
print("Separación en training y test.")
print("El conjunto training tiene " + str(len(y_train)) + " elementos, que corresponde al " + str(100*len(y_train)/len(y))+ "% del conjunto")
print("En el conjunto Training hay: ")
for i in range(3):
    print("\t" + str(numpy.count_nonzero(y_train == i)) + " elementos de la clase " + clases[i])

# Parte 3
print("\n")


