# Práctica 0
# Aprendizaje Automático - Grupo A2
# Universidad de Granada - Grado en Ingeniería Informática - Curso 2019/2020
# Javier Rodríguez Rodríguez - @doblerodriguez

import matplotlib.pyplot as mpl 
import numpy
import sklearn.datasets
import sklearn.model_selection

################################## Parte 1 #################################################
print("Parte 1:")

# Importamos desde scikit-learn la base de datos de iris
iris = sklearn.datasets.load_iris()

# Obtenemos las características (X) y la clase (y)
X = iris.data
y = iris.target

# Nos quedamos con los datos de interés: las últimas dos 
X_ult2 = X[:, [-2, -1]]

# Separamos las características en distintos arrays según la categoría a la que pertenezcan
# Para ello, encontramos los índices en los que los valores consecutivos dejan de ser iguales
# (cambia la clase), y particionamos el array en estos índices.
splitter, = numpy.where(y[:-1] != y[1:])
X_ult2 = numpy.split(X_ult2, splitter + 1)

# Obtenemos el nombre de las características, las clases y creamos una lista de colores. 
# Estos datos servirán para identificar adecuadamente el gráfico
clases = iris.target_names
caracteristicas = iris.feature_names
colores = ['red', 'green', 'blue']

# Graficamos cada clase, tomando en cada iteración su color, las características pertenecientes
# a la clase, y el nombre de dicha clase. Para ello utilizamos la función scatter
for color, datos_clase, nombre_clase in zip(colores, X_ult2, clases):
    mpl.scatter(datos_clase[:,0], datos_clase[:,1], c=color, label=nombre_clase)

# Le indicamos a matplotlib que grafique la leyenda, así como etiquetas en ambos ejes y título
mpl.legend()
mpl.xlabel(caracteristicas[-2])
mpl.ylabel(caracteristicas[-1])
mpl.title("Parte 1")

# Mostramos en pantalla el gráfico
mpl.show()


############################################# Parte 2 ##########################################
input("Pulse enter para continuar")
print("Parte 2:")

# Separamos los datos, tanto en X como en y, utilizados para training y test. Definimos el tamaño
# de test a 20% (y por ende, training en 80%). Stratify=y hace que se conserve la proporción de 
# las clases en ambos conjuntos
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=.2, stratify=y)
# Se imprime por pantalla en detalle esta división, comprobando la correctitud de las proporciones
# y las proporciones de clases
print("Separación en training y test.")
print("El conjunto training tiene " + str(len(y_train)) + " elementos, que corresponde al " + str(100*len(y_train)/len(y))+ "% del conjunto")
print("En el conjunto Training hay: ")
for i in range(3):
    print("\t" + str(numpy.count_nonzero(y_train == i)) + " elementos de la clase " + clases[i])
print("Conservando así la proporción de las clases")

############################################# Parte 3 ###########################################
input("Pulse enter para continuar")
print("Parte 3:")

# Creamos el intervalo de valores equiespaciados.
valores = numpy.linspace(0, 2*numpy.pi, 100)
# Definimos una lista de 3 arrays, donde cada uno almacena el resultado de su respectiva 
# operación para los 100 valores
trigonometricas = [numpy.sin(valores), numpy.cos(valores), numpy.sin(valores) + numpy.cos(valores)]
# Creamos listas para datos de formato del gráfico y su leyenda"
formas_linea = ['k--', 'b--', 'r--']
nombres_func = ['sin(x)', 'cos(x)', 'sin(x) + cos(x)']

# Generamos y mostramos por pantalla el diagrama
for funcion, linea, nombre_func in zip(trigonometricas, formas_linea, nombres_func):
    mpl.plot(valores, funcion, linea, label=nombre_func)

mpl.legend()
mpl.title("Parte 3")
mpl.show()




