# Aprendizaje Automático - Grupo A2
# Universidad de Granada - Grado en Ingeniería Informática - Curso 2019/2020
# Javier Rodríguez Rodríguez - @doblerodriguez


#------------------------------- Práctica 03 - Ejercicio 01 ---------------------------------#
#------------------------ Optical Recognition of Handwritten Digits -------------------------#

# 1.
# La base de datos está descrita en .names
# Las columnas son numéricas enteras entre 0 y 16 y representan distintos parámetros del bitmap de cada dígito 
# La variable de clase es numérica, entre el intervalo 0..9 (el dígito que representan)
# Se trata de un problema de aprendizaje supervisado, pues los datos de entrenamiento están etiquetados
# Es un problema de clasificación

# 2.
# Funciones lineales: son fáciles, están en sklearn, y sirven.

# 3.
# Train y Test vienen directamente particionadas en los datos.

# 4.
# Los datos vienen preprocesados

# 5.
# Precisión (accuracy_score)

# 6.
# SAGA y SGD

# 7.
# L1

# 8.
# Regresión logística
# Perceptron solo por interés académico

# 9.w
# Probando varias combinaciones mediante CV 

# 10.
# 1-score

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model 
from sklearn import model_selection
from warnings import filterwarnings

# Evita que advertencias de sklearn se impriman en pantalla
#filterwarnings('ignore')

# Función auxiliar para detener la ejecución del script entre cada apartado
def stop():
    input("\nPulse ENTER para continuar\n")

# Extraemos los datos de los ficheros de la fuente
train = np.genfromtxt(Path(__file__).parent / f"datos/optdigits.tra", delimiter=',')
test = np.genfromtxt(Path(__file__).parent / f"datos/optdigits.tes", delimiter=',')

# Separamos las variables clasificatorias de la variable de clase en ambos conjuntos
X_train = train[:,:-1]
y_train = train[:,-1]
X_test = test[:,:-1]
y_test = test[:,-1]

# Comprobamos que las clases estén balanceadas, contando y graficando mediante barras las ocurrencias
# de cada clase
unique, count = np.unique(y_train, return_counts=True)
plt.bar(unique, count)
plt.xlabel("Dígito")
plt.ylabel("Ocurrencias")
plt.title("Balance de clases en conjunto de entrenamiento")
plt.show()
stop()

# Mismo procedimiento para el conjunto Test
unique, count = np.unique(y_test, return_counts=True)
plt.bar(unique, count)
plt.xlabel("Dígito")
plt.ylabel("Ocurrencias")
plt.title("Balance de clases en conjunto de Test")
plt.show()
stop()

# Calculamos la correlación entre los datos de entrenamiento para decidir qué regularización usar
# rowvar=False traspone la matriz (cada característica es una fila)
correlation = np.corrcoef(X_train, rowvar=False)
correlation = np.nan_to_num(correlation)

# Grafica un mapa de calor con la correlación. Los valores NaN se sustituyen por 0s (la característica es constante)
plt.imshow(correlation)
plt.colorbar()
plt.title("Correlación entre características")
plt.show()
stop()

# Distintos hiper-parámetros a probar
parameters = {"alpha" : [0.0001, 0.001, 0.01, 0.1]}


# Pruebo los distintos modelos y técnicas, haciendo CV con los hiperparámetros para cada uno de ellos
# Perceptron() es una implementación directa de sklearn de SGD para el modelo Perceptrón
# LogisticRegressionCV es un módulo que implementa directamente CV sobre este modelo
# por lo que no hace falta utilizar GridSearch y explorar manualmente los hiperparámetros
print("Calculando la precisión de cada técnica y modelo con distintos hiperparámetros")
classifiers = [("SGD (RL)", model_selection.GridSearchCV(linear_model.SGDClassifier(random_state=0, loss='log', penalty='l1'), parameters)), 
                ("SGD (Perceptron)", model_selection.GridSearchCV(linear_model.Perceptron(random_state=0, penalty='l1'), parameters)),
                ("SAGA (RL)", linear_model.LogisticRegressionCV(solver='saga', random_state=0, penalty='l1'))
                ]

# Comprobamos los resultados
for name, clf in classifiers:
    clf = clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(f"Precisión de {name} con los mejores hiper-parámetros: {score}")

stop()

