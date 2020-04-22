# Aprendizaje Automático - Grupo A2
# Universidad de Granada - Grado en Ingeniería Informática - Curso 2019/2020
# Javier Rodríguez Rodríguez - @doblerodriguez


#------------------------------- Práctica 02 - Ejercicio 01 ---------------------------------#
#------------------------------------- MODELOS LINEALES -------------------------------------#

import matplotlib.pyplot as plt
import numpy as np

# Fijamos la semilla
np.random.seed(1)

# Función auxiliar para detener la ejecución del script entre cada apartado
def stop():
    input("\nPulse ENTER para continuar\n")

# Apartado a) Algoritmo Perceptron
# Generamos el vector de pares (x,y) a través de simula_unif según los parámetros especificados
def ajusta_PLA(datos, label, max_iters, vini):
    iters = 0
    w = np.copy(vini)
    change = True
    while (iters < max_iters or not change):
        for i in np.arange(len(datos[:,0])):
            print(w.reshape(1,-1).shape)
            print(datos[:,i].shape)
            if (np.sign(np.dot(w.reshape(1,-1), datos[:,i])) != label[i]):
                w = w + label[i]*datos[:,i]
                change = True
        iters+=1
    print(iters)
    return w

# 1) 
# Generamos los datos simulados en el apartado 2a de la sección 1
def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b

# Generamos los puntos a través de simula_unif con los parámetros especificados
x_unif = simula_unif(100, 2, [-50,50])
# Simulamos la recta a través de simula_recta en el mismo intervalo
a,b = simula_recta([-50,50])
# Determinamos las etiquetas a partir del valor de la función signo de la distancia entre los puntos
# y la recta, f(x,y) = y - ax - b
y = np.sign(x_unif[:,1] - a*x_unif[:,0] - b)

# Ejecutamos PLA con distintos parámetros de inicialización
w = ajusta_PLA(x_unif, y, 10000, np.zeros(y.size))