#------------------------------- Práctica 01 - Ejercicio 01 ---------------------------------#
#---------------------EJERCICIO SOBRE LA BÚSQUEDA ITERATIVA DE ÓPTIMOS ----------------------#
# Aprendizaje Automático - Grupo A2
# Universidad de Granada - Grado en Ingeniería Informática - Curso 2019/2020
# Javier Rodríguez Rodríguez - @doblerodriguez

import matplotlib.pyplot as plt
import numpy as np

#-------------------------------------- Apartado 1 ------------------------------------------#
# Implementar el algoritmo de gradiente descendente (1 punto)


# Hola Javier, de vuelta a tus comentarios monólogos
# Esta es la función a implementar realmente. Las otras cosas no son más que parámetros
# w = datos
# lr = learning rate
# grad_func = gradiente de la función
# func = función
# epsilon = threshold para detener iteración
# max_iters = máximas iteraciones

# Qué hace este peazo'e código feo
# La idea es alcanzar el valor mínimo (local) de una función, lo más "rápido" posible,
# pero con el cuidado de tomar un lr suficientemente pequeño como para alcanzar realmente ese
# mínimo.
# Para eso, restamos al valor de partida de la función su derivada por la tasa de aprendizaje,
# hacemos de este resultado el valor de partida de la función y repetimos el proceso hasta que 
# la disminución sea por debajo de épsilon, y por tanto, negligible 
# (cuando la derivada ~= 0 -> está en un mínimo local)
def gradient_descent(w, lr, grad_func, func, epsilon, max_iters=10000):
    step = epsilon
    i = 0
    while((i < max_iters) & (step < epsilon)):
        next_w = w - lr * grad_func(w)
        step = np.linalg.norm(next_w - w)
        w = next_w
        i += 1
    return w, i

def E(w):
    u = w[0]
    v = w[1]
    e = np.e
    return (u * e**v - (2 * v * e**-u))**2

def Eu(w):
    u = w[0]
    v = w[1]
    e = np.e
    return 2 * (u * e**v - 2 * v * e**-u) * (e**v + 2 * v * e**-u)

def Ev(w):
    u = w[0]
    v = w[1]
    e = np.e
    return 2 * (u * e**v - 2 * v * e**-u) * (u * e**v - 2 * e**-u)

def gradE(w):
    return np.array([Eu(w), Ev(w)])

