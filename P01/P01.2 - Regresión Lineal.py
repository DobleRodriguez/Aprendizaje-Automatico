# Aprendizaje Automático - Grupo A2
# Universidad de Granada - Grado en Ingeniería Informática - Curso 2019/2020
# Javier Rodríguez Rodríguez - @doblerodriguez


#------------------------------- Práctica 01 - Ejercicio 02 ---------------------------------#
#---------------------------- EJERCICIO SOBRE REGRESIÓN LINEAL ------------------------------#

import pathlib as pl

import matplotlib.pyplot as plt
import numpy as np

def read_data(x_file, y_file):
    # Cargamos la información de los ficheros mediante NumPy
    x_data = np.load(x_file)
    y_data = np.load(y_file)

    # En X guardamos las instancias etiquetadas con 1 y 5
    x = x_data[(y_data == 1) | (y_data == 5)]
    x = np.hstack((np.ones((len(x), 1)), x))
    # En y guardamos etiquetas para dichas instancias, -1 si es un 1 y 1 si es un 5
    y = np.select([y_data == 1, y_data==5], [-1, 1])
    y = y[y != 0]
    return x, y

def pseudoinverse_method(x, y):
    x_pinv = np.linalg.pinv(x)
    return np.matmul(x_pinv, y)

def stochastic_gradient_descent(x, y, lr, max_iters, batch_size):
    
    pass

x_train, y_train = read_data(pl.Path(__file__).parent / f"datos/X_train.npy", 
    pl.Path(__file__).parent / f"datos/y_train.npy")

pseudoinverse_method(x_train, y_train)
