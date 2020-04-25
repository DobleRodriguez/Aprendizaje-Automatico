# Aprendizaje Automático - Grupo A2
# Universidad de Granada - Grado en Ingeniería Informática - Curso 2019/2020
# Javier Rodríguez Rodríguez - @doblerodriguez


#------------------------------- Práctica 02 - Ejercicio 03 ---------------------------------#
#----------------------------------------- BONUS --------------------------------------------#

import pathlib as pl

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

# Apartado 3) Clasificación de Dígitos ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Función auxiliar para detener la ejecución del script entre cada apartado
def stop():
    input("\nPulse ENTER para continuar\n")

def read_data(x_file, y_file):
    # Función de carga de la información de los ficheros mediante NumPy
    x_data = np.load(x_file)
    y_data = np.load(y_file)

    # En X guardamos las instancias etiquetadas con 1 y 5
    x = x_data[(y_data == 4) | (y_data == 8)]
    # Agregamos columna de unos para el término independiente
    x = np.hstack((np.ones((len(x), 1)), x))
    # En y guardamos etiquetas para dichas instancias, -1 si es un 1 y 1 si es un 5
    y = np.select([y_data == 4, y_data==8], [-1, 1])
    y = y[y != 0]
    return x, y

# a) El problema consiste en clasificar los puntos dados (intensidad promedio y simetría) como -1 si se
# trata de un 4 y como 1 si se trata de un 8. 

# b) 
def pla_pocket(datos, label, max_iters, vini):
    # Igual que PLA del ejercicio 2
    w = np.copy(vini)
    # Pero hacemos una copia adicional, pocket_w
    pocket_w = np.copy(vini)
    change = True
    iters = 0
    while (iters < max_iters and change):
        change = False
        # Hacemos una primera iteración de PLA
        for i in np.arange(len(label)):
            if (np.sign(np.dot(w, datos[i,:])) != label[i]):
                w = w + label[i]*datos[i,:]
                change = True
        # Calculamos el error respecto a w y pocket_w
        ein_w = np.count_nonzero(np.sign(np.matmul(datos, w)) != label)/label.size
        ein_pocket = np.count_nonzero(np.sign(np.matmul(datos, pocket_w)) != label)/label.size
        # Si el error de w es menor, actualizamos pocket_w
        if (ein_w < ein_pocket):
            pocket_w = np.copy(w)
        iters+=1
    return pocket_w, iters

# Código auxiliar similar al del ejercicio 2
def graph_points_frontier(x,y,label,a,b,c=-1,title=""):
    for i in np.unique(label):
        plt.scatter(x[label == i], y[label == i], label=i)
    # Graficamos la recta, donde x coincide con los valores de x en el intervalo, e y está determinado
    # por y = ax + b
    plt.autoscale(axis="x", tight=True)
    xlims = plt.gca().get_xlim()
    xlist = np.linspace(xlims[0], xlims[1], 1000)
    plt.gca().set_ylim(plt.gca().get_ylim())
    plt.plot(xlist, -(a*xlist + b)/c, label="Función calculada", c='green')
    # Ajustamos la gráfica para graficar correctamente la línea
    # Etiquetamos los ejes y la leyenda
    plt.legend()
    plt.title(title)
    plt.xlabel("Intensidad promedio")
    plt.ylabel("Simetría")
    plt.show()

# 1)
# Extraemos los datos al igual que hicimos en el ejercicio 3 de la práctica 1
x_train, y_train = read_data(pl.Path(__file__).parent / f"datos/X_train.npy", 
    pl.Path(__file__).parent / f"datos/y_train.npy")

x_test, y_test = read_data(pl.Path(__file__).parent / f"datos/X_test.npy", 
    pl.Path(__file__).parent / f"datos/y_test.npy")

# Aplicamos PLA_Pocket
w, iters  = pla_pocket(x_train, y_train, 500, np.zeros(x_train.shape[1]))

# Graficamos la función ajustada con el conjunto train
graph_points_frontier(x_train[:,1], x_train[:,2], y_train, w[1], w[0], w[2], "PLA_Pocket sobre datos de entrenamiento")
stop()
# Y el conjunto test
graph_points_frontier(x_test[:,1], x_test[:,2], y_test, w[1], w[0], w[2], "PLA_Pocket sobre datos de test")
stop()
# Calculamos los errores respecto al conjunto train y test respectivamente
ein = np.count_nonzero(np.sign(np.matmul(x_train, w)) != y_train)/y_train.size
etest = np.count_nonzero(np.sign(np.matmul(x_test, w)) != y_test)/y_test.size
print(f"El error de la función respecto a los datos de train es {ein}")
print(f"El error de la función respecto a los datos de test es {etest}")
stop()
# 3) Calculamos la cota máxima de Eout para cada muestra con delta = 0.05
eout_in = ein + np.sqrt(1/(2*y_train.size)*np.log(2/0.05))
eout_test = etest + np.sqrt(np.log(2/0.05)/(2*y_test.size))
print("Con tolerancia delta = 0.05")
print(f"La cota máxima de Eout basada en Ein es {eout_in}")
print(f"La cota máxima de Eout basada en Etest es {eout_test}")
stop()