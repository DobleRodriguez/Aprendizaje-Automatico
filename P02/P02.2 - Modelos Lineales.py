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
    #input("\nPulse ENTER para continuar\n")
    pass

# Apartado a) Algoritmo Perceptron
def ajusta_PLA(datos, label, max_iters, vini):
    iters = 0
    w = np.copy(vini)
    change = True
    while (iters < max_iters and change):
        change = False
        for i in np.arange(len(label)):
            if (np.sign(np.dot(w, datos[i,:])) != label[i]):
                w = w + label[i]*datos[i,:]
                change = True
        iters+=1
    return w, iters

# 1) 
# Generamos los datos simulados en el apartado 2a de la sección 1

# Utilizamos las funciones auxiliares especificadas
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

# Generamos directamente los valores para el apartado 2) (y así asegurar que coincidan con los del ejercicio 1)
noised_y = np.copy(y)

# Por cada uno de los valores de etiqueta
for i in np.unique(y):
    # Seleccionamos el 10% de las etiquetas con un valor
    noised_indices = np.random.choice(np.flatnonzero(y == i), int(np.rint(y[y == i].size * 0.10)), replace=False)
    # Las cambiamos en el vector copia
    noised_y[noised_indices] = -y[noised_indices]

# Función auxiliar para graficar los datos junto a la frontera
def graph_points_frontier(x,y,label,a,b,c=-1,title=""):
    for i in np.unique(label):
        plt.scatter(x[label == i], y[label == i], label=i)
    # Graficamos la recta, donde x coincide con los valores de x en el intervalo, e y está determinado
    # por y = ax + b
    plt.autoscale(axis="x", tight=True)
    xlist = np.linspace(-50, 50, 1000)
    plt.gca().set_ylim(plt.gca().get_ylim())
    plt.plot(xlist, -(a*xlist + b)/c, label="Recta de separación", c='green')
    # Ajustamos la gráfica para graficar correctamente la línea
    # Etiquetamos los ejes y la leyenda
    plt.legend()
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()



# Graficamos los datos iniciales, similar a los simulados en el apartado 2a del ejercicio 1
graph_points_frontier(x_unif[:,0], x_unif[:,1], y, a, b, title="Nube de puntos etiquetados, con línea de frontera")

# Concatenamos una columna de 1 a los datos para ejecutar correctamente PLA
x_unif = np.concatenate((np.ones((x_unif.shape[0],1)), x_unif), axis=1)


# Generamos un vector para almacenar el número de iteraciones de cada ejecución
niters_zero = np.empty(10)
niters_rand = np.empty(10)

# Ejecutamos PLA con distintos parámetros de inicialización
for i in np.arange(10):
    w_zero = np.zeros(x_unif.shape[1])
    w, niters_zero[i] = ajusta_PLA(x_unif, y, 10000, np.zeros(x_unif.shape[1]))
    graph_points_frontier(x_unif[:,1], x_unif[:,2], y, w[1], w[0], w[2], 
    "Nube de puntos etiquetados, con frontera\ncalculada mediante PLA con entrada 0")
    stop()
    print(f"El algoritmo converge en {niters_zero[i]} iteraciones")
    stop()

    w_rand = np.random.uniform(size=x_unif.shape[1])
    w, niters_rand[i] = ajusta_PLA(x_unif, y, 10000, w_rand)
    graph_points_frontier(x_unif[:,1], x_unif[:,2], y, w[1], w[0], w[2], 
    "Nube de puntos et iquetados, con frontera\ncalculada mediante PLA con entrada aleatoria en [0,1]")
    stop()
    print(f"El algoritmo converge en {niters_rand[i]} iteraciones")
    stop()


print(f"En promedio, para w inicial de vector de ceros, toma {np.mean(niters_zero)} iteraciones en converger")
print(f"En promedio, para w inicial de vector de valores aleatorios entre 0 y 1, toma {np.mean(niters_rand)} iteraciones en converger")
stop()

# 2)
print("Aplicamos ruido al 10% de los datos")
for i in np.arange(10):
    w_zero = np.zeros(x_unif.shape[1])
    w, niters_zero[i] = ajusta_PLA(x_unif, noised_y, 10000, np.zeros(x_unif.shape[1]))
    graph_points_frontier(x_unif[:,1], x_unif[:,2], noised_y, w[1], w[0], w[2], 
    "Nube de puntos etiquetados, con frontera\ncalculada mediante PLA con entrada 0")
    stop()
    print(f"El algoritmo converge en {niters_zero[i]} iteraciones")
    stop()

    w_rand = np.random.uniform(size=x_unif.shape[1])
    w, niters_rand[i] = ajusta_PLA(x_unif, noised_y, 10000, w_rand)
    graph_points_frontier(x_unif[:,1], x_unif[:,2], noised_y, w[1], w[0], w[2], 
    "Nube de puntos et iquetados, con frontera\ncalculada mediante PLA con entrada aleatoria en [0,1]")
    stop()
    print(f"El algoritmo converge en {niters_rand[i]} iteraciones")
    stop()


print(f"En promedio, para w inicial de vector de ceros, toma {np.mean(niters_zero)} iteraciones en converger")
print(f"En promedio, para w inicial de vector de valores aleatorios entre 0 y 1, toma {np.mean(niters_rand)} iteraciones en converger")
stop()

# Apartado b) Regresión Logística

