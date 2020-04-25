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
    pass

# Apartado a) Algoritmo Perceptron ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    xlims = plt.gca().get_xlim()
    xlist = np.linspace(xlims[0], xlims[1], 1000)
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
    print(w_zero)
    w, niters_zero[i] = ajusta_PLA(x_unif, y, 10000, w_zero)

    w_rand = np.random.uniform(size=x_unif.shape[1])
    print(w_rand)
    w, niters_rand[i] = ajusta_PLA(x_unif, y, 10000, w_rand)

print(f"Para cada ejecución de PLA con w inicial = vector de ceros, tomó {niters_zero} iteraciones en converger\nEn promedio, {np.mean(niters_zero)} iteraciones")
stop()
print(f"Para cada ejecución de PLA con w inicial = vector de valores aleatorios en [0,1], tomó {niters_rand} iteraciones en converger\nEn promedio, {np.mean(niters_rand)} iteraciones")

# 2)
print("Aplicamos ruido al 10% de los datos")
# Repetimos el experimento anterior, pero con las etiquetas con ruido
for i in np.arange(10):
    w_zero = np.zeros(x_unif.shape[1])
    w, niters_zero[i] = ajusta_PLA(x_unif, noised_y, 10000, w_zero)

    w_rand = np.random.uniform(size=x_unif.shape[1])
    w, niters_rand[i] = ajusta_PLA(x_unif, noised_y, 10000, w_rand)



print(f"Para cada ejecución de PLA con w inicial = vector de ceros, tomó {niters_zero} iteraciones en converger\nEn promedio, {np.mean(niters_zero)} iteraciones")
stop()
print(f"Para cada ejecución de PLA con w inicial = vector de valores aleatorios en [0,1], tomó {niters_rand} iteraciones en converger\nEn promedio, {np.mean(niters_rand)} iteraciones")
stop()

# Apartado b) Regresión Logística ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 1)
def sgd_logistic(x, y, lr=0.01, epsilon=0.01):
    # Asignamos inicialmente 0 a los pesos
    w = np.zeros(x.shape[1])
    # Generador de números aleatorios para crear los minibatch
    step = epsilon

    # Índices de cada elemento. Contabilizan los batches (tamaño 1)
    indices = np.arange(y.size)
    np.random.shuffle(indices)
    # Contabilizamos épocas
    while (step >= epsilon):
        previous_w = w
        for i in indices:
            w = w - lr * -y[i]*x[i,:]*sigmoid(-y[i]*np.dot(w, x[i,:]))
        step = np.linalg.norm(previous_w - w)
        previous_w = w
    return w

def graph_regression(x, y, w, x_label="", y_label="", w_label="", title=""):
    # Función para graficar los conjuntos de datos y la frontera de decisión para algún método

    # Graficamos los datos, identificados según su etiqueta
    for i in np.unique(y):
        plt.scatter(x[y == i, 1], x[y == i, 2], label=i)
    
    # Calculamos la pendiente y el punto de intercección
    intercept = -w[0] / w[2]
    slope = -(w[0] / w[2]) / (w[0] / w[1]) # -w[1]/w[2]
    
    # Ajustamos el eje x para fijar la recta a los extremos
    x_min, x_max = plt.gca().get_xlim()
    plt.autoscale(axis="x", tight=True)
    plt.gca().set_ylim(plt.gca().get_ylim())

    # Determinamos los valores de x e y para la función de pesos
    regres_x = np.linspace(x_min, x_max, y.size)
    regres_y = slope*regres_x + intercept

    # La graficamos junto a toda la información identificativa
    plt.plot(regres_x, regres_y, label=w_label)
    plt.legend()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

# Función sigmoide, necesaria para la regresión logística
def sigmoid(z):
    e = np.e
    return 1 / (1 + e**-z)

# Generamos una recta en el plano [0,2]x[0,2]
a,b = simula_recta([0,2])
# Generamos 100 puntos aleatorios en el plano anterior
x = simula_unif(100, 2, [0,2])
# Le añadimos 1 al inicio para el correcto procesamiento
x = np.concatenate((np.ones((x.shape[0],1)), x), axis=1)
# Y los etiquetamos según su posición respecto a la recta
y = np.sign(x[:,2] - a*x[:,1] - b)
# Graficamos los datos etiquetados
graph_points_frontier(x[:,1], x[:,2], y, a, b, title="Nube de datos etiquetados con recta aleatoria en [0,2]x[0,2]")
stop()
# Calculamos un modelo que se ajuste mediante SGD con RL
w = sgd_logistic(x, y)
# Y lo graficamos
graph_regression(x,y,w,"X", "Y", "Regresión logística", "Nube de datos etiquetados junto a regresión logística")
stop()

# 2)
print("Sobre nuestra muestra el error es")
ein = np.sum(np.log(1 + np.e**(-y*np.matmul(x,w)))) / y.size
print(f"Ein = {ein}")
stop()
print("Generamos una nueva muestra de tamaño 2000, y calculamos el error sobre ésta")
x_out = simula_unif(2000, 2, [0,2])
x_out = np.concatenate((np.ones((x_out.shape[0],1)), x_out), axis=1)
y_out= np.sign(x_out[:,2] - a*x_out[:,1] - b)
eout = np.sum(np.log(1 + np.e**(-y_out*np.matmul(x_out, w)))) / y_out.size
print(f"Eout = {eout}")
stop()