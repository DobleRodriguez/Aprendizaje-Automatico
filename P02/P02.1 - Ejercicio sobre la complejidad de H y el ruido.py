# Aprendizaje Automático - Grupo A2
# Universidad de Granada - Grado en Ingeniería Informática - Curso 2019/2020
# Javier Rodríguez Rodríguez - @doblerodriguez


#------------------------------- Práctica 02 - Ejercicio 01 ---------------------------------#
#---------------------EJERCICIO SOBRE LA COMPLEJIDAD DE H Y EL RUIDO ------------------------#

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

# Funciones preprogramadas de generación

# Fijamos la semilla
np.random.seed(1)

def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna se usará una N(0,sqrt(5)) y para la segunda N(0,sqrt(7))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
        
    return out

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

# Función auxiliar para detener la ejecución del script entre cada apartado
def stop():
    input("\nPulse ENTER para continuar\n")

# Apartado 1. Dibujar una gráfica con la nube de puntos de salida correspondiente.
#a)
# Generamos el vector de pares (x,y) a través de simula_unif según los parámetros especificados
nube_a = simula_unif(50, 2, [-50,50])
# Los graficamos en forma de nube de puntos
plt.scatter(nube_a[:,0], nube_a[:,1])
plt.title("Nube de puntos con simula_unif")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
stop()

#b)
# Repetimos el procedimiento anterior, pero generando los pares a través de simula_gaus
nube_b = simula_gaus(50, 2, [5,7])
plt.scatter(nube_b[:,0], nube_b[:,1])
plt.title("Nube de puntos con simula_gaus")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
stop()

# Apartado 2. Valorar la influencia del ruido en la selección de la complejidad de la clase de
# funciones. 
np.random.seed(1)

def graph_points_frontier(x,y,label,a,b,title):
    for i in np.unique(label):
        plt.scatter(x[label == i], y[label == i], label=i)
    # Graficamos la recta, donde x coincide con los valores de x en el intervalo, e y está determinado
    # por y = ax + b
    plt.gca().set_ylim(plt.gca().get_ylim())
    xlist = np.linspace(-50, 50, 1000)
    plt.plot(xlist, a*xlist + b, label="Recta de separación", c='green')
    # Ajustamos la gráfica para graficar correctamente la línea
    plt.autoscale(axis="x", tight=True)
    # Etiquetamos los ejes y la leyenda
    plt.legend()
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


# Generamos los puntos a través de simula_unif con los parámetros especificados
x_unif = simula_unif(100, 2, [-50,50])
# Simulamos la recta a través de simula_recta en el mismo intervalo
a,b = simula_recta([-50,50])
# Determinamos las etiquetas a partir del valor de la función signo de la distancia entre los puntos
# y la recta, f(x,y) = y - ax - b
y = np.sign(x_unif[:,1] - a*x_unif[:,0] - b)

# a)
graph_points_frontier(x_unif[:,0], x_unif[:,1], y, a, b, "Nube de puntos etiquetados, con línea de frontera")
stop()

# b)
# Hacemos una copia del vector de etiquetas
noised_y = np.copy(y)

# Por cada uno de los valores de etiqueta
for i in np.unique(y):
    # Seleccionamos el 10% de las etiquetas con un valor
    noised_indices = np.random.choice(np.flatnonzero(y == i), int(np.rint(y[y == i].size * 0.10)), replace=False)
    # Las cambiamos en el vector copia
    noised_y[noised_indices] = -y[noised_indices]

# Graficamos la nueva nube de puntos con el ruido aplicado
graph_points_frontier(x_unif[:,0], x_unif[:,1], noised_y, a, b, "Nube de puntos etiquetados con ruido, con línea de frontera")
stop()

# c)
# Definimos las nuevas funciones de clasificación
def f1(x,y):
    return (x - 10)**2 + (y - 20)**2 - 400

def f2(x,y):
    return 0.5*(x + 10)**2 + (y - 20)**2 - 400

def f3(x,y):
    return 0.5*(x - 10)**2 - (y + 20)**2 - 400

def f4(x,y):
    return y - 20*x**2 - 5*x + 3

# Definimos una función auxiliar para generar las gráficas con las nuevas fronteras (no lineales)
# de clasificación
def graph_contour(coords, labels, function, title=""):
    # Graficamos la nube de puntos etiquetados
    for i in np.unique(labels):
        plt.scatter(coords[labels == i, 0], coords[labels == i, 1], label=i)
    # Definimos 1000 puntos equidistantes dentro del cuadrado -50, 50 en el que se contienen nuestros puntos
    xlist = np.linspace(-50, 50, 1000)
    ylist = np.copy(xlist)
    x, y = np.meshgrid(xlist, ylist)
    # Calculamos la función en los 1000x1000 puntos definidos
    z = function(x,y)
    # Graficamos la frontera a partir de los resultados obtenidos
    CS = plt.contour(x,y,z,[0])
    # Y la etiquetamos 
    CS.collections[0].set_label("Frontera de la función")
    # Ajustamos la visualización de la gráfica
    plt.autoscale(axis='x', tight=True)
    plt.title(title)
    plt.legend()
    plt.show()

# Calculamos el nivel de precisión de la recta comparado con los datos etiquetados con ruido
# Como aplicamos ruido a 10% de la muestra, el nivel de precisión será del 90%
accuracy_f0 = sklearn.metrics.accuracy_score(noised_y, y)
print(f"La recta original clasifica los datos con ruido con {accuracy_f0*100}% de precisión")
stop()

# Para cada una de las nuevas funciones, aplicamos el mismo cálculo, para ver si para alguna función
# los resultados mejoran
graph_contour(x_unif, noised_y, f1, "Frontera de clasificación f(x,y) = (x-10)^2 + (y-20)^2 - 400")
stop()
accuracy_f1 = sklearn.metrics.accuracy_score(noised_y, np.sign(f1(x_unif[:,0], x_unif[:,1])))
print(f"La función f(x,y) = (x-10)^2 + (y-20)^2 - 400 clasifica los datos con {accuracy_f1*100}% de precisión")
stop()
graph_contour(x_unif, noised_y, f2, "Frontera de clasificación f(x,y) = 0,5(x+10)^2 + (y-20)^2 - 400")
stop()
accuracy_f2 = sklearn.metrics.accuracy_score(noised_y, np.sign(f2(x_unif[:,0], x_unif[:,1])))
print(f"La función f(x,y) = 0,5(x+10)^2 + (y-20)^2 - 400 clasifica los datos con {accuracy_f2*100}% de precisión")
stop()
graph_contour(x_unif, noised_y, f3, "Frontera de clasificación f(x,y) = 0,5(x-10)^2 - (y+20)^2 - 400")
stop()
accuracy_f3 = sklearn.metrics.accuracy_score(noised_y, np.sign(f3(x_unif[:,0], x_unif[:,1])))
print(f"La función f(x,y) = 0,5(x-10)^2 - (y+20)^2 - 400 clasifica los datos con {accuracy_f3*100}% de precisión")
stop()
graph_contour(x_unif, noised_y, f4, "Frontera de clasificación f(x,y) = y - 20x^2 - 5x + 3")
stop()
accuracy_f4 = sklearn.metrics.accuracy_score(noised_y, np.sign(f4(x_unif[:,0], x_unif[:,1])))
print(f"La función f(x,y) = y - 20x^2 - 5x + 3 clasifica los datos con {accuracy_f4*100}% de precisión")
stop()
