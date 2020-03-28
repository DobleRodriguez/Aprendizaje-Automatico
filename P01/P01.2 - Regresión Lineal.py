# Aprendizaje Automático - Grupo A2
# Universidad de Granada - Grado en Ingeniería Informática - Curso 2019/2020
# Javier Rodríguez Rodríguez - @doblerodriguez


#------------------------------- Práctica 01 - Parte 02 ---------------------------------#
#---------------------------- EJERCICIO SOBRE REGRESIÓN LINEAL ------------------------------#

import pathlib as pl

import matplotlib.pyplot as plt
import numpy as np

# Función auxiliar para detener la ejecución del script entre cada apartado
def stop():
    input("\nPulse ENTER para continuar\n")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Ejercicio 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def read_data(x_file, y_file):
    # Función de carga de la información de los ficheros mediante NumPy
    x_data = np.load(x_file)
    y_data = np.load(y_file)

    # En X guardamos las instancias etiquetadas con 1 y 5
    x = x_data[(y_data == 1) | (y_data == 5)]
    # Agregamos columna de unos para el término independiente
    x = np.hstack((np.ones((len(x), 1)), x))
    # En y guardamos etiquetas para dichas instancias, -1 si es un 1 y 1 si es un 5
    y = np.select([y_data == 1, y_data==5], [-1, 1])
    y = y[y != 0]
    return x, y

def error(x, y, w):
    # Función de medición del error cuadrático
    return np.linalg.norm(np.matmul(x,w) - y) ** 2 / y.size

def pseudoinverse_method(x, y):
    # Calculamos la pseudoinversa y retornamos los pesos dados por este algoritmo
    x_pinv = np.linalg.pinv(x)
    return np.matmul(x_pinv, y)

def stochastic_gradient_descent(x, y, lr, max_iters, minibatch_size, epsilon):
    # Asignamos inicialmente 0 a los pesos
    w = np.zeros(x.shape[1])
    # Generador de números aleatorios para crear los minibatch

    # Hasta alcanzar el máximo de iteraciones
    for i in np.arange(max_iters):
        # Generamos minibatches
        minibatch_indices = np.random.default_rng().choice(len(y), minibatch_size)
        x_minibatch = x[minibatch_indices]
        y_minibatch = y[minibatch_indices]
        # Calculamos la estimación estocástica del gradiente para el minibatch, y actualizamos pesos
        # según ésta
        w = w - lr * 2/minibatch_size * np.matmul(x_minibatch.T , (np.matmul(x_minibatch, w) - y_minibatch))
        # Si el error está por debajo de la cota mínima, se detiene la ejecución
        if (error(x, y, w) < epsilon):
            break
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

print("Parte 2: EJERCICIO SOBRE REGRESIÓN LINEAL")
print("Ejercicio 1\n")

# Carga de ficheros (usando pathlib para evitar posibles problemas de path)
x_train, y_train = read_data(pl.Path(__file__).parent / f"datos/X_train.npy", 
    pl.Path(__file__).parent / f"datos/y_train.npy")

x_test, y_test = read_data(pl.Path(__file__).parent / f"datos/X_test.npy", 
    pl.Path(__file__).parent / f"datos/y_test.npy")

# Calculamos los pesos con ambos métodos usando los datos de entrenamiento
w_pseudo = pseudoinverse_method(x_train, y_train)
# Para el SGD utilizamos lr = 0.01 y minibatch_size = 64 por ser valores generalmente adecuados
# Épsilon y máx_iters se escogen evitando detener el algoritmo antes de lo deseado  
w_sgd = stochastic_gradient_descent(x_train, y_train, 0.01, 5000, 64, 1e-14)


# Calculamos la bondad según el error para los datos de entrenamiento y test
print("Para el algoritmo de la pseudoinversa: ")
print(f"Bondad usando Ein: {error(x_train, y_train, w_pseudo)}")
print(f"Bondad usando Eout: {error(x_test, y_test, w_pseudo)}")
stop()

# Graficamos los datos de entrenamiento y la frontera de decisión para ambos métodos
print("Gráfica de datos de entrenamiento junto a frontera de decisión para el algoritmo de la pseudoinversa")
graph_regression(x_train, y_train, w_pseudo, "Intensidad promedio", "Simetría",
    "w con pseudoinversa", "Regresión lineal con algoritmo de la pseudoinversa")
stop()

# Ídem a lo realizado con el algoritmo de la pseudoinversa
print("Para el algoritmo Gradiente Descendente Estocástico: ")
print(f"Bondad usando Ein: {error(x_train, y_train, w_sgd)}")
print(f"Bondad usando Eout: {error(x_test, y_test, w_sgd)}")
stop()

print(("Gráfica de datos de entrenamiento junto a frontera de decisión" 
    " para el algoritmo Gradiente Descendente Estocástico"))
graph_regression(x_train, y_train, w_sgd, "Intensidad promedio", "Simetría",
    "w con sgd", "Regresión lineal con gradiente descendente estocástico")
stop()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Ejercicio 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Genera la muestra de puntos 2D contenidos en un cuadrado
def simula_unif(N, dims, size):
    uniform_square = np.random.default_rng().uniform(-size, size, (N, dims))
    return uniform_square

# Función a considerar
def func(w):
    x1 = w[0]
    x2 = w[1]
    return np.sign((x1-0.2)**2 + x2**2 - 0.6)

# Selecciona una muestra correspondiente al porcentaje dado y les cambia el valor
def apply_noise(y, percent):
    noised_indices = np.random.default_rng().choice(y.size, int(np.rint(y.size * percent/100)), 
        replace=False)
    y[noised_indices] *= -1
    return y

# Grafica la frontera de decisión curva
def graph_contour(x, y, w, x_label="", y_label="", w_label="", title=""):
    # Grafica los puntos
    for i in np.unique(y):
        plt.scatter(x[y == i, 1], x[y == i, 2], label=i)
    
    # Delimita el espacio de la gráfica
    x_min, x_max = plt.gca().get_xlim()
    y_min, y_max = plt.gca().get_ylim()
    plt.gca().set_xlim(plt.gca().get_xlim())
    plt.gca().set_ylim(plt.gca().get_ylim())


    # Determinamos los valores de x e y para la función de pesos
    regres_x = np.linspace(x_min, x_max, y.size)
    regres_y = np.linspace(y_min, y_max, y.size)
    X, Y = np.meshgrid(regres_x, regres_y)
    ellipse = w[0] + X*w[1] + Y*w[2] + X*Y*w[3] + w[4]*X**2 + w[5]*Y**2
    # La graficamos junto a toda la información identificativa
    plt.contour(X, Y, ellipse, [0])
    plt.legend()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

# Generamos 1000 puntos 2D en [-1,1]x[-1,1]
print("Ejercicio 2")
x = simula_unif(1000, 2, 1)

# Los graficamos
plt.scatter(x[:,0], x[:,1])
plt.title("Mapa de puntos 2D")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

stop()

# Generamos las etiquetas aplicando la función al vector
y = np.apply_along_axis(func, 1, x)
# Añadimos ruido
y = apply_noise(y, 10)

# Graficamos nuevamente el mapa, ahora etiquetado
for i in np.unique(y):
    plt.scatter(x[y == i,0], x[y == i, 1], label=i)

plt.title("Mapa de etiquetas (con ruido)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()

stop()

print("Utilizando como vector de características (1, x1, x2)")

# Agregamos la columna de 1
x = np.column_stack((np.ones(len(x)), x))
# Hacemos una ejecución de SGD
w = stochastic_gradient_descent(x, y, 0.01, 500, 64, 1e-14)

# Calculo de Ein
print(f"Se obtienen pesos w = {w}")
print(f"El error de ajuste Ein usando Gradiente Descendente Estocástico (SGD) es: {error(x, y, w)}")

# Gráfica resultane
graph_regression(x, y, w, 'x1', 'x2', 'w con SGD', "Regresión lineal con SGD para (1, x1, x2)")

stop()
# d)
# Acumulamos los errores de cada experimento en los vectores errin y errout
errin = np.empty(1000)
errout = np.empty(1000)
print("Realizamos 1000 experimentos para calcular Ein y Eout promedios")
for i in np.arange(1000):
    # Repetimos el experimento 1000 veces
    x = np.column_stack((np.ones(1000), simula_unif(1000, 2, 1)))
    y = apply_noise(np.apply_along_axis(func, 1, x[:,1:]), 10)
    w = stochastic_gradient_descent(x, y, 0.01, 500, 64, 1e-14)
    errin[i] = error(x, y, w)
    # Generamos un nuevo conjunto para usarlo de test
    x_out = np.column_stack((np.ones(1000), simula_unif(1000, 2, 1)))
    y_out = apply_noise(np.apply_along_axis(func, 1, x_out[:,1:]), 10)
    errout[i] = error(x_out, y_out, w)

# Calculamos errores promedio
print(f"El valor de Ein promedio para los 1000 experimentos es: {np.mean(errin)}")
print(f"El valor de Eout promedio para los 1000 experimentos es: {np.mean(errout)}")
stop()

print("Utilizando como vector de características (1, x1, x2, x1x2, x1^2, x2^2)")

# Repetimos el procedimiento para el nuevo vector de características (1,x1,x2,x1x2,x1^2,x2^2)
x = np.column_stack((np.ones(1000), simula_unif(1000, 2, 1)))
# Importante conservar el orden de las columnas al juntarlas.
x = np.column_stack((x, np.array([x[:,1] * x[:,2], x[:,1]**2, x[:,2]**2]).T))
y = np.apply_along_axis(func, 1, x[:,1:])
y = apply_noise(y, 10)

w = stochastic_gradient_descent(x, y, 0.01, 500, 64, 1e-14)

# Graficamos el comportamiento
graph_contour(x, y, w, 'x1', 'x2', 'w con SGD', "Regresión lineal con SGD para (1, x1, x2, x1x2, x1^2, x2^2)")



print(f"Se obtienen pesos w = {w}")
print(f"El error de ajuste Ein usando Gradiente Descendente Estocástico (SGD) es: {error(x, y, w)}")
stop()
print("Realizamos 1000 experimentos para calcular Ein y Eout promedios")
for i in np.arange(1000):
    # De igual forma a anteriormente, repetimos 1000 experimentos iguales.
    x = np.column_stack((np.ones(1000), simula_unif(1000, 2, 1)))
    x = np.column_stack((x, np.array([x[:,1] * x[:,2], x[:,1]**2, x[:,2]**2]).T))
    y = np.apply_along_axis(func, 1, x[:,1:])
    y = apply_noise(y, 10)
    w = stochastic_gradient_descent(x, y, 0.01, 500, 64, 1e-14)
    errin[i] = error(x, y, w)
    x_out = np.column_stack((np.ones(1000), simula_unif(1000, 2, 1)))
    x_out = np.column_stack((x_out, np.array([x_out[:,1] * x_out[:,2], x_out[:,1]**2, x_out[:,2]**2]).T))
    y_out = np.apply_along_axis(func, 1, x_out[:,1:])
    y_out = apply_noise(y_out, 10)
    errout[i] = error(x_out, y_out, w)

print(f"El valor de Ein promedio para los 1000 experimentos es: {np.mean(errin)}")
print(f"El valor de Eout promedio para los 1000 experimentos es: {np.mean(errout)}")

stop()

