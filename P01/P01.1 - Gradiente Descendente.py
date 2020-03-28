# Aprendizaje Automático - Grupo A2
# Universidad de Granada - Grado en Ingeniería Informática - Curso 2019/2020
# Javier Rodríguez Rodríguez - @doblerodriguez


#------------------------------- Práctica 01 - Ejercicio 01 ---------------------------------#
#---------------------EJERCICIO SOBRE LA BÚSQUEDA ITERATIVA DE ÓPTIMOS ----------------------#

import matplotlib.pyplot as plt
import numpy as np

# Función auxiliar para detener la ejecución del script entre cada apartado
def stop():
    input("\nPulse ENTER para continuar\n")

# 1) Implementar el algoritmo de gradiente descendente (1 punto)
def gradient_descent(w, lr, grad_func, epsilon, max_iters=10000):

    # Queremos que el paso inicie siendo superior al tamaño mínimo para asegurar una iteración
    step = epsilon+1
    #Contabilizamos las iteraciones ejecutadas
    iters = 0
    # Mientras no se alcance el número máximo de iteraciones y se avance más del paso mínimo
    while((iters < max_iters) & (step > epsilon)):
        # Calculamos los nuevos valores para la función
        next_w = w - lr * grad_func(w)
        # Calculamos cuánto se avanzó
        step = np.linalg.norm(next_w - w)
        # Actualizamos los pesos y la cantidad de iteraciones
        w = next_w
        iters += 1
    # Devolvemos los valores de las coordenadas para los que la función se minimiza
    return w

# 2) Considerar la función E(u,v) = (ue^v - 2ve^u)
def E(w):
    u = w[0]
    v = w[1]
    e = np.e
    return (u * e**v - (2*v * e**-u))**2

# 2)a) Calcular analíticamente y mostrar la expresión del gradiente de la función E(u,v)

# Derivada parcial de E respecto a u
def Eu(w):
    u = w[0]
    v = w[1]
    e = np.e
    return 2 * (u * e**v - 2*v *e**-u) * (e**v + 2*v * e**-u)

# Derivada parcial de E respecto a v
def Ev(w):
    u = w[0]
    v = w[1]
    e = np.e
    return 2 * (u * e**v - 2*v * e**-u) * (u * e**v - 2 * e**-u)

# Gradiente de E (vector con las derivadas parciales respecto a sus coordenadas)
def gradE(w):
    return np.array([Eu(w), Ev(w)])


print("Parte 1: EJERCICIO SOBRE LA BÚSQUEDA ITERATIVA DE ÓPTIMOS")
print("1.2) Para la función E(u,v) = (ue^v - 2ve^u) en el punto (u,v) = (1,1) con tasa de aprendizaje 0.1")

# Tomamos los datos iniciales de interés (u,v) = (1,1) y el lr = 0.1
# Partimos de la realización de 0 iteraciones
min_iters = 0
w = np.array([1,1])
lr = 0.1
# Hasta no alcanzar un valor de E(u,v) < 10^-14, incrementa en 1 el máximo de iteraciones,
# explorando así las soluciones obtenidas para cada número de iteraciones hasta alcanzar el mínimo
# necesario
while(E(w) > 10 ** -14):
    min_iters += 1
    w = gradient_descent(w, lr, gradE, min_iters)

print(f"b) Toma {min_iters} iteraciones alcanzar u,v tales que E(u,v) = {E(w)} < 1e-14")
stop()
print(f"c) Este valor se alcanzó para las coordenadas (u,v) = ({w[0]}, {w[1]})")
stop()

# 3) Considerar ahora la función  f(x,y) = (x-2)^2 + 2(y+2)^2 + 2sin(2pi*x)sin(2pi*y)
def f(w):
    x = w[0]
    y = w[1]
    pi = np.pi
    return (x-2)**2 + 2 * (y+2)**2 + 2 * np.sin(2*pi*x) * np.sin(2*pi*y)

# Derivada parcial de f respecto a x
def fx(w):
    x = w[0]
    y = w[1]
    pi = np.pi
    return 2 * (x-2) + 4*pi * np.cos(2*pi*x) * np.sin(2*pi*y)

# Derivada parcial de f respecto a y
def fy(w):
    x = w[0]
    y = w[1]
    pi = np.pi
    return 4 * (y+2) + 4*pi * np.sin(2*pi*x) * np.cos(2*pi*y)

# Gradiente de f (vector con las derivadas parciales respecto a sus coordenadas)
def gradf(w):
    return np.array([fx(w), fy(w)])

# Función auxiliar para graficar el comportamiento del gradiente descendiente a medida
# que incrementa el número de iteraciones.
# En vez de incrementar el número máximo de iteraciones y repetir constantemente las ejecuciones,
# basta con conservar el valor de los pesos w obtenidos tras una iteración, graficarlos, e iniciar
# otra iteración pero pasando como w los pesos obtenidos en la iteración anterior
def descent_graph(w, lr, grad_func, func, epsilon, max_iters, fmt):
    for niter in np.arange(max_iters):
        w  = gradient_descent(w, lr, gradf, epsilon, 1)
        plt.plot(niter, f(w), fmt)
    plt.xlabel("N° de iteraciones")
    plt.ylabel("f (x,y)")
    plt.title(f"Gradiente Descendente con tasa de aprendizaje = {lr}")
    plt.show()


print("1.3) Para la función f(x,y) = (x-2)^2 + 2(y+2)^2 + 2sin(2pi*x)sin(2pi*y)")
print("a) Minimizar la función con (x,y) = (1,-1), tasa de aprendizaje = 0.01 y máximo de 50 iteraciones:")
# Tomamos los datos de inicio (x,y) = (1,-1), lr = 0.01 y máximo de iteraciones 50.
lr = 0.01
epsilon = 0
w = np.array([1, -1])
max_iters = 50
descent_graph(w, lr, gradf, f, epsilon, max_iters, 'bo')
plt.show()
stop()

print("Y con tasa de aprendizaje = 0.1")
# Y ahora para lr = 0.1
lr = 0.1
descent_graph(w, lr, gradf, f, epsilon, max_iters, 'bo')
plt.show()
stop()

print("b) Para los valores de (x,y) = {(2.1, -2.1), (3, -3), (1.5, 1.5), (1, -1)}, obtener el valor mínimo de f")
# Establecemos un vector con los pares de datos de interés
values = np.array([2.1, -2.1, 3, -3, 1.5, 1.5, 1, -1]).reshape(4,2)
# Así como lr y máximas iteraciones
lr = 0.01
max_iters = 500

# Calculamos, para cada par (x, y), un vector que contenga los valores x, y, f(x, y) para el que se 
# minimiza la función, y los vamos almacenando en un vector general.
minimums = np.empty(0)
for w0 in values:
    w = gradient_descent(w0, lr, gradf, 0, max_iters)
    minimums = np.append(minimums, np.append(w, f(w)))
minimums = minimums.reshape(4,3)


# Imprimimos, de la información almacenada previamente, la tabla de resultados
print(f"{'x0,y0':>20} {'x':>20} {'y':>20} {'f(x, y)':>20}")
for i in np.arange(minimums.shape[0]):
    print(f"{np.array2string(values[i]):>20} {minimums[i,0]:>20} {minimums[i,1]:>20} {minimums[i,2]:>20}")
stop()
