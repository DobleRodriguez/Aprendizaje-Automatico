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

def newton_method(w, lr, grad_func, hess_func, max_iters=10000, epsilon=1e-14):
    # Queremos que el paso inicie siendo superior al tamaño mínimo para asegurar una iteración
    step = epsilon+1
    #Contabilizamos las iteraciones ejecutadas
    iters = 0
    # Mientras no se alcance el número máximo de iteraciones y se avance más del paso mínimo
    while((iters < max_iters) & (step > epsilon)):
        # Calculamos los nuevos valores para la función
        next_w = w - np.matmul(np.linalg.inv(hessf(w), )
        # Calculamos cuánto se avanzó
        step = np.linalg.norm(next_w - w)
        # Actualizamos los pesos y la cantidad de iteraciones
        w = next_w
        iters += 1
    # Devolvemos los valores de las coordenadas para los que la función se minimiza
    return w

def f(w):
    x = w[0]
    y = w[1]
    pi = np.pi
    return (x-2)**2 + 2 * (y+2)**2 + 2 * np.sin(2*pi*x) * np.sin(2*pi*y)

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

def fx2(w):
    x = w[0]
    y = w[1]
    pi = np.pi
    return (-8 * pi**2 * np.sin(2*pi*x) * np.sin(2*pi*y)) + 2

def fxy(w):
    x = w[0]
    y = w[1]
    pi = np.pi
    return 8 * pi**2 * np.cos(2*pi*x) * np.cos(2*pi*y)

def fy2(w):
    x = w[0]
    y = w[1]
    pi = np.pi
    return (-8 * pi**2 * np.sin(2*pi*x) * np.sin(2*pi*y)) + 4

# Gradiente de f (vector con las derivadas parciales respecto a sus coordenadas)
def gradf(w):
    return np.array([fx(w), fy(w)])

def hessf(w):
    return np.array([[fx2(w), fxy(w)], [fxy(w), fy2(w)]])

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
