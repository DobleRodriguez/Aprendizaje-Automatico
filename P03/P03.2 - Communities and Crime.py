# Aprendizaje Automático - Grupo A2
# Universidad de Granada - Grado en Ingeniería Informática - Curso 2019/2020
# Javier Rodríguez Rodríguez - @doblerodriguez


#------------------------------- Práctica 03 - Ejercicio 02 ---------------------------------#
#---------------------------------- Communities and Crime -----------------------------------#

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model 
from sklearn import model_selection
from sklearn import impute
from sklearn import metrics
from warnings import filterwarnings

# Evita que advertencias de sklearn se impriman en pantalla
#filterwarnings('ignore')

# Función auxiliar para detener la ejecución del script entre cada apartado
def stop():
    input("\nPulse ENTER para continuar\n")

# Extraemos los datos del fichero de la fuente, transformando los valores faltantes en NaN
data = np.genfromtxt(Path(__file__).parent / f"datos/communities.data", delimiter=',', missing_values='?')

# Separamos el conjunto en Train y Test
train, test = model_selection.train_test_split(data, test_size=400, random_state=42)
# Separamos las variables clasificatorias de la variable de clase en ambos conjuntos
X_train = train[:,:-1]
y_train = train[:,-1]
X_test = test[:,:-1]
y_test = test[:,-1]

# Preprocesamos los datos. En primer lugar, eliminamos las primeras 4 columnas, pues corresponden
# a datos geográficos identificativos que no aportan ninguna información relevante para la regresión

X_train = X_train[:,5:]
X_test = X_test[:,5:]

# Reemplazamos los valores faltantes por la media de los valores existentes
imp = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
X_train = imp.fit_transform(X_train)
X_test = imp.fit_transform(X_test)

correlation = np.corrcoef(X_train, rowvar=False)

# Grafica un mapa de calor con la correlación. Los valores NaN se sustituyen por 0s (la característica es constante)
plt.imshow(correlation)
plt.colorbar()
plt.title("Correlación entre características")
plt.show()
stop()

# Distintos hiper-parámetros a probar
parameters = {"alpha" : [0.1, 1, 10]}

# Pruebo los distintos modelos y técnicas, haciendo CV con los hiperparámetros para cada uno de ellos
# RidgeCV es una implementación de OLS con regularización Ridge (L2) que implementa directamente CV
# SGD es la implementación para regresión de SGD de sklearn, especificada con regularización l2
print("Calculando la precisión de cada técnica y modelo con distintos hiperparámetros")
classifiers = [("OLS (RL)", linear_model.RidgeCV()), 
                ("SGD (RL)", model_selection.GridSearchCV(linear_model.SGDRegressor(random_state=0, penalty='l2'), parameters)),
                ]

# Comprobamos los resultados
for name, clf in classifiers:
    clf = clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    score = metrics.mean_squared_error(y_test, prediction)
    print(f"Error de {name} con los mejores hiper-parámetros: {score}")

stop()
