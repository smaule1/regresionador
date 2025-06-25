import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import scipy.optimize as op
import pandas as pd
import math

# Constantes
rng = default_rng()
#colors = ['b','g','r','c','y','k','m']
colors = ['y']
current_color = 0

# Funcion para obtener color de forma ciclica (probablemente innecesaria)
def get_color():
    global current_color
    c = colors[current_color]
    current_color = (current_color+1) % len(colors)
    return c


##################################################
# Funciones para calcular funciones matematicas
##################################################

# Genera el resultado (y) de una funcion lineal
# Puede añadir aleatoriedad al resultado para utilizarlo para hacer un analisis de regresion
def gen_lineal(x, a, b, noise=0., n_outliers=0, seed=None):
    rng = default_rng(seed)

    y = a + b*x

    #Agregar variacion
    error = noise * rng.standard_normal(x.size)
    outliers = rng.integers(0, x.size, n_outliers)
    error[outliers] *= 10

    return y + error

#Polinomial
def gen_polinomial(x, b0=0, b1=0, b2=0, b3=0, b4=0, b5=0, b6=0, noise=0., n_outliers=0, seed=None):
    rng = default_rng(seed)

    y = b0 + b1*x + b2*pow(x,2) + b3*pow(x,3) + b4*pow(x,4) + b5*pow(x,5) + b6*pow(x,6)

    #Agregar variacion
    error = noise * rng.standard_normal(x.size)
    outliers = rng.integers(0, x.size, n_outliers)
    error[outliers] *= 10

    return y + error

#Exponencial
def gen_exponencial(x, a, b, noise=0., n_outliers=0, seed=None):
    rng = default_rng(seed)

    y = a * pow(b, x)

    #Agregar variacion
    error = noise * rng.standard_normal(x.size)
    outliers = rng.integers(0, x.size, n_outliers)
    error[outliers] *= 10

    return y + error

#Potencia
def gen_potencia(x, b, m, noise=0., n_outliers=0, seed=None):
    rng = default_rng(seed)

    y = b * pow(x,m)

    #Agregar variacion
    error = noise * rng.standard_normal(x.size)
    outliers = rng.integers(0, x.size, n_outliers)
    error[outliers] *= 10

    return y + error

#Logistica
def gen_log(x, b, m1, m2, noise=0., n_outliers=0, seed=None):
    rng = default_rng(seed)

    y = m1 / (1 + pow(math.e, -m2*(x - b)))

    #Agregar variacion
    error = noise * rng.standard_normal(x.size)
    outliers = rng.integers(0, x.size, n_outliers)
    error[outliers] *= 10

    return y + error


def gen_reciproca(x, a, b, noise=0., n_outliers=0, seed=None):

    
    y = a + b/x

    return y
 

# Variables globales que tienen los datos observados para hacer el analisis de regresion
x_obs = np.array([0,1,2,3])
y_obs = np.array([1,3,5,7])


###########################################################################################################
# Funciones para hacer analisis de regresion a partir de y_obs y x_obs
###########################################################################################################33

#### Lineal

# Genera un array con los residuos que se usan para calcular suma de cuadrados
# variables es un array con las constantes de la funcion lineal, osea a y b.
def linear_err(variables, x, y):
    return variables[0] + variables[1]*x - y

# Calcula una regresion lineal a partir de los datos observados
# Y añade la linea al grafico para que se muestre al ejecutar plot()
# Acepta un rango para hacer el analisis en una zona especifica
def calcular_lineal(start=None, end=None, extend=True):
    x0 = [0,0]
    x_local = x_obs[start:end]
    y_local = y_obs[start:end]
    resultado = least_squares(linear_err, x0, args=(x_local, y_local))
    y_estimado = gen_lineal(x_obs, *resultado.x)    
    c=get_color()
    plt.plot(x_local, y_estimado[start:end], c=c, lw=2, label=f"y = {resultado.x[0]:.2f} + {resultado.x[1]:.2f} * x ")
    if extend:
        plt.plot(x_obs[:start], y_estimado[:start], c=c, linestyle="dashed")
        plt.plot(x_obs[end:], y_estimado[end:], c=c, linestyle="dashed")
    print(resultado.x)


##### Polinomial

def polinomial_err(variables, x, y):
    poli = 0
    for i, b in enumerate(variables):
        poli += b * pow(x,i)                
    return poli - y

def polinomial_string(variables):
    label = "y =  "
    for i, b in enumerate(variables): 
        label += f"{b:.1f}*x^{i} + "
    return label

def calcular_polinomial(n, f=0, start=None, end=None, extend=True):
    x0 = [1]*n 
    x_local = x_obs[start:end]
    y_local = y_obs[start:end]
    if f == 0:
        f = max(y_local)*10
        print(f)
    resultado = least_squares(polinomial_err, x0, args=(x_local, y_local), loss='soft_l1', f_scale=f)
    y_estimado = gen_polinomial(x_obs, *resultado.x)    
    c=get_color()
    plt.plot(x_local, y_estimado[start:end], c=c, lw=2, label=polinomial_string(resultado.x))
    if extend:
        plt.plot(x_obs[:start], y_estimado[:start], c=c, linestyle="dashed")
        plt.plot(x_obs[end:], y_estimado[end:], c=c, linestyle="dashed")
    print(resultado.x)

#Exponencial

def exponencial_err(variables, x, y):                 
    return variables[0] * pow(math.e, variables[1] * x) - y

def exponencial_string(variables):    
    return f"y = {variables[0]:.2f} * e^{variables[1]:.5f}*x"    

def calcular_exponencial(f=0, start=None, end=None, extend=True):
    x0 = [0,0]
    x_local = x_obs[start:end]
    y_local = y_obs[start:end]
    if f == 0:
        f = max(y_local)*10
        print(f)
            
    resultado = least_squares(exponencial_err, x0, args=(x_local, y_local), loss='soft_l1', f_scale=f)    
    y_estimado = gen_exponencial(x_obs, *resultado.x)    
    c=get_color()
    plt.plot(x_local, y_estimado[start:end], c=c, lw=2, label=exponencial_string(resultado.x))
    if extend:
        plt.plot(x_obs[:start], y_estimado[:start], c=c, linestyle="dashed")
        plt.plot(x_obs[end:], y_estimado[end:], c=c, linestyle="dashed")
    print(resultado.x)

#Potencia

def potencia_err(variables, x, y):                 
    return variables[0] * pow(x, variables[1]) - y

def potencia_string(variables):    
    return f"y = {variables[0]:.2f} * x^{variables[1]:.2f}"    

def calcular_potencia(f=0, start=None, end=None, extend=True):
    x0 = [0,0]
    x_local = x_obs[start:end]
    y_local = y_obs[start:end]
    if f == 0:
        f = max(y_local)
        print(f)
            
    resultado = least_squares(potencia_err, x0, args=(x_local, y_local), loss='soft_l1', f_scale=f)
    y_estimado = gen_potencia(x_obs, *resultado.x)    
    c=get_color()
    plt.plot(x_local, y_estimado[start:end], c=c, lw=2, label=potencia_string(resultado.x))
    if extend:
        plt.plot(x_obs[:start], y_estimado[:start], c=c, linestyle="dashed")
        plt.plot(x_obs[end:], y_estimado[end:], c=c, linestyle="dashed")
    print(resultado.x)

#Logistica
#No sirve

def log_err(variables, x, y):                 
    return variables[1] / (1 + pow(math.e, -variables[2]* (x + variables[0]))) - y

def log_string(variables):    
    return f"y = {variables[1]} / (1 + e^(-{variables[2]}*x + {variables[0]}))^ 1/{variables[0]}"    

def calcular_log(f=0, start=None, end=None):
    x0 = [max(x_obs)/2, max(y_obs), 1]
    x_local = x_obs[start:end]
    y_local = y_obs[start:end]
    if f == 0:
        f = max(y_local)*10
        print(f)
        
    bounds = ([0,max(y_obs)-1,0], [max(x_obs),max(y_obs),10])
    resultado = least_squares(log_err, x0, args=(x_local, y_local), loss='soft_l1', f_scale=f, bounds=bounds, verbose=1, ftol=10**-24, gtol=10**-24, xtol=10**-12)
    y_estimado = gen_log(x_obs, *resultado.x)    
    plt.plot(x_obs, y_estimado, c=get_color(), lw=2, label=log_string(resultado.x))
    print(resultado.x)






#Cargar csv 
def load_csv():
    global x_obs, y_obs
    df = pd.read_csv("japan_data.csv")
    x_obs = df["date"].to_numpy()
    y_obs = df["total_cases"].to_numpy()
    x_obs = x_obs.astype(float)
    y_obs = y_obs.astype(float)


# Genera un grafico con los datos de las variables globales y cualquier otro plot que
# se haya incluido antes

def plot():
    plt.scatter(x_obs, y_obs, c='m')  
    plt.legend()          
    #margen_x = 0.1 * (max(x_obs)- min(x_obs))
    #margen_y = 0.1 * (max(y_obs)- min(y_obs))
    #plt.xlim(min(x_obs)-margen_x,max(x_obs)+margen_x)
    #plt.ylim(min(y_obs)-margen_y,max(y_obs)+margen_y)
    plt.show()
