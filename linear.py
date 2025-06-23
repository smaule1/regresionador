import math
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats 


#Input
x = [18,19,19.5,19.7,19.9]
y = [19, 40, 79, 130, 397]
x = np.array(x)
y = np.array(y)
delta = 0.05



#Codigo

def calc_lineal(x, y):

    n = len(x)
    print(f"n = {n}")

    suma_x = sum(x)
    print(f"{suma_x=}")
    suma_y = sum(y)
    print(f"{suma_y=}")

    prom_x = suma_x/n
    print(f"promedio x = {prom_x}")
    prom_y = suma_y/n
    print(f"promedio y = {prom_y}")
    suma_xy = math.sumprod(x,y)
    print(f"suma de producto x*y = {suma_xy}")


    suma_x2 = sum(list(map(lambda x: pow(x, 2), x)))
    print(f"suma x cuadrado = {suma_x2}")

    suma_y2 = sum(list(map(lambda x: pow(x, 2), y)))
    print(f"suma y cuadrado = {suma_y2}")

    b = (1/n * suma_xy - prom_x*prom_y) / (1/n * suma_x2 - pow(prom_x, 2))
    print(f"b = {b}")

    a = prom_y - b*prom_x
    print(f"a = {a}")

    sxx = suma_x2 - pow(suma_x, 2)/n
    print(f"{sxx=}")
    syy = suma_y2 - pow(suma_y, 2)/n
    print(f"{syy=}")
    sxy = suma_xy - suma_x*suma_y/n
    print(f"{sxy=}")
    sce = syy - b*sxy
    print(f"{sce=}")
    s2 = sce/(n-2)
    print(f"{s2=}")
    s = math.sqrt(s2)
    print(f"{s=}")

    #Intervalos

    t = stats.t.ppf(delta/2, n-2)
    print(f"{t=}")

    intervalos_a = (a + t*s*math.sqrt(suma_x2/(n*sxx)), a - t*s*math.sqrt(suma_x2/(n*sxx)))
    print(f"{intervalos_a=}")
    intervalos_b = (b + t*(s*math.sqrt(1/sxx)), b - t*(s*math.sqrt(1/sxx)))
    print(f"{intervalos_b=}")

    #Correlación

    r = b * math.sqrt(sxx/syy)
    print(f"{r=}")
    r2 = pow(r,2)
    print(f"{r2=}")


    return (a, b)



def gen_lineal(x, a, b):
    y = a + b*x
    return y

def lineal():
    res = calc_lineal()
    y_estimado = gen_lineal(x, *res)
    plt.plot(x, y_estimado, lw=2, label=f"y = {res[0]:.2f} + {res[1]:.2f} * x ")

    

def gen_hiperbolica(x, a, b):
    return x / (a*x + b)

def hiperobolica():    
    x1 = 1/x
    y1 = 1/y
    res = calc_lineal(x1, y1)
    x_linea = np.linspace(x[0], x[-1], 100)
    y_estimado = gen_hiperbolica(x_linea,*res)
    plt.plot(x_linea, y_estimado, lw=2, label=f"y = {res[0]:.2f} + {res[1]:.2f} * x ")



#Plotting


def plot():
    plt.scatter(x, y)  
    plt.legend()      
    plt.ticklabel_format(style="plain")
    margen_x = 0.1 * (max(x)- min(x))
    margen_y = 0.1 * (max(y)- min(y))
    plt.xlim(min(x)-margen_x,max(x)+margen_x)
    plt.ylim(min(y)-margen_y,max(y)+margen_y)
    plt.show()





