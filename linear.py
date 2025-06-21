import math
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats 
from numpy.random import default_rng



#input
x = [0,1,2,3,5,6]
y = [3066, 2949, 2794,2723, 2426, 2309]
delta = 0.1

#codigo

if len(x) != len(y):
    sys.exit

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

# Mas cosas

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

# Intervalos

t = stats.t.ppf(delta/2, n-2)

intervalos_a = (a + t*s*math.sqrt(suma_x2/(n*sxx)), a - t*s*math.sqrt(suma_x2/(n*sxx)))
print(f"{intervalos_a=}")
intervalos_b = (b + t*(s*math.sqrt(1/sxx)), b - t*(s*math.sqrt(1/sxx)))
print(f"{intervalos_b=}")

# Correlación

r = b * math.sqrt(sxx/syy)
print(f"{r=}")
r2 = pow(r,2)
print(f"{r2=}")

#Plotting

xd = np.array(x)
yd = xd*b + a

# y_min = xd*intervalos_b[0] + a
# y_max = xd*intervalos_b[1] + a

str_formula = f"ŷ = {b:.2f}x + {a:.2f}"
print(str_formula)



#ax = plt.axes()
plt.plot(xd, yd)
# plt.plot(xd, y_min)
# plt.plot(xd, y_max)
plt.scatter(x, y, color="#b6ff0d")
plt.legend([str_formula, "holabuenas", "minimo", "maximo"])
plt.show()







