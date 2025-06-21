
import scipy.stats as stats 
import math

#input

# b = 1.841526
# a = 0.208217

n = 6
sum_x = 585

sum_y = 64.5

sum_x2 = 63125
sum_y2 = 724.25

sum_xy = 5875

prom_x = sum_x/n
prom_y = sum_y/n

#delta = 0.5

b = (1/n * sum_xy - prom_x*prom_y) / (1/n * sum_x2 - pow(prom_x, 2))
print(f"b = {b}")

# Resultados

sxx = sum_x2 - pow(sum_x, 2)/n
print(f"{sxx=}")
syy = sum_y2 - pow(sum_y, 2)/n
print(f"{syy=}")
sxy = sum_xy - sum_x*sum_y/n
print(f"{sxy=}")
sce = syy - b*sxy 
print(f"{sce=}")
s2 = sce/(n-2)
print(f"{s2=}")
s = math.sqrt(s2)
print(f"{s=}")

# t = stats.t.ppf(delta/2, n-2)

# print(t)

intervalos_a = a + t*s*math.sqrt(sum_x2/(n*sxx))
print(f"{intervalos_a=}")
intervalos_a = a - t*s*math.sqrt(sum_x2/(n*sxx))
print(f"{intervalos_a=}")
intervalos_b = b + t*(s*math.sqrt(1/sxx))
print(f"{intervalos_b=}")
intervalos_b = b - t*(s*math.sqrt(1/sxx))
print(f"{intervalos_b=}")
# x= 8

# intervalo_y = t*s*math.sqrt(1+1/n+(pow(x-sum_x/n,2)/sxx))
# print(f"{intervalo_y=}")

