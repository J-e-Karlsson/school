from math import sqrt

def eq1(lst):
    result = 0
    for n in lst:
        y = (((n - (n**3))**2)*(len(lst)-1)) / sqrt(n)
        result += y
    return result
