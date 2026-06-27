def f3(x):
    for i in range(1000):
        if i % 7 == 0:
            x.add(i)
    return x
    
print(f3(set()))

