def f5(x, y):
    times = 0
    while x < y:
        x *= 3
        y -= 2
        times += 1 
    return times


def g(the_int):
    my_dict = {}
    for i in range(the_int):
        my_dict[i] = f5(i, i**2) 
    return my_dict

def main():
    call_value = 20
    print(g(call_value))

main()

