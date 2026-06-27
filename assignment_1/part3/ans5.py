def f5():
    ask = input("Do you want to exit? y/n")
    while ask != "y":
        ask = input("Do you want to exit? y/n")
        if ask == "y":
            break

f5()
