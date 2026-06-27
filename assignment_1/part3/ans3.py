def f3(x):
    new_lst = []
    for e in x:
        if type(e) == int and e % 2 == 0:
            new_lst.append(e)
    return new_lst

print(f3(["words", 6598, "I", 6336, "put", 7545, "I", 8839, "that", 5379, "together", 1114, "that", 6204, "together", 6070, "words", 8564, "are", 8036]))
