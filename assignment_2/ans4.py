def f4(x):
    word_count = {}
    for i in x:
        word_count[i] = len(i)
    return word_count

print(f4(["colorless", "green", "ideas", "sleep", "furiously"]))
