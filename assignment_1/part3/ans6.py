def f6(x):
    listan = []
    for i in range(1,len(x)+1):
        listan.append(x[:i])
    return listan

words = "these are some words that we use"
print(f6(words.split()))
