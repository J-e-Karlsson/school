def f7(x):
    list_of_lists = []
    for i in range(1,len(myList)):
        y = (myList[i-1:i])
        z = (myList[i:i+1])
        list_of_lists.append(y+z)
            

    return list_of_lists

x = "here is an example sentence"
myList = x.split()
print(f7(myList))

