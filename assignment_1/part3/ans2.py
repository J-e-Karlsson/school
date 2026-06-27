def f2(sentence):
    new_lst = []
    for e in range(len(sentence)):
            new_lst.append(sentence[e-1] + " " + sentence[e])
    
    return new_lst

print(f2("colorless green ideas sleep furiously".split()))
