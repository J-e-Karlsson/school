def f_score(prec, reca):
    f =(prec * reca) / (prec + reca)
    return f * 2

#print(f_score(89, 111))
