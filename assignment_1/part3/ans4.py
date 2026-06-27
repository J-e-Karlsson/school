def f4(x):
    swedish_ö = []        
    for word in x: 
            new_word_small = word.lower()
            new_word_ö = new_word_small.replace("o", "ö")
            swedish_ö.append(new_word_ö)
        

    return swedish_ö

print(f4("Vi bor i Sverige och det kan kännas ovant och otrevligt ibland".split()))

