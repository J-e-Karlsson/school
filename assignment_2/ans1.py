def is_consonant(bokstav):
    strng_small = bokstav.lower()
    consonants = "bcdfgjklmnpqstvxzhrwy"
    if strng_small in consonants:
        return True
    


def get_count(strng):
    c_count = 0
    for char in strng:
        if is_consonant(char) == True:
            c_count += 1

    return c_count



def main():
    text = "test"
    consonant_count = get_count(text)
    print(consonant_count)


main()

