from math import sqrt
from collections import defaultdict


def get_lines(textpath):
    x = open(textpath, "r")
    xtwo = x.readlines()
    x.close()
    return xtwo



def get_lexicon(svtextpath):
    x = open(svtextpath, "r", encoding = "utf-8")
    x_read = x.read()
    x_listed = x_read.split()
    x.close()
    return x_listed



def is_vowel(character):
    vowels = "aeiouåäöAEIOUYÅÄÖ"
    if character in vowels:
        return True
    else:
        return False


def get_vowel_count(word):
    v_count = 0
    for char in word:
        if is_vowel(char) == True:
            v_count += 1
    return v_count


def get_word_vowels(lst_of_words):
    my_dict = {}
    for word in lst_of_words:
        my_dict[word] = get_vowel_count(word)    
    return my_dict

def get_token_vowels(lst_string, dict_words_vowels):
    my_list = []
    split_lst_string = []
    for i in lst_string:
        l = i.lower()
        z = l.split()
        split_lst_string.extend(z)

    for word in dict_words_vowels:
        small = word.lower()
        if small in split_lst_string:
            x = get_word_vowels(split_lst_string).get(small)
            my_list.append(x)
    return my_list


def get_mean(integers):
    mean = sum(integers)/len(integers)
    return mean


def get_stdev(integers_again):
    stdev_almost = 0
    for i in range (len(integers_again)):
        x = ((integers_again[i] - (get_mean(integers_again))) ** 2)
        stdev_almost += x
    stdev = sqrt(stdev_almost / (len(integers_again)-1))
    return stdev


def main():
    x = get_mean(get_token_vowels(get_lines("swe-sentences.txt"), (get_lexicon("sv-utf8.txt"))))
    y = get_stdev(get_token_vowels(get_lines("swe-sentences.txt"), (get_lexicon("sv-utf8.txt")))) 
    return x, y 


print(main())

