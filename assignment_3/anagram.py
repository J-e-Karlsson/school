import sys
import re
from collections import defaultdict
def get_lexicon(filepath):
    x = open(filepath, "r")
    lines = x.readlines()
    x.close()
    return lines

def get_sorted(lst):
    return sorted(lst)

    
def get_dict(string_lst):
    my_dict = defaultdict(list)
    for word in string_lst:
        almost_clean = word.strip("\n").lower()
        word_sorted = get_sorted(almost_clean)
        sorted_together = "".join(word_sorted)
        my_dict[sorted_together].append(almost_clean)
    return my_dict

"""
    for word in string_lst:
        chars = []
        almost_clean = word.strip("\n")
        clean_string = re.escape(almost_clean)
        for char in clean_string:
            small = char.lower()
            chars.append(small)

        sorted_chars = get_sorted(chars)
        my_dict[clean_string] = sorted_chars
        """

def get_anagrams(dict_sorted, sys_input):
    ordered = "".join(get_sorted(sys_input))
    if ordered in dict_sorted:
        anagrams = dict_sorted.get(ordered)
        for word in anagrams:
            if sys_input in anagrams:
                anagrams.remove(sys_input)
        result = "\n".join(anagrams)
    else:
        result = None
    return result


"""
    for key in dict_sorted:
        if dict_sorted[key] == dict_sorted[given]:
            anagrams.append(key)
    anagrams.remove(sys_input)
    return anagrams
"""

def main(): #yields 3 different results: gives anagrams, gives none, or gives nothing (if input in dict but has no anagrams)
    inp = sys.argv[1].lower()
    path = "sv-utf8.txt"
    lexicon = get_lexicon(path)
    dictionary = get_dict(lexicon)
    result = get_anagrams(dictionary, inp)
    return result


print(main())

if __name__ == '__main__':
    main()


