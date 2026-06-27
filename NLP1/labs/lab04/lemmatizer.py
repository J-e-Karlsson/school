import sys

def noun_lemma(word):
    if word.endswith("s"):
        return word[:-1]
    else:
        return word

def verb_lemma(word):
    #if word.endswith("s"):
        #return word[:-1]
    if word in ("has", "had"):
        return "have"
    elif word in ("being","is", "are", "been", "was", "were"):
        return "be"
    elif word.startswith("\'"):
        return "would"
    elif word.endswith("ed"):
        if word[:-2].endswith("t"):
            return word[:-1]
        else:
            return word[:-2]
    elif word.endswith("ing"):
        return word[:-3] + "e"
    else:
        return word

def adj_lemma(word):
    if word.endswith("er"):
        return word[:-2]
    elif word.endswith("est"):
        return word[:-3]
    else:
        return word

def aux_lemma(word):
    if word in ("has", "had"):
        return "have"
    elif word in ("being","is", "are", "been", "was", "were"):
        return "be"
    elif word.startswith("\'"):
        return "would"
    else:
        return word.lower()

for line in sys.stdin:
    if line.strip():
        (word, tag) = line.strip().split("\t")
        lemma = word
        if tag == "PROPN":
            lemma = word
        elif tag == "NOUN":
            lemma = noun_lemma(word.lower())
        elif tag == "VERB":
            lemma = verb_lemma(word.lower())
        elif tag == "ADJ":
            lemma = adj_lemma(word.lower())
        elif tag == "AUX":
            lemma = aux_lemma(word.lower())
        else:
            lemma = word.lower()
        print("{0}\t{1}\t{2}".format(word, tag, lemma))
    else:
        print()

