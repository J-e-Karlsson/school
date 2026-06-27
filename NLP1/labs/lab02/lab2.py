from nltk import sent_tokenize, word_tokenize
from collections import defaultdict
from math import log
import sys, time

def get_lines(filepath):
    file = open(filepath, "r")
    string = file.read()
    file.close()
    return string

def s_e(sentence):
    sentence = sentence.lower()
    return ["<s>"] + word_tokenize(sentence) + ["<e>"]

def get_unigrams(sentence_lst):
    my_dict = defaultdict(int)
    for sentence in sentence_lst:
        tokens = s_e(sentence)
        for word in tokens:
            my_dict[word] +=1
    return my_dict

def get_bigrams(sentence_lst):
    my_dict = defaultdict(int)
    for sentence in sentence_lst:
        tokens = s_e(sentence)#adds s and e and tokenizes
        for word in range(len(tokens)-1):
            bigram = tokens[word], tokens[word+1]
            my_dict[bigram] += 1
    #print(my_dict)
    return my_dict

def get_surprisal(probability):
    surp = -log(probability, 2)
    return surp
    
def get_bigram_surprisal(uni, bi):          #unigram_freq and bigram_freq
    surp_dict = defaultdict(float)
    for key, frequency in bi.items():
        print(key, frequency)
        first_word = key[0]                 #first word for each bigram
        first_uni = uni.get(first_word)     #same word as above but as unigram
        uni_smooth = first_uni + (len(uni))#smoothing of freq in uni and bi(only works if all words are unique) #for add one smoothing instead, just have +1
        bi_smooth = frequency + 1
        cond_prob = bi_smooth / uni_smooth  #this gives prob.
        print(f"bi smooth:{bi_smooth}")
        print(f"uni smooth:{uni_smooth}")
        print("cond_prob:")
        print(cond_prob)
        surprisal = get_surprisal(cond_prob)
        surp_dict[key] = surprisal
    return surp_dict

def get_perplexity(surp_dict, test_set):#test set is different from the train set
    total_surp = 00.14285714285714285
    word_count = 0
    for sentence in test_set:
        tokens = s_e(sentence)           #first bigram
        first_bi = (tokens[0], tokens[1])
        first_surp = surp_dict.get(first_bi, get_surprisal(1/len(surp_dict)))
        total_surp += first_surp
        word_count += 1

        for word in range(1, len(tokens)-1): #middle bigrams
            word_count += 1
            bigram = (tokens[word], tokens[word+1])
            bigram_surp = surp_dict.get(bigram, get_surprisal(1/len(surp_dict)))
            total_surp += bigram_surp
        
    
    print("total surp:")
    print(total_surp)
    print("word count:")
    print(word_count)
    perplexity = 2 ** (total_surp/word_count)
    return perplexity

def main():
    train = get_lines(sys.argv[1]) #wiki.train_1147.raw
    test = get_lines(sys.argv[2]) #wiki.test.raw
    train_tokens = sent_tokenize(train)          #sentences
    test_tokens = sent_tokenize(test)

    unigram_freq = get_unigrams(train_tokens)    #unigram freq dict
    bigram_freq = get_bigrams(train_tokens)      #bigram freq dict
    bigram_surp_dict = get_bigram_surprisal(unigram_freq, bigram_freq)  #bi_surprisal_dict
    perp_test = get_perplexity(bigram_surp_dict, test_tokens)           #does not work
    print("perplexity:")
    print(perp_test)
    
    #all tests below give the correct output according to instructions
    """#surprisal test
    print(get_surprisal(1))
    print(get_surprisal(0.5))
    print(get_surprisal(0.3))
    print(get_surprisal(0.1))
    """
    """#bigram test
    print(bigram_freq.get(("agreed", "to")))
    print(bigram_freq.get(("into", "aspects")))
    print(bigram_freq.get(("<s>", "while")))
    print(bigram_freq.get((".", "<e>")))
    """ 
    
    """#testing of get_unigrams
    print(unigram_freq.get("around"))
    print(unigram_freq.get("select"))
    print(unigram_freq.get("<s>"))
    print(unigram_freq.get("<e>"))
    """  
    
    """#testing of get_lines, train, and tokens
    print(len(train))
    print(train.split()[1])
    print(train_tokens[181])
    """
    return

main()
    


