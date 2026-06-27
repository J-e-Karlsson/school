import sys

def get_lines(filepath):
    x = open(filepath, "r")
    xtwo = x.readlines()
    x.close()
    return xtwo

def get_clean(string):#called in get_dict
    return string.strip("\n").replace(" ", "")#remove newline and whitespace

def get_dict(lines):
    my_dict = {}
    for pair in lines:
        new_pair = get_clean(pair).split(",")#splits the intended key/value pairs on comma
        my_dict[new_pair[0]] = new_pair[1]#0 = the english word = the key, 1 = the swedish word, the value
    return my_dict

def quiz(my_dict):
    correct = []
    incorrect = []
    for key in my_dict.keys():
        answer = input("What is the word for " + key + " in Swedish? ")
        if answer.lower() == my_dict[key]:
            correct.append([key, answer.lower()])
        else:
            incorrect.append([key, answer.lower()])
    return correct, incorrect

def final():
    ask = input("Do you want to try again? y/n\n")
    if ask == "y":
        return main()#if yes(y), restart function
    else:
        return #if not yes(y), end this function and let main return "Bye"

def main():
    lines = get_lines(sys.argv[1])#csv file
    dictionary = get_dict(lines)
    correct, incorrect = quiz(dictionary)
    print("These are the words you got correct: {}".format(correct))
    print("These are the words you got wrong: {}".format(incorrect))
    final()#calls function asking to try again
    return "Bye"
    


print(main())

