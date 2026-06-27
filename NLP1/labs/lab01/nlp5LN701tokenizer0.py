import re, sys

# Define a function that takes as input a filepath and that returns
#the lines of that file. Use the method .readlines()

def get_lines(filepath):
    x = open(filepath, "r")
    y = x.readlines()
    x.close()
    return y

def main():
    # Add code for a function call here. Pass in sys.argv[1] to the function
    lines = get_lines(sys.argv[1])
    for line in lines:
        for token in re.split("\s+", line.strip()):#\s is whitespace
            print(token)

main()
