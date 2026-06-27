import sys

def test(filepath):
    x = open(filepath, "r")
    y = x.readlines()
    x.close()
    return y

print(repr(test(sys.argv[1])))
