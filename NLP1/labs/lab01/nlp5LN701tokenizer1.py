import re, sys

# Add code here as you did in nlp5LN701tokenizer0.py 
def get_lines(filepath):
    file = open(filepath, "r")
    lines = file.readlines()
    file.close()
    return lines

def main():
    lines = get_lines(sys.argv[1])
    pattern = r"""(?x)
                \d+(?:[.,]\d+)* #\$?\d+(?:[.,]\d+)*%?
                | \w+(?:-\w+)+ #words with hyphens
                | (?:[A-Z]\.)+ #abbreviations
                | (?:[A-Z][a-z]{,2}\.)
                | \w+ #words
                | \.\.\. #ellipses
                |[$&%,;:.!?\"] #punctuation
                """
    for line in lines:
        tokens = [token for token in re.findall(pattern, line.strip()) if token.strip()]
        for token in tokens:
            print(token)

if __name__ == "__main__":
    main()

