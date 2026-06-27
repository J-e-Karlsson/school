import os

def get_lines(path):
    with open(path, "r") as f:
        lines = f.readlines()
    return lines

def accuracy(lines1, lines2):
    total = 0
    correct = 0
    for compare, ref in zip(lines1, lines2):
        """
        Strip the lines first to remove \n and get rid of the differences in 
        trailing whitespace output between the models
        """
        compare = compare.strip()
        ref = ref.strip()
        if len(ref) == 1 and ref.isalpha():
            if ref == compare:
                total += 1
                correct += 1
            else:
                total += 1
        else:
            if ref == compare:
                total += 1
                correct += 1
            else:
                total += 1
    fraction = correct/total
    acc = fraction*100
    return f"{correct}/{total} = {acc}%"


def main():
    """NMT paths were added at stage 4"""
    outpath = "out.sv" #SMT
    outpath2 = "translations.char" #NMT
    refpath = "DATA_FOLDER/swedish.hs-sv.test.sv"
    basepath = "DATA_FOLDER/swedish.hs-sv.test.hs"


    outlines = get_lines(outpath)
    outlines2 = get_lines(outpath2)
    reflines = get_lines(refpath)
    baselines = get_lines(basepath)

    outacc = accuracy(outlines, reflines)
    outacc2 = accuracy(outlines2, reflines)
    baseacc = accuracy(baselines, reflines)
    print("SMT translation accuracy: ", outacc)
    print("NMT translation accuracy: ", outacc2)
    print("Baseline: ", baseacc)

main()



"""
Plan:
- open all three files
- function takes two files at once (out.sv and reference first round, then historic file + reference)
- readlines on both files
- iterate both lists as zip(l1,l2)
- if reference files list current index is a special character and nothing else, then skip
- compare all other indices and add +1 to correct if correct
- always do +1 to total

- function is run twice, first getting translation accuracy, then baseline accuracy

This code is to be handed in
"""