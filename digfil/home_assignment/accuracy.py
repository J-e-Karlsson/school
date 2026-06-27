#This only works if the words are perfectly aligned, which they aren't
def acc(norm, orig, gold):
    norm_score = 0
    norm_total = 0
    orig_score = 0
    orig_total = 0
    
    i_norm = 0 #enumerate did not work once a word written separately was reached (whereas vs where as), causes misalignment
    i_orig = 0
    for word in gold:
        norm_total += 1
        orig_total += 1
        """print("GOLD ", word)
        if i_norm < len(norm):
            print("NORM ", norm[i_norm])
        if i_orig < len(orig):
            print("ORIG ", orig[i_orig])"""
        """NORM"""
        if i_norm < len(norm) and word == norm[i_norm]:
            #print("NORM ", word, i_norm)
            norm_score += 1
            i_norm += 1
        elif i_norm + 1 < len(norm) and word == norm[i_norm] + norm[i_norm+1]: #deals with misaligned words by putting them together and skipping second word in iteration
            norm_score += 1
            i_norm += 2
        else:
            i_norm += 1
        """ORIG"""
        if i_orig < len(orig) and word == orig[i_orig]:
            #print("ORIG ", word, i_orig)
            orig_score += 1
            i_orig += 1
        elif i_orig + 1 < len(orig) and word == orig[i_orig] + orig[i_orig+1]: #deals with misaligned words by putting them together and skipping second word in iteration
            orig_score += 1
            i_orig += 2
        else:
            i_orig += 1
    
    norm_acc = norm_score/norm_total
    orig_acc = orig_score/orig_total

    norm_count = f"{norm_score}/{norm_total}"
    orig_count = f"{orig_score}/{orig_total}"

    return norm_acc, norm_count, orig_acc, orig_count

#This only works if the words are perfectly aligned, which they aren't
def acc(norm, orig, gold):
    norm_score = 0
    orig_score = 0
    norm_total = 0
    orig_total = 0
    
    for i, word in enumerate(gold):
        norm_total += 1
        orig_total += 1
        if i < len(norm) and word == norm[i]:
            norm_score += 1
        if i < len(orig) and word == orig[i]:
            orig_score += 1
    
    norm_acc = norm_score / norm_total
    orig_acc = orig_score / orig_total

    norm_count = f"{norm_score}/{norm_total}"
    orig_count = f"{orig_score}/{orig_total}"

    return norm_acc, norm_count, orig_acc, orig_count

def main():
    with open("normalised.txt", "r") as f1, open("subset_large.txt", "r") as f2, open("subset_large_en.txt", "r") as f3:
        norm = f1.read()
        orig = f2.read()
        gold = f3.read()
    
    norm_split = norm.split()
    orig_split = orig.split()
    gold_split = gold.split()

    norm_acc, norm_count, orig_acc, orig_count = acc(norm_split, orig_split, gold_split)

    print(f"Normalised accuracy: {norm_acc*100}% ({norm_count})")
    print(f"Original accuracy: {orig_acc*100}% ({orig_count})")

main()