import spacy

def tag(text, nlp):
    doc = nlp(text)
    return [token.pos_ for token in doc]

def compare(norm, orig):
    changed_tags = 0
    for tagA, tagB in zip(norm,orig):
        if tagA != tagB:
            changed_tags += 1
    return changed_tags


def main():
    nlp = spacy.load("en_core_web_sm")
    with open("normalised.txt", "r") as f1, open("subset_large.txt", "r") as f2:
        norm = f1.read()
        orig = f2.read()

    norm_tags = tag(norm, nlp)
    orig_tags = tag(orig, nlp)
    print(len(norm_tags))
    print(len(orig_tags))
    total = min(len(norm_tags), len(orig_tags)) #should be the same but to be safe
    changed_tags = compare(norm_tags, orig_tags)

    print(f"Changed tags: {changed_tags}/{total} ({changed_tags/total*100}%)")

main()