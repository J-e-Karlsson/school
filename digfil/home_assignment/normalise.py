import re

def matcher(text):
    """Patterns"""
    #see report for explanations
    p1 = r"\b[vV]"
    p2 = r"(?<=d)e\b"
    p3 = r"tt\b"
    p4 = r"yng"
    p5 = r"þ"
    p6 = r"we\b"
    p7 = r"sch"
    p8 = r"oo(?![kdn])"
    p9 = r"yow"
    p10 = r"hew"
    """Apply rules"""
    new_text = text
    new_text = re.sub(p1, "u", new_text)
    new_text = re.sub(p2, "", new_text)
    new_text = re.sub(p3, "t", new_text)
    new_text = re.sub(p4, "ing", new_text)
    new_text = re.sub(p5, "th", new_text)
    new_text = re.sub(p6, "w", new_text)
    new_text = re.sub(p7, "sh", new_text)
    new_text = re.sub(p8, "o", new_text)
    new_text = re.sub(p9, "you", new_text)
    new_text = re.sub(p10, "how", new_text)
    return new_text

def main():
    with open("subset_large.txt", "r") as f1:
        file = f1.read()
    new_file = matcher(file)
    with open("normalised.txt", "w") as f2:
        f2.write(new_file)

main()

