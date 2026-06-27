import os

indir = "swedish"
outdir = "DATA_FOLDER2"
os.makedirs(outdir, exist_ok = True)

for filename in os.listdir(indir):
    newlines = []
    file = os.path.join(indir, filename)
    outfile = os.path.join(outdir, filename)

    with open(file, "r") as f:
        lines = f.readlines()

    for line in lines:
        if line.strip() != "": #Detect empty lines and skip them
            newline = " ".join(line)
            newlines.append(newline)

    with open(outfile, "w") as new_f:
        for l in newlines:
            new_f.write(l)

"""
Plan:
- Copy all Swedish data files to my directory
- Slice test data at 3000 lines using head in CLI

- Open each file in write
- Iterate lines in each file (one word per line)
- Insert " " between each character
    - If line is empty, skip line (for example line 51 in dev.hs)


This code is to be handed in
"""