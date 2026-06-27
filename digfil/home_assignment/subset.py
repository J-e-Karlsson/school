with open("icamet-samples_en.txt", "r") as f:
    file = f.readlines()

break_point = int(len(file)/10)
print(len(file))
print(break_point)
small_set = file[:break_point] #cut at 10% of lines
large_set = file[break_point:]


with open("subset_small_en.txt", "w") as small, open("subset_large_en.txt", "w") as large:
    for line in small_set:
        small.write(line + "\n")
    for line in large_set:
        large.write(line + "\n")
