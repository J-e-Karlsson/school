#1
word = "flowers"
print(word[3])
#2
sentence = "The man sells flowers"
print(sentence[17])
#3
words = ["The man sells flowers"]
print(words[0][17])
#4
lst_words = ["The", "man", "sells", "flowers"]
print(lst_words[3])
#5
mat_clause = ["The", "man", "sells", "flowers"]
print(mat_clause[3][3])
#6
oerc = ["The", "woman", ["the", "man", "is", "afraid", "of"], "has", "a", "gun"]
print(oerc[2][3][5])
#7
oerc_embed = ("The", "man", ("the", "woman", ("who", "many", "boys", "like"), "is", "afraid", "of"), "has", "a", "gun")
print(oerc_embed[2][2][0])
#8
oerc_embed = ("The", "man", ("the", "woman", ("who", "many", "boys", "like"), "is", "afraid", "of"), "has", "a", "gun")
print(oerc_embed[2][2][1][0])
#9
fruits = ["apple", "banana", "orange"]
print(fruits)
#10
a_string = "I love programming"
print(len(a_string))
#11
names = ["Alice", "Bob", "Charlie", "David", "Eve"]
print(names[-1])
#12
message = "Hello, world!"
print(message.upper())
#13
scores = [85, 92, 78]
print(max(scores))
#14
grades = ("A", "B", "C", "D")
print(len(grades))
#15
animals = ["dog", "cat", "rabbit", "hamster", "turtle"]
print((animals[:3]))
#16
nums = (5, 10, 15)
print(sum(nums))
