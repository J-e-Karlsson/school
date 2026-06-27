def divisible_13():
    number_of_numbers = 0
    sum_of_numbers = 0
    for i in range(1, 1000001):
        if i % 13 == 0:
            number_of_numbers += 1
            sum_of_numbers += i
    return number_of_numbers, sum_of_numbers

print(divisible_13())
           
