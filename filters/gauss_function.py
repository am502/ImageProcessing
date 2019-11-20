import math


def gauss(input_data):
    sigma = input_data[0]
    x = input_data[1]
    y = input_data[2]
    return math.e ** ((-x * x - y * y) / 2 / sigma / sigma) / 2 / math.pi / sigma / sigma


array = list(map(int, input().split()))
result = gauss(array)
print(result)
