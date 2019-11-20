import math

import numpy as np


def gauss(sigma, x, y):
    return math.e ** ((-x * x - y * y) / 2 / sigma / sigma) / 2 / math.pi / sigma / sigma


def calculate_radius_by_sigma(sigma):
    return round(sigma * 6) + 1


def gauss_kernel(sigma, radius):
    kernel = np.zeros((radius, radius), dtype=float)
    border = int(radius / 2)
    for i in range(-border, border + 1):
        for j in range(-border, border + 1):
            kernel[i + border][j + border] = gauss(sigma, i, j)
    return kernel / kernel.sum()


def show_kernel(kernel):
    height, width = kernel.shape
    for i in range(height):
        for j in range(width):
            print(round(kernel[i][j], 5), end=' ')
        print()


array = list(map(float, input().split()))
result = gauss_kernel(array[0], calculate_radius_by_sigma(array[0]))
show_kernel(result)
