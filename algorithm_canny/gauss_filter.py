import math

import numpy as np


class GaussFilter:
    def __init__(self, image, radius, sigma):
        self.image = image
        self.height, self.width = image.shape
        self.radius = radius
        self.border = int(radius / 2)
        self.sigma = sigma
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self):
        kernel = np.zeros((self.radius, self.radius), dtype=float)
        for i in range(-self.border, self.border + 1):
            for j in range(-self.border, self.border + 1):
                kernel[i + self.border][j + self.border] = self.gauss_function(i, j)
        return kernel / kernel.sum()

    def gauss_function(self, x, y):
        return math.e ** ((-x * x - y * y) / 2 / self.sigma / self.sigma) / 2 / math.pi / self.sigma / self.sigma

    def filter(self):
        filtered_image = self.image.copy()
        for i in range(self.height):
            for j in range(self.width):
                filtered_image[i][j] = self.calculate_value(i, j)
        return filtered_image

    def calculate_value(self, row_index, column_index):
        result_value = 0
        for i in range(-self.border + row_index, self.border + row_index + 1):
            for j in range(-self.border + column_index, self.border + column_index + 1):
                result_value += (self.get_image_value(i, j)
                                 * self.kernel[i + self.border - row_index][j + self.border - column_index])
        return result_value

    def get_image_value(self, i, j):
        if i < 0:
            i = -i - 1
        if j < 0:
            j = -j - 1
        if i >= self.height:
            i = self.height - (i - self.height) - 1
        if j >= self.width:
            j = self.width - (j - self.width) - 1
        return self.image[i][j]
