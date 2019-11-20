import math

import cv2
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
                current_i, current_j = i, j
                if i < 0:
                    current_i = -i - 1
                if j < 0:
                    current_j = -j - 1
                if i >= self.height:
                    current_i = self.height - (i - self.height) - 1
                if j >= self.width:
                    current_j = self.width - (j - self.width) - 1
                result_value += (self.image[current_i][current_j]
                                 * self.kernel[i + self.border - row_index][j + self.border - column_index])
        return result_value


def show_image(*images):
    for i, image in enumerate(images):
        cv2.namedWindow(str(i), cv2.WINDOW_NORMAL)
        cv2.imshow(str(i), image)
    while True:
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            break


def calculate_radius_by_sigma(sigma):
    return round(sigma * 6) + 1


def main():
    s = 0.66
    original_image = cv2.imread('../tiger.png', 0)

    gauss_filter = GaussFilter(original_image, calculate_radius_by_sigma(s), s)

    show_image(original_image, gauss_filter.filter())
