import math

import cv2
import numpy as np


class GaussianFilter:
    def __init__(self, image, radius, sigma):
        self.image = image
        self.height, self.width = image.shape
        self.radius = radius
        self.border = int(radius / 2)
        self.sigma = sigma
        self.kernel = self.gaussian_kernel()

    def gaussian_kernel(self):
        kernel = np.zeros((self.radius, self.radius), dtype=float)
        for i in range(-self.border, self.border + 1):
            for j in range(-self.border, self.border + 1):
                kernel[i + self.border][j + self.border] = self.gaussian_function(i, j)
        return kernel / kernel.sum()

    def gaussian_function(self, x, y):
        return 1 / math.sqrt(2 * math.pi) / self.sigma * math.e ** ((-x * x - y * y) / 2 / self.sigma / self.sigma)

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
                    current_i = self.height - (i - self.height) - 2
                if j >= self.width:
                    current_j = self.width - (j - self.width) - 2
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


original_image = cv2.imread('tiger.png')
hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
_, _, v = cv2.split(hsv)
g_filter = GaussianFilter(v, 5, 0.66)
print(g_filter.kernel)
print(g_filter.kernel.sum())
ii = g_filter.filter()
print(ii.shape)
show_image(ii, cv2.imread('pt.png'), original_image)
