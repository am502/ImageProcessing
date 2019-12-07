import math

import numpy as np


class Sobel:
    def __init__(self, image):
        self.image = image
        self.height, self.width = image.shape
        self.radius = 3
        self.border = int(self.radius / 2)
        self.m_x, self.m_y = self.masks()

    def masks(self):
        m_y = np.zeros((self.radius, self.radius), dtype=int)
        m_y[0] = [-1, -2, -1]
        m_y[2] = -m_y[0]
        return m_y.T, m_y

    def calculate_gradient(self):
        magnitudes = np.zeros((self.height, self.width), dtype=np.uint8)
        angles = np.zeros((self.height, self.width), dtype=float)
        for i in range(self.height):
            for j in range(self.width):
                g_x, g_y = self.calculate_values(i, j)
                magnitudes[i][j] = math.sqrt(g_x * g_x + g_y * g_y)
                if g_x == 0:
                    if g_y == 0:
                        angle = 0
                    else:
                        angle = 90
                else:
                    angle = math.atan2(g_y, g_x) * 180 / math.pi
                    if 22.5 <= angle < 67.5:
                        angle = 45
                    elif 67.5 <= angle < 112.5:
                        angle = 90
                    elif 112.5 <= angle < 157.5:
                        angle = 135
                    else:
                        angle = 0
                angles[i][j] = angle
        return magnitudes, angles

    def calculate_values(self, row_index, column_index):
        g_x_value, g_y_value = 0, 0
        for i in range(-self.border + row_index, self.border + row_index + 1):
            for j in range(-self.border + column_index, self.border + column_index + 1):
                current_value = self.get_image_value(i, j)
                g_x_value += (current_value * self.m_x[i + self.border - row_index][j + self.border - column_index])
                g_y_value += (current_value * self.m_y[i + self.border - row_index][j + self.border - column_index])
        return g_x_value, g_y_value

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
