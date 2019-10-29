import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_summed_area_table(image):
    height, width = image.shape
    s = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            s_x = 0
            s_x_y = 0
            s_y = 0
            if i - 1 >= 0:
                s_x = s[i - 1][j]
            if j - 1 >= 0:
                s_y = s[i][j - 1]
            if i - 1 >= 0 and j - 1 >= 0:
                s_x_y = s[i - 1][j - 1]
            s[i][j] = image[i][j] + s_x + s_y - s_x_y
    return s


def main():
    r = 128
    k = 0.4
    w = 20

    bins_count = 256

    original_image = cv2.imread('text.png')

    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)




main()
