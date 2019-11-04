import math

import cv2
import numpy as np


def calculate_s(image):
    height, width = image.shape
    s1 = np.zeros((height, width))
    s2 = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            s1_x, s2_x = 0, 0
            s1_y, s2_y = 0, 0
            s1_x_y, s2_x_y = 0, 0
            if i - 1 >= 0:
                s1_x = s1[i - 1][j]
                s2_x = s2[i - 1][j]
            if j - 1 >= 0:
                s1_y = s1[i][j - 1]
                s2_y = s2[i][j - 1]
            if i - 1 >= 0 and j - 1 >= 0:
                s1_x_y = s1[i - 1][j - 1]
                s2_x_y = s2[i - 1][j - 1]
            current_pixel = int(image[i][j])
            s1[i][j] = current_pixel + s1_x + s1_y - s1_x_y
            s2[i][j] = current_pixel * current_pixel + s2_x + s2_y - s2_x_y
    return s1, s2


def calculate_sum(s, i1, j1, i2, j2):
    s_i = 0
    s_j = 0
    s_i_j = 0
    if i1 - 1 >= 0:
        s_i = s[i1 - 1][j2]
    if j1 - 1 >= 0:
        s_j = s[i2][j1 - 1]
    if i1 - 1 >= 0 and j1 - 1 >= 0:
        s_i_j = s[i1 - 1][j1 - 1]
    return s_i_j + s[i2][j2] - s_i - s_j


def threshold(image, thresh, i1, j1, i2, j2):
    for i in range(i1, i2):
        for j in range(j1, j2):
            if image[i][j] >= thresh:
                image[i][j] = 255
            else:
                image[i][j] = 0


def show_image(*images):
    for i, image in enumerate(images):
        cv2.namedWindow(str(i), cv2.WINDOW_NORMAL)
        cv2.imshow(str(i), image)
    while True:
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            break


def main():
    r = 128
    k = 0.2
    w = 25

    original_image = cv2.imread('text.png')

    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    height, width = grayscale_image.shape

    s1, s2 = calculate_s(grayscale_image)

    for i in range(0, height, w):
        for j in range(0, width, w):
            n, m = w, w

            if i + w > height:
                i2 = height
                n = height - i
            else:
                i2 = i + w
            if j + w > width:
                j2 = width
                m = width - j
            else:
                j2 = j + w

            mean = calculate_sum(s1, i, j, i2 - 1, j2 - 1) / n / m

            variance = calculate_sum(s2, i, j, i2 - 1, j2 - 1) / n / m - mean * mean

            t = mean * (1 + k * (math.sqrt(variance) / r - 1))

            threshold(grayscale_image, t, i, j, i2, j2)

    show_image(grayscale_image)


main()
