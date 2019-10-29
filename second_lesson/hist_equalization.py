import cv2
import matplotlib.pyplot as plt
import numpy as np

orig_image = cv2.imread('landscape.png')
gray_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
height, width = gray_image.shape

bins_quantity = 256
pixels_quantity = height * width

y = np.arange(bins_quantity)


def process_image(image):
    x = np.zeros(bins_quantity)
    for i in range(height):
        for j in range(width):
            x[image[i][j]] += 1
    return x


def normalize_histogram(x):
    x_normalized = [i / max(x) for i in x]
    cdf = np.zeros(bins_quantity, dtype=float)
    previous_sum = 0
    for i, e in enumerate(x):
        previous_sum += (e / pixels_quantity)
        cdf[i] = previous_sum
    return cdf, x_normalized


def main():
    x = process_image(gray_image)

    plt.bar(y, x)
    plt.show()

    cdf, x_normalized = normalize_histogram(x)

    plt.bar(y, x_normalized)
    plt.plot(cdf, 'r')
    plt.show()

    equalized_image = gray_image.copy()
    for i in range(height):
        for j in range(width):
            equalized_image[i][j] = round(cdf[gray_image[i][j]] * (bins_quantity - 1))

    x_equalized = process_image(equalized_image)
    cdf_e, x_e_normalized = normalize_histogram(x_equalized)

    plt.bar(y, x_e_normalized)
    plt.plot(cdf_e, 'r')
    plt.show()

    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Image', equalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()
