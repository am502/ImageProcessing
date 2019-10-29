import cv2
import matplotlib.pyplot as plt
import numpy as np


def calculate_histogram(image, bins_count):
    height, width = image.shape
    histogram = np.zeros(bins_count)
    for i in range(height):
        for j in range(width):
            histogram[image[i][j]] += 1
    return histogram


def otsu(image, histogram, bins_count):
    total_pixels_count = image.size

    total_mean = 0
    for i in range(bins_count):
        total_mean += (i * histogram[i])
    total_mean /= total_pixels_count

    max_sigma = -1
    max_thresh = -1

    first_class_pixels_count = 0
    first_class_intensity = 0
    for i in range(bins_count - 1):
        first_class_pixels_count += histogram[i]
        first_class_intensity += (i * histogram[i])
        w0 = first_class_pixels_count / total_pixels_count
        w1 = 1 - w0
        m0 = first_class_intensity / total_pixels_count
        m0 /= w0
        m1 = (total_mean - w0 * m0) / w1
        current_sigma = w0 * w1 * (m0 - m1) * (m0 - m1)
        if current_sigma > max_sigma:
            max_sigma = current_sigma
            max_thresh = i
    return max_thresh


def threshold(image, thresh):
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            if image[i][j] >= thresh:
                image[i][j] = 255
            else:
                image[i][j] = 0
    return image


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
    bins_count = 256

    x_axis = np.arange(bins_count)

    original_image = cv2.imread('original.jpg')

    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    histogram = calculate_histogram(grayscale_image, bins_count)

    plt.bar(x_axis, histogram)
    plt.show()

    thresh = otsu(grayscale_image, histogram, bins_count)

    processed_image = threshold(grayscale_image, thresh)

    show_image(original_image, processed_image)


main()
