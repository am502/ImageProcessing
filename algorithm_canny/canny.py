import cv2

from algorithm_canny.gauss_filter import GaussFilter
from algorithm_canny.nms import NonMaximumSuppression
from algorithm_canny.sobel import Sobel
from algorithm_canny.tpl import TwoPassLabeling


def show_image(*images):
    for i, image in enumerate(images):
        cv2.namedWindow(str(i), cv2.WINDOW_NORMAL)
        cv2.imshow(str(i), image)
    while True:
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            break


def canny(radius, sigma, high_threshold, low_threshold):
    original_image = cv2.imread('resources/tiger.png', 0)

    filtered_image = GaussFilter(original_image, radius, sigma).filter()

    magnitudes, angles = Sobel(filtered_image).calculate_gradient()

    suppressed_image = NonMaximumSuppression(magnitudes, angles).suppress()

    suppressed_image_copy = suppressed_image.copy()

    height, width = suppressed_image_copy.shape
    for i in range(height):
        for j in range(width):
            if suppressed_image_copy[i][j] < low_threshold:
                suppressed_image_copy[i][j] = 0

    labels = TwoPassLabeling(suppressed_image_copy).label()

    strong_labels = set()
    for i in range(height):
        for j in range(width):
            if suppressed_image_copy[i][j] > high_threshold:
                strong_labels.add(labels[i][j])

    for i in range(height):
        for j in range(width):
            if labels[i][j] not in strong_labels:
                suppressed_image_copy[i][j] = 0

    show_image(original_image, filtered_image, magnitudes, suppressed_image, suppressed_image_copy)


canny(5, 1.4, 10, 100)
