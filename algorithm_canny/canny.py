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


def get_optimal_thresholds(image):
    high_threshold = image.max() * 0.3
    return high_threshold / 2, high_threshold


def canny(image, radius, sigma, low_threshold, high_threshold):
    filtered_image = GaussFilter(image, radius, sigma).filter()

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
            if labels[i][j] in strong_labels:
                suppressed_image_copy[i][j] = 255
            else:
                suppressed_image_copy[i][j] = 0

    show_image(image, filtered_image, magnitudes, suppressed_image, suppressed_image_copy)


original_image = cv2.imread('resources/maestro.jpg', 0)
low, high = get_optimal_thresholds(original_image)

canny(original_image, 5, 1.4, low, high)
