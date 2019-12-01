import cv2
import numpy as np


class MedianFilter:
    def __init__(self, image, radius):
        self.image = image
        self.height, self.width = image.shape
        self.radius = radius
        self.border = int(radius / 2)
        self.bins_count = 256

    def filter(self):
        filtered_image = self.image.copy()
        base_histogram = self.calculate_histogram(0, 0)
        current_histogram = base_histogram.copy()
        for i in range(self.height):
            for j in range(self.width):
                filtered_image[i][j] = self.calculate_median_by_histogram(current_histogram)
                self.shift_histogram_right(current_histogram, i, j)
            self.shift_histogram_down(base_histogram, i, 0)
            current_histogram = base_histogram.copy()
        return filtered_image

    def calculate_histogram(self, row_index, column_index):
        histogram = np.zeros(self.bins_count, dtype=int)
        for i in range(-self.border + row_index, self.border + row_index + 1):
            for j in range(-self.border + column_index, self.border + column_index + 1):
                histogram[self.get_image_value(i, j)] += 1
        return histogram

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

    def calculate_median_by_histogram(self, histogram):
        median_border = self.radius * self.radius / 2
        count = 0
        for i in range(self.bins_count):
            count += histogram[i]
            if count > median_border:
                return i
        return -1

    def shift_histogram_right(self, histogram, center_i, center_j):
        old_column = center_j - self.border
        new_column = center_j + self.border + 1
        for i in range(-self.border + center_i, self.border + center_i + 1):
            histogram[self.get_image_value(i, old_column)] -= 1
            histogram[self.get_image_value(i, new_column)] += 1
        return histogram

    def shift_histogram_down(self, histogram, center_i, center_j):
        old_row = center_i - self.border
        new_row = center_i + self.border + 1
        for j in range(-self.border + center_j, self.border + center_j + 1):
            histogram[self.get_image_value(old_row, j)] -= 1
            histogram[self.get_image_value(new_row, j)] += 1
        return histogram


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
    original_image = cv2.imread('../salt.jpeg', 0)

    median_filter = MedianFilter(original_image, 5)

    show_image(original_image, median_filter.filter())


main()
