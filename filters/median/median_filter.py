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
        buffer = np.zeros(self.radius * self.radius, dtype=int)
        for i in range(self.height):
            for j in range(self.width):
                filtered_image[i][j] = self.calculate_value(i, j, buffer)
        return filtered_image

    def calculate_value(self, row_index, column_index, buffer):
        for i in range(-self.border + row_index, self.border + row_index + 1):
            for j in range(-self.border + column_index, self.border + column_index + 1):
                buffer[(i + self.border - row_index) * self.radius
                       + (j + self.border - column_index)] = self.get_image_value(i, j)
        buffer.sort()
        return buffer[int(self.radius * self.radius / 2)]

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
