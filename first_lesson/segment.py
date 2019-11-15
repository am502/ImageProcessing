import cv2
import numpy as np


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
    orig_image = cv2.imread('segment.jpg')
    hsv_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2HSV)

    approximation = 10

    low = np.array([30 - approximation, 100, 100])
    high = np.array([30 + approximation, 255, 255])

    hsv_mask = 255 - cv2.inRange(hsv_image, low, high)

    min_square = 4000

    processed_img = orig_image.copy()

    contours, _ = cv2.findContours(hsv_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h * w > min_square:
            cv2.rectangle(processed_img, (x, y), (x + w, y + h), (0, 0, 255), 3)

    show_image(orig_image, hsv_image, hsv_mask, processed_img)


main()
