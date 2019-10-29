import cv2
import numpy as np


def show_image(window_name, image):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)


orig_image = cv2.imread('segment.jpg', 1)
hsv_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2HSV)
show_image('Orig image', orig_image)
show_image('Hsv image', hsv_image)

approximation = 10

low = np.array([30 - approximation, 100, 100])
high = np.array([30 + approximation, 255, 255])
mask = cv2.inRange(hsv_image, low, high)

hsv_image[mask > 0] = ([0, 0, 0])
hsv_image[mask == 0] = ([255, 255, 255])
hsv_image = cv2.GaussianBlur(hsv_image, (49, 49), 0)
show_image('Hsv with mask', hsv_image)

temp_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
gray_image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)

contours, hierarchy = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(orig_image, (x, y), (x + w, y + h), (0, 0, 255), 3)

show_image('Orig image with boxes', orig_image)

while True:
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
        break
