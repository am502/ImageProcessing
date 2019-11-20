import cv2

from filters.gauss.gauss_filter import GaussFilter
from filters.gauss.gauss_filter import calculate_radius_by_sigma
from filters.gauss.gauss_filter import show_image

s = 0.66

original_image = cv2.imread('../tiger_color.png')

hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
h_plane, s_plane, v_plane = cv2.split(hsv_image)

gauss_filter = GaussFilter(v_plane, calculate_radius_by_sigma(s), s)
filtered_gray_image = gauss_filter.filter()

filtered_image = cv2.merge([h_plane, s_plane, filtered_gray_image])

show_image(original_image, cv2.cvtColor(filtered_image, cv2.COLOR_HSV2BGR))
