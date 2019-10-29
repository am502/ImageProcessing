import cv2
import matplotlib.pyplot as plt
import numpy as np

orig_image = cv2.imread('image20.png')
gray_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
height, width = gray_image.shape

clip_limit = 80
bins_quantity = 256
pixels_quantity = height * width

n = 8
m = 8

block_height = int(height / n)
block_width = int(width / m)

block_pixels_quantity = block_height * block_width


def get_block(image, i, j):
    block = np.zeros((block_height, block_width), dtype=int)
    row_index = i * block_height
    column_index = j * block_width
    for i in range(row_index, row_index + block_height):
        for j in range(column_index, column_index + block_width):
            block[i - row_index][j - column_index] = image[i][j]
    return block


def calculate_histogram(image, h, w):
    x = np.zeros(bins_quantity, dtype=int)
    for i in range(h):
        for j in range(w):
            x[image[i][j]] += 1
    return x


def clip_histogram(x):
    pixels_to_distribute = 0
    for i in range(len(x)):
        if x[i] > clip_limit:
            pixels_to_distribute += (x[i] - clip_limit)
            x[i] = clip_limit
    part = int(pixels_to_distribute / bins_quantity)
    for i in range(len(x)):
        x[i] += part
    return x


def normalize_histogram(x):
    x_normalized = [i / max(x) for i in x]
    cdf = np.zeros(bins_quantity, dtype=float)
    previous_sum = 0
    for i, e in enumerate(x):
        previous_sum += (e / pixels_quantity)
        cdf[i] = previous_sum
    return cdf, x_normalized


def calculate_cdf(x):
    cdf = np.zeros(bins_quantity, dtype=float)
    previous_sum = 0
    for i, e in enumerate(x):
        previous_sum += (e / block_pixels_quantity)
        cdf[i] = previous_sum
    return cdf


def get_block_indexes_of_pixel(i, j):
    block_row_index = int(i / block_height)
    block_column_index = int(j / block_width)
    return block_row_index, block_column_index


def get_center_indexes_of_block(i, j):
    center_row_index = (i + 1) * block_height - block_height / 2
    center_column_index = (j + 1) * block_width - block_width / 2
    return int(center_row_index), int(center_column_index)


def transform(image, cdf, block_i, block_j, pixel_i, pixel_j):
    return round(cdf[block_i][block_j][image[pixel_i][pixel_j]] * (bins_quantity - 1))


def transform_linear(image, cdf, blocks, pixel_i, pixel_j):
    left = transform(image, cdf, blocks[0][0], blocks[0][1], pixel_i, pixel_j)
    right = transform(image, cdf, blocks[1][0], blocks[1][1], pixel_i, pixel_j)

    l_i, l_j = get_center_indexes_of_block(blocks[0][0], blocks[0][1])
    r_i, r_j = get_center_indexes_of_block(blocks[1][0], blocks[1][1])

    if r_j - l_j == 0:
        return left * (r_i - pixel_i) / (r_i - l_i) + right * (pixel_i - l_i) / (r_i - l_i)
    else:
        return left * (r_j - pixel_j) / (r_j - l_j) + right * (pixel_j - l_j) / (r_j - l_j)


def transform_bilinear(image, cdf, blocks, y, x):
    l_u = transform(image, cdf, blocks[0][0], blocks[0][1], y, x)
    r_u = transform(image, cdf, blocks[1][0], blocks[1][1], y, x)
    l_d = transform(image, cdf, blocks[2][0], blocks[2][1], y, x)
    r_d = transform(image, cdf, blocks[3][0], blocks[3][1], y, x)

    y1, x1 = get_center_indexes_of_block(blocks[0][0], blocks[0][1])
    y2, x2 = get_center_indexes_of_block(blocks[3][0], blocks[3][1])

    i1 = l_u * (x2 - x) / (x2 - x1) + r_u * (x - x1) / (x2 - x1)
    i2 = l_d * (x2 - x) / (x2 - x1) + r_d * (x - x1) / (x2 - x1)

    return i1 * (y2 - y) / (y2 - y1) + i2 * (y - y1) / (y2 - y1)


def main():
    cdf = np.ndarray((n, m, bins_quantity))
    for i in range(n):
        for j in range(m):
            current_block = get_block(gray_image, i, j)
            cdf[i][j] = calculate_cdf(calculate_histogram(current_block, block_height, block_width))

    result = gray_image.copy()
    for i in range(height):
        for j in range(width):
            b_r_index, b_c_index = get_block_indexes_of_pixel(i, j)
            c_r_index, c_c_index = get_center_indexes_of_block(b_r_index, b_c_index)

            if i <= c_r_index and j <= c_c_index:
                if b_r_index == 0 and b_c_index == 0:
                    new_pixel_value = transform(result, cdf, b_r_index, b_c_index, i, j)
                elif b_r_index == 0:
                    blocks = np.array([[b_r_index, b_c_index - 1], [b_r_index, b_c_index]])
                    new_pixel_value = transform_linear(result, cdf, blocks, i, j)
                elif b_c_index == 0:
                    blocks = np.array([[b_r_index - 1, b_c_index], [b_r_index, b_c_index]])
                    new_pixel_value = transform_linear(result, cdf, blocks, i, j)
                else:
                    blocks = np.array([[b_r_index - 1, b_c_index - 1], [b_r_index - 1, b_c_index],
                                       [b_r_index, b_c_index - 1], [b_r_index, b_c_index]])
                    new_pixel_value = transform_bilinear(result, cdf, blocks, i, j)
            elif i <= c_r_index and j >= c_c_index:
                if b_r_index == 0 and b_c_index == m - 1:
                    new_pixel_value = transform(result, cdf, b_r_index, b_c_index, i, j)
                elif b_r_index == 0:
                    blocks = np.array([[b_r_index, b_c_index], [b_r_index, b_c_index + 1]])
                    new_pixel_value = transform_linear(result, cdf, blocks, i, j)
                elif b_c_index == m - 1:
                    blocks = np.array([[b_r_index - 1, b_c_index], [b_r_index, b_c_index]])
                    new_pixel_value = transform_linear(result, cdf, blocks, i, j)
                else:
                    blocks = np.array([[b_r_index - 1, b_c_index], [b_r_index - 1, b_c_index + 1],
                                       [b_r_index, b_c_index], [b_r_index, b_c_index + 1]])
                    new_pixel_value = transform_bilinear(result, cdf, blocks, i, j)
            elif i >= c_r_index and j <= c_c_index:
                if b_r_index == n - 1 and b_c_index == 0:
                    new_pixel_value = transform(result, cdf, b_r_index, b_c_index, i, j)
                elif b_r_index == n - 1:
                    blocks = np.array([[b_r_index, b_c_index - 1], [b_r_index, b_c_index]])
                    new_pixel_value = transform_linear(result, cdf, blocks, i, j)
                elif b_c_index == 0:
                    blocks = np.array([[b_r_index, b_c_index], [b_r_index + 1, b_c_index]])
                    new_pixel_value = transform_linear(result, cdf, blocks, i, j)
                else:
                    blocks = np.array([[b_r_index, b_c_index - 1], [b_r_index, b_c_index],
                                       [b_r_index + 1, b_c_index - 1], [b_r_index + 1, b_c_index]])
                    new_pixel_value = transform_bilinear(result, cdf, blocks, i, j)
            elif i >= c_r_index and j >= c_c_index:
                if b_r_index == n - 1 and b_c_index == m - 1:
                    new_pixel_value = transform(result, cdf, b_r_index, b_c_index, i, j)
                elif b_r_index == n - 1:
                    blocks = np.array([[b_r_index, b_c_index], [b_r_index, b_c_index + 1]])
                    new_pixel_value = transform_linear(result, cdf, blocks, i, j)
                elif b_c_index == m - 1:
                    blocks = np.array([[b_r_index, b_c_index], [b_r_index + 1, b_c_index]])
                    new_pixel_value = transform_linear(result, cdf, blocks, i, j)
                else:
                    blocks = np.array([[b_r_index, b_c_index], [b_r_index, b_c_index + 1],
                                       [b_r_index + 1, b_c_index], [b_r_index + 1, b_c_index + 1]])
                    new_pixel_value = transform_bilinear(result, cdf, blocks, i, j)
            else:
                new_pixel_value = transform(result, cdf, b_r_index, b_c_index, i, j)

            result[i][j] = new_pixel_value

    x = calculate_histogram(result, height, width)
    result_cdf, x_normalized = normalize_histogram(x)
    plt.bar(np.arange(bins_quantity), x_normalized)
    plt.plot(result_cdf, 'r')
    plt.show()

    cv2.namedWindow('Source', cv2.WINDOW_NORMAL)
    cv2.imshow('Source', gray_image)

    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    cv2.imshow('Result', result)

    while True:
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
            break


main()
