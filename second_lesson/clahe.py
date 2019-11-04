import cv2
import numpy as np


def up_round(number):
    rounded = int(number)
    if rounded == number:
        return rounded
    else:
        return rounded + 1


def get_block_with_size_by_indexes(matrix, block_row_index, block_column_index, block_height, block_width):
    height, width = matrix.shape
    start_row_index = block_row_index * block_height
    start_column_index = block_column_index * block_width
    end_row_index = start_row_index + block_height
    end_column_index = start_column_index + block_width
    if end_row_index > height:
        end_row_index = height
    if end_column_index > width:
        end_column_index = width
    block = np.zeros((block_height, block_width), dtype=int)
    for i in range(start_row_index, end_row_index):
        for j in range(start_column_index, end_column_index):
            block[i - start_row_index][j - start_column_index] = matrix[i][j]
    return block, end_row_index - start_row_index, end_column_index - start_column_index


def calculate_histogram(matrix, bins_count):
    height, width = matrix.shape
    histogram = np.zeros(bins_count, dtype=int)
    for i in range(height):
        for j in range(width):
            histogram[matrix[i][j]] += 1
    return histogram


def normalize_histogram(histogram, pixels_count):
    normalized_histogram = np.zeros(len(histogram), dtype=float)
    for i, e in enumerate(histogram):
        normalized_histogram[i] = e / pixels_count
    return normalized_histogram


def clip_histogram(normalized_histogram, clip_limit):
    redistribution = 0
    for i, e in enumerate(normalized_histogram):
        if e > clip_limit:
            redistribution += (e - clip_limit)
            normalized_histogram[i] = clip_limit
    part = redistribution / len(normalized_histogram)
    for i, e in enumerate(normalized_histogram):
        normalized_histogram[i] += part
    return normalized_histogram


def calculate_cdf(normalized_histogram):
    cdf = np.zeros(len(normalized_histogram), dtype=float)
    previous_sum = 0
    for i, e in enumerate(normalized_histogram):
        previous_sum += e
        cdf[i] = previous_sum
    return cdf


def get_block_indexes_by_pixel_indexes(pixel_row_index, pixel_column_index, block_height, block_width):
    block_row_index = int(pixel_row_index / block_height)
    block_column_index = int(pixel_column_index / block_width)
    return block_row_index, block_column_index


def transform(matrix, cdfs, block, pixel_i, pixel_j, bins_count):
    return round(cdfs[block[0]][block[1]][matrix[pixel_i][pixel_j]] * (bins_count - 1))


def transform_linear(image, cdfs, centers, blocks, pixel_i, pixel_j, bins_count):
    first = transform(image, cdfs, blocks[0], pixel_i, pixel_j, bins_count)
    second = transform(image, cdfs, blocks[1], pixel_i, pixel_j, bins_count)

    f_c_i, f_c_j = centers[blocks[0][0]][blocks[0][1]]
    s_c_i, s_c_j = centers[blocks[1][0]][blocks[1][1]]

    if s_c_j - f_c_j == 0:
        return first * (s_c_i - pixel_i) / (s_c_i - f_c_i) + second * (pixel_i - f_c_i) / (s_c_i - f_c_i)
    else:
        return first * (s_c_j - pixel_j) / (s_c_j - f_c_j) + second * (pixel_j - f_c_j) / (s_c_j - f_c_j)


def transform_bilinear(image, cdfs, centers, blocks, pixel_i, pixel_j, bins_count):
    x = pixel_j
    y = pixel_i

    l_u = transform(image, cdfs, blocks[0], pixel_i, pixel_j, bins_count)
    r_u = transform(image, cdfs, blocks[1], pixel_i, pixel_j, bins_count)
    l_d = transform(image, cdfs, blocks[2], pixel_i, pixel_j, bins_count)
    r_d = transform(image, cdfs, blocks[3], pixel_i, pixel_j, bins_count)

    y1, x1 = centers[blocks[0][0]][blocks[0][1]]
    y2, x2 = centers[blocks[3][0]][blocks[3][1]]

    i1 = l_u * (x2 - x) / (x2 - x1) + r_u * (x - x1) / (x2 - x1)
    i2 = l_d * (x2 - x) / (x2 - x1) + r_d * (x - x1) / (x2 - x1)

    return i1 * (y2 - y) / (y2 - y1) + i2 * (y - y1) / (y2 - y1)


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
    original_image = cv2.imread('image20.png')
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    height, width = grayscale_image.shape

    clip_limit = 0.025
    bins_count = 256
    rows_count = 8
    columns_count = 8

    block_height = up_round(height / rows_count)
    block_width = up_round(width / columns_count)

    cdfs = np.ndarray((rows_count, columns_count, bins_count))
    centers = np.ndarray((rows_count, columns_count, 2))
    for i in range(rows_count):
        for j in range(columns_count):
            current_block, cur_block_h, cur_block_w = get_block_with_size_by_indexes(
                grayscale_image, i, j, block_height, block_width)
            histogram = calculate_histogram(current_block, bins_count)
            normalized_histogram = normalize_histogram(histogram, cur_block_h * cur_block_w)
            cdfs[i][j] = calculate_cdf(clip_histogram(normalized_histogram, clip_limit))

            center_row_index = (i + 1) * cur_block_h - cur_block_h / 2
            center_column_index = (j + 1) * cur_block_w - cur_block_w / 2
            centers[i][j] = [int(center_row_index), int(center_column_index)]

    processed_image = grayscale_image.copy()
    for i in range(height):
        for j in range(width):
            block_i, block_j = get_block_indexes_by_pixel_indexes(i, j, block_height, block_width)
            center_i, center_j = centers[block_i][block_j]

            if i <= center_i and j <= center_j:
                if block_i == 0 and block_j == 0:
                    block = np.array([block_i, block_j])
                    new_pixel_value = transform(processed_image, cdfs, block, i, j, bins_count)
                elif block_i == 0:
                    blocks = np.array([[block_i, block_j - 1], [block_i, block_j]])
                    new_pixel_value = transform_linear(processed_image, cdfs, centers, blocks, i, j, bins_count)
                elif block_j == 0:
                    blocks = np.array([[block_i - 1, block_j], [block_i, block_j]])
                    new_pixel_value = transform_linear(processed_image, cdfs, centers, blocks, i, j, bins_count)
                else:
                    blocks = np.array([[block_i - 1, block_j - 1], [block_i - 1, block_j],
                                       [block_i, block_j - 1], [block_i, block_j]])
                    new_pixel_value = transform_bilinear(processed_image, cdfs, centers, blocks, i, j, bins_count)
            elif i <= center_i and j >= center_j:
                if block_i == 0 and block_j == columns_count - 1:
                    block = np.array([block_i, block_j])
                    new_pixel_value = transform(processed_image, cdfs, block, i, j, bins_count)
                elif block_i == 0:
                    blocks = np.array([[block_i, block_j], [block_i, block_j + 1]])
                    new_pixel_value = transform_linear(processed_image, cdfs, centers, blocks, i, j, bins_count)
                elif block_j == columns_count - 1:
                    blocks = np.array([[block_i - 1, block_j], [block_i, block_j]])
                    new_pixel_value = transform_linear(processed_image, cdfs, centers, blocks, i, j, bins_count)
                else:
                    blocks = np.array([[block_i - 1, block_j], [block_i - 1, block_j + 1],
                                       [block_i, block_j], [block_i, block_j + 1]])
                    new_pixel_value = transform_bilinear(processed_image, cdfs, centers, blocks, i, j, bins_count)
            elif i >= center_i and j <= center_j:
                if block_i == rows_count - 1 and block_j == 0:
                    block = np.array([block_i, block_j])
                    new_pixel_value = transform(processed_image, cdfs, block, i, j, bins_count)
                elif block_i == rows_count - 1:
                    blocks = np.array([[block_i, block_j - 1], [block_i, block_j]])
                    new_pixel_value = transform_linear(processed_image, cdfs, centers, blocks, i, j, bins_count)
                elif block_j == 0:
                    blocks = np.array([[block_i, block_j], [block_i + 1, block_j]])
                    new_pixel_value = transform_linear(processed_image, cdfs, centers, blocks, i, j, bins_count)
                else:
                    blocks = np.array([[block_i, block_j - 1], [block_i, block_j],
                                       [block_i + 1, block_j - 1], [block_i + 1, block_j]])
                    new_pixel_value = transform_bilinear(processed_image, cdfs, centers, blocks, i, j, bins_count)
            elif i >= center_i and j >= center_j:
                if block_i == rows_count - 1 and block_j == columns_count - 1:
                    block = np.array([block_i, block_j])
                    new_pixel_value = transform(processed_image, cdfs, block, i, j, bins_count)
                elif block_i == rows_count - 1:
                    blocks = np.array([[block_i, block_j], [block_i, block_j + 1]])
                    new_pixel_value = transform_linear(processed_image, cdfs, centers, blocks, i, j, bins_count)
                elif block_j == columns_count - 1:
                    blocks = np.array([[block_i, block_j], [block_i + 1, block_j]])
                    new_pixel_value = transform_linear(processed_image, cdfs, centers, blocks, i, j, bins_count)
                else:
                    blocks = np.array([[block_i, block_j], [block_i, block_j + 1],
                                       [block_i + 1, block_j], [block_i + 1, block_j + 1]])
                    new_pixel_value = transform_bilinear(processed_image, cdfs, centers, blocks, i, j, bins_count)
            else:
                block = np.array([block_i, block_j])
                new_pixel_value = transform(processed_image, cdfs, block, i, j, bins_count)

            processed_image[i][j] = new_pixel_value

    show_image(original_image, processed_image)


main()
