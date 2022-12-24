from PIL import Image
import numpy as np
from joblib import Parallel, delayed
import timeit


def gaussian_blur(width, std_dev) -> np.ndarray:
    assert width % 2 == 1
    output = np.zeros((width, width))
    out_sum = 0.0
    radius = np.floor(width / 2).astype(int)
    for i in range(width):
        for j in range(width):
            y = i - radius
            x = j - radius
            value = np.exp(- (y * y + x * x) / (2 * std_dev * std_dev)) / (2 * np.pi * std_dev * std_dev)
            output[i, j] = value
            out_sum += value
    output = np.divide(output, out_sum)
    return output


def convolution(arr: np.ndarray, kernel: np.ndarray, cores: int) -> np.ndarray:
    in_height, in_width, channels = arr.shape
    k_height, k_width = kernel.shape

    assert in_height >= k_height
    assert in_width >= k_width
    assert channels == 1 or channels == 3
    assert cores > 0

    r_height = np.floor(k_height / 2).astype(int)
    r_width = np.floor(k_width / 2).astype(int)

    padded_arr = pad_zeros(arr, r_height, r_width)

    # ------------------------ Metodo 1 ------------------------------
    slice_height = np.ceil(in_height / cores).astype(int)
    # n_slices = cores
    n_slices = np.round(cores / 2).astype(int)
    if n_slices < 1:
        n_slices = 1

    assert n_slices > 0

    output_slices = Parallel(n_jobs=n_slices)(delayed(convolve_pixel)
                                              (padded_arr, kernel, in_height, in_width, channels, y, y + slice_height,
                                               0, in_width)
                                              for y in range(0, in_height, slice_height))

    output = output_slices[0]
    for i in range(1, len(output_slices)):
        if output_slices[i] is not None:
            output = np.vstack((output, output_slices[i]))

    # ------------------------ Metodo 2 ------------------------------
    # slice_height = np.ceil(in_height / cores).astype(int)
    # slice_width = np.ceil(in_width / cores).astype(int)
    # start_coords = []
    # for y in range(0, in_height, slice_height):
    #     for x in range(0, in_width, slice_width):
    #         start_coords.append((x, y))
    #
    # output_slices = Parallel(n_jobs=cores)(delayed(convolve_pixel)
    #                                        (padded_arr, kernel, in_height, in_width, channels, y, y + slice_height,
    #                                         x, x + slice_width)
    #                                        for x, y in start_coords)
    #
    # rows = []
    # for y in range(cores):
    #     row = output_slices[y * cores]
    #     for x in range(1, cores):
    #         if output_slices[x + y * cores] is not None:
    #             row = np.hstack((row, output_slices[x + y * cores]))
    #     rows.append(row)
    #
    # output = rows[0].copy()
    # for y in range(1, len(rows)):
    #     output = np.vstack((output, rows[y]))

    output = (np.rint(output)).astype(np.uint8)
    return output


def convolve_pixel(padded_arr: np.ndarray, kernel: np.ndarray, in_height: int, in_width: int, channels: int,
                   start_y: int, end_y: int, start_x: int, end_x: int) -> np.ndarray | None:
    start_x = start_x if start_x > -1 else 0
    start_y = start_y if start_y > -1 else 0
    end_x = end_x if end_x <= in_width else in_width
    end_y = end_y if end_y <= in_height else in_height

    if start_x >= in_width or start_y >= in_height or end_x <= 0 or end_y <= 0:
        return None

    k_height, k_width = kernel.shape
    output = np.zeros((end_y - start_y, end_x - start_x, channels))
    for c in range(channels):
        for i in range(start_y, end_y):
            for j in range(start_x, end_x):
                o = 0.0
                for ik in range(k_height):
                    for jk in range(k_width):
                        o += kernel[ik, jk] * padded_arr[i + ik, j + jk, c]
                if o > 255:
                    o = np.uint8(255)
                output[i - start_y, j - start_x, c] = o
    return output


def pad_zeros(arr, pad_w, pad_h) -> np.ndarray:
    return np.pad(arr, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), 'constant')


if __name__ == '__main__':
    img = Image.open(r"./test.jpg")

    a = np.asarray(img)
    b = None

    # Convolution
    n_rep = 3
    for p in range(5):
        processes = 2 ** p
        start_time = timeit.default_timer()
        for n in range(n_rep):
            b = convolution(a, gaussian_blur(9, 3), processes)
        end_time = timeit.default_timer()
        print("Time (s):", (end_time - start_time) / n_rep, ", Processes:", processes)

    res = Image.fromarray(b)
    res.save('./res.jpg', 'jpeg')
