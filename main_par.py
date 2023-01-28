from PIL import Image
import numpy as np
from joblib import Parallel, delayed, dump, load
import timeit
from tempfile import mkdtemp
import os


def gaussian_blur(width: int, std_dev: float) -> np.ndarray:
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


def convolution(input: np.ndarray, kernel: np.ndarray, proc: int) -> np.ndarray:
    in_height, in_width, channels = input.shape
    k_height, k_width = kernel.shape

    r_height = np.floor(k_height / 2).astype(int)
    r_width = np.floor(k_width / 2).astype(int)

    padded_input = np.pad(input, ((r_width, r_width), (r_height, r_height), (0, 0)), 'edge')

    savedir = mkdtemp()
    padded_input_path = os.path.join(savedir, 'padded_input.joblib')
    dump(padded_input, padded_input_path, compress=True)

    slice_height = np.ceil(in_height / proc).astype(int)

    output_slices = Parallel(n_jobs=proc)(delayed(convolve_slice)
                                          (padded_input_path, kernel, in_height, in_width, channels, y,
                                           y + slice_height, 0, in_width)
                                          for y in range(0, in_height, slice_height))

    output = output_slices[0]
    for i in range(1, len(output_slices)):
        if output_slices[i] is not None:
            output = np.vstack((output, output_slices[i]))

    os.remove(padded_input_path)

    output = (np.rint(output)).astype(np.uint8)
    return output


def convolve_slice(padded_input_path: str, kernel: np.ndarray, in_height: int, in_width: int, channels: int,
                   start_y: int, end_y: int, start_x: int, end_x: int) -> np.ndarray | None:
    start_x = start_x if start_x > -1 else 0
    start_y = start_y if start_y > -1 else 0
    end_x = end_x if end_x <= in_width else in_width
    end_y = end_y if end_y <= in_height else in_height

    if start_x >= in_width or start_y >= in_height or end_x <= 0 or end_y <= 0:
        return None

    padded_input = load(padded_input_path)

    k_height, k_width = kernel.shape
    output = np.zeros((end_y - start_y, end_x - start_x, channels))
    for c in range(channels):
        for i in range(start_y, end_y):
            for j in range(start_x, end_x):
                o = 0.0
                for ik in range(k_height):
                    for jk in range(k_width):
                        o += kernel[ik, jk] * padded_input[i + ik, j + jk, c]
                if o > 255:
                    o = np.uint8(255)
                output[i - start_y, j - start_x, c] = o
    return output


if __name__ == '__main__':
    img = Image.open(r"./input/test.jpg")

    input_img = np.asarray(img)
    output_img = None

    n_rep = 1
    times = []
    for p in range(5):
        processes = 2 ** p
        start_time = timeit.default_timer()
        for n in range(n_rep):
            output_img = convolution(input_img, gaussian_blur(9, 3), processes)
        end_time = timeit.default_timer()
        time = (end_time - start_time) / n_rep
        times.append((processes, time))
        print("Time (s):", time, ", Processes:", processes)

    print("Process-Times:", times)
    out_img = Image.fromarray(output_img)
    out_img.save('./output/res.jpg', 'jpeg')
