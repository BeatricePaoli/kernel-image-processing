from PIL import Image
import numpy as np
import timeit


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


def convolution(input: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    in_height, in_width, channels = input.shape
    k_height, k_width = kernel.shape
    r_height = np.floor(k_height / 2).astype(int)
    r_width = np.floor(k_width / 2).astype(int)

    padded_input = np.pad(input, ((r_width, r_width), (r_height, r_height), (0, 0)), 'edge')
    output = np.zeros((in_height, in_width, channels))

    for c in range(channels):
        for i in range(in_height):
            for j in range(in_width):
                o = 0.0
                for ik in range(k_height):
                    for jk in range(k_width):
                        o += kernel[ik, jk] * padded_input[i + ik, j + jk, c]
                if o > 255:
                    o = np.uint8(255)
                output[i, j, c] = o
    output = (np.rint(output)).astype(np.uint8)
    return output


if __name__ == '__main__':
    img = Image.open(r"./input/test.jpg")
    input_img = np.asarray(img)

    start_time = timeit.default_timer()
    output_img = convolution(input_img, gaussian_blur(9, 3))
    end_time = timeit.default_timer()
    print("Time (s): ", end_time - start_time)

    out_img = Image.fromarray(output_img)
    out_img.save('./output/res.jpg', 'jpeg')
