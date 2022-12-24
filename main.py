from PIL import Image
import random
import numpy as np
import timeit


def crop(arr: np.ndarray) -> np.ndarray:
    height, width, channels = arr.shape
    out_height = random.randint(1, height)
    out_width = random.randint(1, width)
    y = random.randrange(0, height - out_height)
    x = random.randrange(0, width - out_width)
    out = arr[y: y + out_height, x: x + out_width, :]
    return out


def box_blur(width: int) -> np.ndarray:
    assert width % 2 == 1
    return np.ones((width, width)) / 9


def gaussian_blur(width: int, std_dev: float) -> np.ndarray:
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


def convolution(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    in_height, in_width, channels = arr.shape
    k_height, k_width = kernel.shape
    r_height = np.floor(k_height / 2).astype(int)
    r_width = np.floor(k_width / 2).astype(int)

    assert in_height >= k_height
    assert in_width >= k_width
    assert channels == 3

    padded_arr = pad_zeros(arr, r_height, r_width)
    output = np.zeros((in_height, in_width, channels))

    for c in range(channels):
        for i in range(in_height):
            for j in range(in_width):
                o = 0.0
                for ik in range(k_height):
                    for jk in range(k_width):
                        o += kernel[ik, jk] * padded_arr[i + ik, j + jk, c]
                if o > 255:
                    o = np.uint8(255)
                output[i, j, c] = o
    output = (np.rint(output)).astype(np.uint8)
    return output


def pad_zeros(arr: np.ndarray, pad_w: int, pad_h: int) -> np.ndarray:
    return np.pad(arr, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), 'constant')


def double_brightness(val: np.uint8):
    out = val * 2
    if out > 255:
        return np.uint8(255)
    return out


if __name__ == '__main__':
    img = Image.open(r"./test.jpg")
    # img_hsv = img.convert('HSV')

    # a = np.asarray(img_hsv)
    a = np.asarray(img)

    # print(a)
    # print(a.shape)

    # Brightness
    # b = np.floor_divide(a, 2)
    # b = np.vectorize(double_brightness)(a)

    # Hue Shift (HSV)
    # b = a
    # b[..., 0] = (b[..., 0] + 70) % 255

    # Grayscale
    # b = np.dot(a[..., :3], [0.2989, 0.5870, 0.1140])
    # b = b.astype(np.uint8)

    # Flip
    # b = np.flip(a, axis=0)
    # b = np.flip(a, axis=1)

    # Padding
    # b = pad_zeros(a, 2, 2)

    # Convolution
    # b = convolution(a, box_blur(3))
    start_time = timeit.default_timer()
    b = convolution(a, gaussian_blur(9, 3))
    end_time = timeit.default_timer()
    print("Time (s): ", end_time - start_time)

    # Crop
    # b = crop(a)

    # Rotation
    # b = np.rot90(a, -1, (0, 1))

    # print(b)
    # print(b.shape)

    res = Image.fromarray(b)
    # res = Image.fromarray(b, 'HSV').convert('RGB')
    res.save('./res.jpg', 'jpeg')
