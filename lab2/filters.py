import numpy as np
import torch
import torch.nn.functional as F


def conv_nested(image, kernel):
    height_image, width_image = image.shape
    height_kernel, width_kernel = kernel.shape
    output = np.zeros((height_image, width_image))

    pad_height = height_kernel // 2
    pad_width = width_kernel // 2

    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    for i in range(height_image):
        for j in range(width_image):
            region = padded_image[i:i + height_kernel, j:j + width_kernel]
            output[i, j] = np.sum(region * kernel)

    return output


def zero_pad(image, pad_height, pad_width):
    height, width = image.shape

    padded_output = np.zeros((height + 2 * pad_height, width + 2 * pad_width))

    padded_output[pad_height:pad_height + height, pad_width:pad_width + width] = image

    return padded_output


def conv_fast(image, kernel):
    height_image, width_image = image.shape
    height_kernel, width_kernel = kernel.shape
    output = np.zeros((height_image, width_image))

    pad_height = height_kernel // 2
    pad_width = width_kernel // 2

    padded_image = zero_pad(image, pad_height, pad_width)

    flipped_kernel = np.flip(kernel)

    for i in range(height_image):
        for j in range(width_image):
            region = padded_image[i:i + height_kernel, j:j + width_kernel]
            output[i, j] = np.sum(region * flipped_kernel)

    return output


def conv_faster(image, kernel):
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    kernel_tensor = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    output_tensor = F.conv2d(image_tensor, kernel_tensor, padding=(kernel.shape[0] // 2, kernel.shape[1] // 2))

    output = output_tensor.squeeze(0).squeeze(0).numpy()

    return output


def cross_correlation(image, kernel):
    kernel_float = kernel.astype(np.float64)
    image_float = image.astype(np.float64)

    height_image, width_image = image.shape
    height_kernel, width_kernel = kernel.shape
    result = np.zeros((height_image, width_image))
    padded_image = zero_pad(image_float, height_kernel // 2, width_kernel // 2)
    sum_of_squares_kernel = np.sum(kernel_float ** 2)

    for i in range(height_image):
        for j in range(width_image):
            image_slice = padded_image[i:i + height_kernel, j:j + width_kernel]
            coefficient = np.sqrt(sum_of_squares_kernel * np.sum(image_slice ** 2))
            result[i, j] = np.sum(image_slice * kernel_float) / coefficient

    return result


def zero_mean_cross_correlation(image, kernel):
    output = np.zeros_like(image)

    kernel_zero_mean = kernel - np.mean(kernel)
    output = cross_correlation(image, kernel_zero_mean)

    return output


def normalized_cross_correlation(image, kernel):
    kernel_float = kernel.astype(np.float64)
    image_float = image.astype(np.float64)

    height_image, width_image = image.shape
    height_kernel, width_kernel = kernel.shape
    output = np.zeros((height_image, width_image))
    padded_image = zero_pad(image_float, height_kernel // 2, width_kernel // 2)

    sigma_kernel = np.std(kernel_float)
    mean_kernel = np.mean(kernel_float)
    normalized_kernel = (kernel_float - mean_kernel) / (sigma_kernel + 1e-10)
    sum_of_squares_kernel = np.sum(kernel_float ** 2)

    for i in range(height_image):
        for j in range(width_image):
            image_slice = padded_image[i:i + height_kernel, j:j + width_kernel]
            mean_image_slice = np.mean(image_slice)
            sigma_image_slice = np.std(image_slice)
            normalized_image_slice = (image_slice - mean_image_slice) / (sigma_image_slice + 1e-10)
            coefficient = np.sqrt(sum_of_squares_kernel * np.sum(image_slice ** 2))
            output[i, j] = np.sum(normalized_image_slice * normalized_kernel) / (coefficient + 1e-10)

    return output
